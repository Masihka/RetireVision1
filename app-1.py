"""
floorplan3d.py - Floor plan to 3D model
=======================================
Single file. Deploy by dropping this plus requirements.txt into a repo; the
filename does not matter and there is no local import to get wrong.

    streamlit run floorplan3d.py

The pipeline lives at module level and the UI lives in main(), so this file can
also be imported and tested headlessly:

    import floorplan3d as f
    f.selftest()

TWO MODES, because two kinds of drawing need different treatment.

  LINE DRAWING - monochrome CAD-style plans. Walls are found by a morphological
  sieve that keeps long horizontal and vertical runs, and rooms are the enclosed
  components of free space.

  COLOUR-FILLED - real-estate marketing plans, where rooms carry flat colour
  fills. Three properties of these defeat the line-drawing mode outright, and
  each is handled explicitly below:

    1. The image usually holds a floor plan AND a site plan side by side.
       Feeding both in treats the site boundary as walls, so panels are detected
       from the column profile of ink and the user picks one.

    2. Windows are drawn as gaps in the outer wall, leaving short wall segments
       between them. On a real listing plan, min_line_px=25 rejected those
       segments, free space leaked out of the envelope, and room detection
       returned ZERO rooms at every other setting swept. min_line_px near 5 with
       close_px near 9 seals windows while leaving doorways open.

    3. Room labels are as thick as the walls. Measured on that plan: wall
       strokes ~3 px, text strokes ~4 px. Thickness cannot separate them, and
       the closing fuses letters into word-sized blobs that clear any sane
       min_blob_px, so labels extrude as walls. The fix uses the colour coding
       itself: text sits INSIDE a filled room, walls sit ON its boundary, so ink
       strictly interior to a fill is discarded.

Validated on a 4-bedroom listing plan: 10.13 x 25.42 m envelope, 180.4 m2 of
floor across 11 rooms, scale fitted to 2.3% scatter across five labelled rooms.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import trimesh
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

MAX_FACES = 150_000


# =========================================================================== #
#  PART 1 - parameters
# =========================================================================== #


@dataclass(frozen=True)
class WallParams:
    dark_ink: bool = True
    adaptive: bool = False
    block: int = 35
    C: int = 10
    axis_aligned: bool = True
    min_line_px: int = 40
    min_thick_px: int = 0
    close_px: int = 5
    min_blob_px: int = 200


@dataclass(frozen=True)
class GeomParams:
    px_per_m: float = 100.0
    eps_rel: float = 0.002
    wall_height_m: float = 2.7
    floor_thickness_m: float = 0.12
    min_wall_area_m2: float = 0.02
    min_room_area_m2: float = 1.5


@dataclass(frozen=True)
class Opening:
    x: float
    y: float
    width: float = 0.9
    height: float = 2.05
    sill: float = 0.0
    angle_deg: float = 0.0
    depth: float = 1.0


# =========================================================================== #
#  PART 2 - binarisation and wall isolation
# =========================================================================== #


def binarize(gray: np.ndarray, p: WallParams) -> np.ndarray:
    """uint8 {0,255} mask, 255 = ink.

    Otsu is the default because printed plans are strongly bimodal; on the
    validated listing plan it chose 145, cleanly above the lightest fill and
    well below the ink. Adaptive is the fallback for photographs and shaded
    scans where the background level drifts across the page.
    """
    if gray.ndim != 2:
        raise ValueError("binarize expects a single-channel image")
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    mode = cv2.THRESH_BINARY_INV if p.dark_ink else cv2.THRESH_BINARY
    if p.adaptive:
        blk = max(3, int(p.block) | 1)
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, blk, float(p.C))
    _, bw = cv2.threshold(g, 0, 255, mode | cv2.THRESH_OTSU)
    return bw


def _remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def extract_walls(bw: np.ndarray, p: WallParams) -> np.ndarray:
    """Isolate wall structure from an ink mask.

    The sieve is an opening with a 1 x L horizontal element and an L x 1 vertical
    element, so only runs at least L pixels long in one of those directions
    survive. That is the structure of an orthogonal wall and not the structure of
    glyphs, dimension arrows, furniture icons or hatching. It also erases
    diagonal and curved walls, so set axis_aligned=False for those.

    Ordering was decided by measurement. min_thick_px runs BEFORE close_px:
    placed after, it does nothing, because the closing has already fused thin
    annotations into thick blobs. On a plan hatched with 5 px strokes the hatched
    room read 39.6 m2 for every value when the opening ran last, against 45.2 m2
    true; running it first at 9 px recovered 44.3 m2. The cost is that a hollow
    double-line wall is still two thin strokes at that point, so any non-zero
    min_thick_px erases it. Leave it at 0 unless the walls are solid.

    close_px is the most consequential parameter and it fails silently when
    overdriven. Set it a little above the drawn wall thickness and no higher: on
    a plan with a dimension line 25 px clear of the envelope, raising it from 21
    to 24 bridged the line into the wall and inflated the footprint by 4.8%.
    Nothing downstream can undo that.
    """
    wall = bw
    if p.axis_aligned:
        L = max(3, int(p.min_line_px))
        wall = cv2.bitwise_or(
            cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))),
            cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))),
        )
    if p.min_thick_px > 1:
        k = int(p.min_thick_px)
        wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if p.close_px > 0:
        k = int(p.close_px)
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return _remove_small_components(wall, int(p.min_blob_px))


def _run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    a = (mask > 0).astype(np.int8)
    if axis == 0:
        a = a.T
    a = np.pad(a, ((0, 0), (1, 1)))
    d = np.diff(a, axis=1)
    s, e = np.argwhere(d == 1), np.argwhere(d == -1)
    if len(s) != len(e):
        return np.empty(0, dtype=np.int64)
    return (e[:, 1] - s[:, 1]).astype(np.int64)


def estimate_wall_thickness_px(wall: np.ndarray) -> float:
    """Modal scanline run length.

    Scanning rows crosses every vertical wall at exactly its thickness, and
    columns do the same for horizontal walls. Wall lengths also enter the
    histogram but spread over many values, whereas thickness piles into a sharp
    mode. Bias measured against synthetic ground truth is a fixed +1 to +2 px
    from the pre-blur widening the ink edge, not a proportional error.

    Rejected alternative: the distance transform gives t = 4 E[DT] for an
    infinite slab, but corners push the medial surface outward and it ran 28-49%
    high on the same plans.
    """
    if not np.any(wall):
        return 0.0
    L = np.concatenate([_run_lengths(wall, 0), _run_lengths(wall, 1)])
    L = L[L > 0]
    return float(np.bincount(L).argmax()) if L.size else 0.0


# =========================================================================== #
#  PART 3 - vectorisation
# =========================================================================== #


def _px_to_world(contour: np.ndarray, px_per_m: float, img_h: int) -> np.ndarray:
    """Pixel coordinates to a right-handed metric frame.

    Image rows increase downwards, world y increases upwards. Omitting the flip
    mirrors the building, which is invisible on a symmetric plan and unfixable
    downstream.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    s = 1.0 / float(px_per_m)
    return np.column_stack([pts[:, 0] * s, (img_h - pts[:, 1]) * s])


def _simplify(contour: np.ndarray, eps_rel: float) -> np.ndarray:
    """Douglas-Peucker with tolerance proportional to perimeter, so behaviour is
    stable across plan sizes and DPI."""
    return cv2.approxPolyDP(contour, max(float(eps_rel) * cv2.arcLength(contour, True), 1e-9), True)


def _clean(poly: Polygon) -> list[Polygon]:
    fixed = poly if poly.is_valid else poly.buffer(0)
    if fixed.is_empty:
        return []
    if isinstance(fixed, Polygon):
        return [fixed]
    return [g for g in getattr(fixed, "geoms", []) if isinstance(g, Polygon)]


def mask_to_polygons(mask: np.ndarray, g: GeomParams, min_area_m2: float | None = None) -> MultiPolygon:
    """Trace a binary mask into metric polygons, holes included.

    RETR_CCOMP gives a two-level hierarchy: parent == -1 rows are outer
    boundaries, their children are holes. Pairing them is what makes a courtyard
    or a stair void come out as a void rather than a solid block.
    """
    min_area = g.min_wall_area_m2 if min_area_m2 is None else min_area_m2
    h = mask.shape[0]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return MultiPolygon()
    hierarchy = hierarchy[0]

    polys: list[Polygon] = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue
        shell = _simplify(cnt, g.eps_rel)
        if len(shell) < 3:
            continue
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            hc = _simplify(contours[child], g.eps_rel)
            if len(hc) >= 3:
                ring = _px_to_world(hc, g.px_per_m, h)
                if Polygon(ring).area >= min_area:
                    holes.append(ring)
            child = hierarchy[child][0]
        cand = Polygon(_px_to_world(shell, g.px_per_m, h), holes)
        polys.extend(q for q in _clean(cand) if q.area >= min_area)

    if not polys:
        return MultiPolygon()
    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        merged = MultiPolygon([merged])
    return MultiPolygon([orient(p, 1.0) for p in merged.geoms])


def rooms_from_walls(wall: np.ndarray, g: GeomParams) -> list[tuple[Polygon, float]]:
    """Enclosed free space, for line-drawing mode.

    Components touching the image border are exterior. This depends on the wall
    loop being closed, so a doorway drawn as a full gap merges two rooms and a
    break in the envelope leaks every room to the outside.
    """
    free = (wall == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    min_px = g.min_room_area_m2 * g.px_per_m ** 2
    out = []
    for i in range(1, n):
        if i in border or stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        for poly in mask_to_polygons(comp, g, min_area_m2=g.min_room_area_m2).geoms:
            out.append((poly, float(poly.area)))
    return sorted(out, key=lambda t: -t[1])


# =========================================================================== #
#  PART 4 - colour-filled plans
# =========================================================================== #


def detect_panels(gray: np.ndarray, ink_thr: int = 100, min_panel_px: int = 40,
                  blob_px: int = 41, margin: int = 15) -> list[tuple[int, int, int, int]]:
    """Find drawing panels as (x0, y0, x1, y1), largest first.

    Listing images pair a floor plan with a site plan, and feeding both in treats
    the site boundary as walls.

    Two stages, because one does not work. First a morphological sieve removes
    text: titles, disclaimers and room labels are short runs, walls and site
    boundaries are long ones. Only then is the survivor closed into per-drawing
    blobs. Closing raw ink instead forces an impossible choice, since the kernel
    must exceed a ~30 px doorway gap to hold a plan together but stay under the
    ~25 px gap to the title block to avoid swallowing it. Sieving first removes
    the constraint entirely.

    Rejected earlier: row and column profiles, which failed twice. Windows on
    opposite walls line up, so six image rows carried no ink and the house split
    in two at y=680, cropping half the building; bridging that gap then let a
    full-width disclaimer line fuse both panels into one 1178 px band.

    The box is grown by `margin` because the sieve trims wall ends: on the
    validated plan the raw box stopped 11 px inside the right wall.
    """
    sp = WallParams(min_line_px=30, close_px=0, min_blob_px=200)
    struct = extract_walls(binarize(gray, sp), sp)
    blob = cv2.morphologyEx(struct, cv2.MORPH_CLOSE,
                            np.ones((int(blob_px), int(blob_px)), np.uint8))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(blob, 8)
    H, W = gray.shape
    out = []
    for i in range(1, n):
        x, y = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        w, h = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if w < min_panel_px or h < min_panel_px:
            continue
        out.append((int(st[i, cv2.CC_STAT_AREA]),
                    (max(0, x - margin), max(0, y - margin),
                     min(W - 1, x + w - 1 + margin), min(H - 1, y + h - 1 + margin))))
    out.sort(key=lambda t: -t[0])
    return [b for _a, b in out]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed regions, leaving the outline exact.

    Flood the exterior from a corner and OR back whatever was not reached. Used
    to close the glyph-shaped holes room labels punch in a colour fill. A
    morphological closing cannot do this job: a kernel wide enough to swallow a
    word also rounds every corner and bridges the ~3 px walls between rooms,
    which in testing deleted every internal wall in the building.
    """
    p = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = p.copy()
    cv2.floodFill(ff, np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(p, cv2.bitwise_not(ff))[1:-1, 1:-1]


def dominant_fills(bgr: np.ndarray, max_colors: int = 6, merge_dist: int = 16,
                   min_frac: float = 0.005) -> list[tuple[tuple[int, int, int], int]]:
    """Detect flat room-fill colours, most common first.

    Colours are quantised to 8 levels per channel and merged greedily: a
    candidate within merge_dist of an already-kept colour is a shade of it, not a
    new room type. The default of 16 is bounded on both sides by real cases: the
    two beiges on the validated plan sit 8 apart and MUST merge, while its beige
    and grey sit exactly 24 apart and must not - at 24 the bedrooms were absorbed
    into the garage and vanished from the model. Page white and ink are excluded by luminance. On the validated
    plan this returns exactly four fills - living, bedroom, wet area, garage -
    and correctly merges the two near-identical beiges that would otherwise
    double-count every bedroom.
    """
    q = bgr.reshape(-1, 3) // 8 * 8
    uniq, cnt = np.unique(q, axis=0, return_counts=True)
    n = len(q)
    keep: list[tuple[tuple[int, int, int], int]] = []
    for i in np.argsort(-cnt):
        c, k = uniq[i], int(cnt[i])
        if k < min_frac * n:
            break
        if c.min() >= 236 or c.max() <= 40:
            continue
        if any(np.abs(c.astype(int) - np.array(p, int)).max() <= merge_dist for p, _ in keep):
            continue
        keep.append((tuple(int(v) for v in c), k))
        if len(keep) >= max_colors:
            break
    return keep


def colour_mask(bgr: np.ndarray, key: Sequence[int], tol: int = 14) -> np.ndarray:
    d = np.abs(bgr.astype(np.int16) - np.array(key, np.int16)).max(axis=2)
    return (d <= tol).astype(np.uint8) * 255


def enclosed_uncoloured(bgr: np.ndarray, min_area: int = 250, white_thr: int = 236) -> np.ndarray:
    """Rooms drawn white rather than filled - kitchens, robes, entries.

    White also covers the page, so only white components that do not touch the
    image border count as rooms.
    """
    w = (bgr.min(axis=2) >= white_thr).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(w, 4)
    edge = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    keep = [i for i in range(1, n) if i not in edge and st[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(lab, keep).astype(np.uint8) * 255


def extract_walls_colour(gray: np.ndarray, fill: np.ndarray, wp: WallParams,
                         band_px: int = 13, min_blob_px: int = 250) -> np.ndarray:
    """Wall ink, restricted to a band around the room fills.

    Text sits inside a filled room; walls sit on its boundary. Discarding ink
    strictly interior to the filled region therefore removes labels that no
    thickness or blob filter can touch - on the validated plan wall strokes
    measured ~3 px and text strokes ~4 px, so the text was the THICKER of the
    two. band_px is how far inside a room wall ink may still be kept.
    """
    interior = cv2.erode(fill_holes(fill), np.ones((int(band_px), int(band_px)), np.uint8))
    w = extract_walls(binarize(gray, wp), wp)
    return _remove_small_components(cv2.bitwise_and(w, cv2.bitwise_not(interior)), int(min_blob_px))


def room_components(bgr: np.ndarray, keys: Sequence[Sequence[int]], tol: int = 14,
                    min_area_px: int = 500) -> list[dict]:
    """Room regions in PIXEL space, before any scale is known.

    Returned before metric conversion on purpose: scale is calibrated from these
    pixel bounding boxes against dimensions printed on the plan, so the regions
    have to exist first.
    """
    out = []
    for ki, key in enumerate(keys):
        m = cv2.morphologyEx(colour_mask(bgr, key, tol), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = fill_holes(m)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 4)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_area_px:
                continue
            out.append({
                "fill": ki,
                "key": tuple(int(v) for v in key),
                "area_px": int(st[i, cv2.CC_STAT_AREA]),
                "x": int(st[i, cv2.CC_STAT_LEFT]), "y": int(st[i, cv2.CC_STAT_TOP]),
                "w": int(st[i, cv2.CC_STAT_WIDTH]), "h": int(st[i, cv2.CC_STAT_HEIGHT]),
                "mask": np.where(lab == i, 255, 0).astype(np.uint8),
            })
    out.sort(key=lambda r: -r["area_px"])
    return out


def fit_scale(refs: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, int]:
    """Least-squares px/m from rooms with printed dimensions.

    Each reference contributes two independent estimates, one per axis. Returns
    (mean, sd, n). On the validated plan five labelled rooms gave 29.43 px/m with
    0.67 sd, i.e. 2.3%; the 0.1 m rounding of printed labels alone accounts for
    +/-1.5% on a 3.3 m room, so little is left unexplained.

    Exclude any room whose two printed numbers disagree with each other. On the
    validated plan the garage's printed 6.4 m implied 28.9 px/m while its 5.4 m
    implied 25.7, a 12% split; one of the two does not describe the coloured
    region and the image cannot say which, so the room was dropped rather than
    allowed to bias the fit.
    """
    s: list[float] = []
    for pw, ph, mw, mh in refs:
        if mw and mw > 0:
            s.append(pw / mw)
        if mh and mh > 0:
            s.append(ph / mh)
    if not s:
        return 0.0, 0.0, 0
    a = np.array(s)
    return float(a.mean()), float(a.std()), len(a)


# =========================================================================== #
#  PART 5 - extrusion
# =========================================================================== #


def _extrude(poly: Polygon, height: float) -> trimesh.Trimesh | None:
    """Prism extrusion with a triangulator fallback.

    earcut (mapbox_earcut, ISC) is preferred over the `triangle` engine, which
    wraps Shewchuk's Triangle and is not licensed for commercial use.
    """
    for kwargs in ({"engine": "earcut"}, {}):
        try:
            return trimesh.creation.extrude_polygon(poly, height=float(height), **kwargs)
        except Exception:
            continue
    return None


def _room_palette(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 4), dtype=np.uint8)
    hues = (np.arange(n) * (180.0 / max(n, 1))).astype(np.uint8).reshape(-1, 1, 1)
    hsv = np.concatenate([hues, np.full_like(hues, 90), np.full_like(hues, 225)], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return np.hstack([rgb, np.full((n, 1), 255, np.uint8)])


def build_scene(walls: MultiPolygon, rooms: Sequence[tuple[Polygon, Sequence[int]]],
                g: GeomParams, include_floor: bool = True) -> trimesh.Scene:
    """Wall prisms plus floor slabs.

    Walls occupy z in [0, height]; slabs occupy [-thickness, 0], so they meet
    exactly at z=0 and no seam opens at the joint.
    """
    scene = trimesh.Scene()
    for i, poly in enumerate(getattr(walls, "geoms", [])):
        m = _extrude(poly, g.wall_height_m)
        if m is None or m.is_empty:
            continue
        m.visual.face_colors = [232, 230, 226, 255]
        scene.add_geometry(m, node_name=f"wall_{i:03d}", geom_name=f"wall_{i:03d}")
    if include_floor:
        for i, (poly, rgb) in enumerate(rooms):
            m = _extrude(poly, g.floor_thickness_m)
            if m is None or m.is_empty:
                continue
            m.apply_translation([0.0, 0.0, -g.floor_thickness_m])
            m.visual.face_colors = [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]
            scene.add_geometry(m, node_name=f"room_{i:03d}", geom_name=f"room_{i:03d}")
    return scene


def cut_openings(scene: trimesh.Scene, openings: Iterable[Opening]) -> tuple[trimesh.Scene, list[str]]:
    """Subtract door/window boxes using the manifold engine.

    Boolean CSG on meshes from simplified contours can fail on degenerate input,
    so each subtraction is attempted independently and failures are reported
    rather than raised.
    """
    ops = list(openings)
    warnings: list[str] = []
    if not ops:
        return scene, warnings
    cutters = []
    for o in ops:
        box = trimesh.creation.box(extents=[float(o.width), float(o.depth), float(o.height)])
        T = trimesh.transformations.rotation_matrix(np.radians(float(o.angle_deg)), [0, 0, 1])
        T[:3, 3] = [float(o.x), float(o.y), float(o.sill) + float(o.height) / 2.0]
        box.apply_transform(T)
        cutters.append(box)
    out = trimesh.Scene()
    for name, geom in scene.geometry.items():
        if not name.startswith("wall"):
            out.add_geometry(geom, node_name=name, geom_name=name)
            continue
        mesh = geom
        for j, cut in enumerate(cutters):
            try:
                res = trimesh.boolean.difference([mesh, cut], engine="manifold")
                if res is not None and not res.is_empty:
                    mesh = res
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"opening {j} vs {name}: {type(exc).__name__}")
        mesh.visual.face_colors = [232, 230, 226, 255]
        out.add_geometry(mesh, node_name=name, geom_name=name)
    return out, warnings


def scene_stats(scene: trimesh.Scene) -> dict:
    b = scene.bounds
    return {
        "parts": len(scene.geometry),
        "vertices": sum(len(g.vertices) for g in scene.geometry.values()),
        "faces": sum(len(g.faces) for g in scene.geometry.values()),
        "bbox_m": None if b is None else (b[1] - b[0]).round(3).tolist(),
    }


# =========================================================================== #
#  PART 6 - demo plan and self-test
# =========================================================================== #


def synthetic_plan(px_per_m: float = 50.0) -> np.ndarray:
    """10 x 7 m two-room plan with a courtyard void and decoy annotation.

    cv2.rectangle strokes are centred on the path, so the 10.0 x 7.0 centreline
    box with 0.2 m walls has a 10.2 x 7.2 m OUTER envelope, and the envelope is
    what contour tracing recovers.
    """
    s = px_per_m
    H, W = int(7 * s) + 80, int(10 * s) + 80
    img = np.full((H, W), 255, np.uint8)
    t = max(2, int(round(0.2 * s)))
    x0, y0 = 40, 40
    x1, y1 = x0 + int(10 * s), y0 + int(7 * s)
    cv2.rectangle(img, (x0, y0), (x1, y1), 0, t)
    cv2.line(img, (x0 + int(6 * s), y0), (x0 + int(6 * s), y1), 0, t)
    cx, cy = x0 + int(2 * s), y0 + int(2 * s)
    cv2.rectangle(img, (cx, cy), (cx + int(2 * s), cy + int(2 * s)), 0, t)
    cv2.putText(img, "BEDROOM 3.5x4.0", (x0 + 20, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    cv2.circle(img, (x1 - 60, y0 + 60), 18, 0, 1)
    return img


def selftest() -> int:
    """Verify the line-drawing path against a plan with known dimensions."""
    px = 50.0
    img = synthetic_plan(px)
    wp = WallParams(min_line_px=30, close_px=7, min_blob_px=150)
    gp = GeomParams(px_per_m=px, eps_rel=0.004, min_room_area_m2=1.0)
    wall = extract_walls(binarize(img, wp), wp)
    t_m = estimate_wall_thickness_px(wall) / px
    polys = mask_to_polygons(wall, gp)
    mnx, mny, mxx, mxy = polys.bounds
    w, h = mxx - mnx, mxy - mny
    holes = sum(len(p.interiors) for p in polys.geoms)
    rooms = rooms_from_walls(wall, gp)
    scene = build_scene(polys, [(p, (200, 200, 200)) for p, _ in rooms], gp)
    glb = scene.export(file_type="glb")
    checks = {
        "footprint 10.20 m": abs(w - 10.2) < 0.15,
        "footprint 7.20 m": abs(h - 7.2) < 0.15,
        "thickness 0.20 m": abs(t_m - 0.2) < 0.05,
        "void preserved": holes >= 3,
        "rooms found": len(rooms) >= 2,
        "glb exported": len(glb) > 1000,
    }
    print(f"footprint {w:.2f} x {h:.2f} m | thickness {t_m:.3f} m | {len(rooms)} rooms")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    return 0 if all(checks.values()) else 1


# =========================================================================== #
#  PART 7 - Streamlit UI
# =========================================================================== #


@st.cache_data(show_spinner=False)
def _decode(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False)
def _panels(gray: np.ndarray, ink_thr: int, min_panel: int, blob_px: int):
    return detect_panels(gray, ink_thr, min_panel, blob_px)


@st.cache_data(show_spinner=False)
def _fills(bgr: np.ndarray, max_colors: int, merge: int, min_frac: float):
    return dominant_fills(bgr, max_colors, merge, min_frac)


def overlay(bgr: np.ndarray, wall: np.ndarray) -> np.ndarray:
    tint = bgr.copy()
    tint[wall > 0] = (40, 40, 220)
    return cv2.cvtColor(cv2.addWeighted(bgr, 0.42, tint, 0.58, 0), cv2.COLOR_BGR2RGB)


def numbered(bgr: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for i, r in enumerate(regions):
        cv2.rectangle(out, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 0, 255), 1)
        cv2.putText(out, str(i + 1), (r["x"] + 4, r["y"] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def scene_to_plotly(scene: trimesh.Scene) -> go.Figure:
    fig = go.Figure()
    for name, geom in scene.geometry.items():
        v, f = geom.vertices, geom.faces
        try:
            r, g, b = geom.visual.face_colors[0][:3]
        except Exception:
            r, g, b = 190, 190, 195
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=f"rgb({r},{g},{b})", flatshading=True, name=name, hoverinfo="name",
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.08, roughness=0.9),
            lightposition=dict(x=100, y=200, z=300)))
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)",
                   zaxis_title="z (m)", camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))))
    return fig


def main() -> None:
    st.set_page_config(page_title="Floor plan to 3D", layout="wide")

    st.sidebar.header("Source")
    up = st.sidebar.file_uploader("Floor plan image",
                                  type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
    demo = st.sidebar.toggle("Use the built-in demo plan", value=up is None)

    if demo:
        bgr = cv2.cvtColor(synthetic_plan(50.0), cv2.COLOR_GRAY2BGR)
    elif up is not None:
        bgr = _decode(up.getvalue())
    else:
        st.title("Floor plan to 3D")
        st.info("Upload a floor plan, or switch on the demo plan to see the pipeline run.")
        return

    mode = st.sidebar.radio(
        "Drawing type", ["Colour-filled plan", "Line drawing"],
        index=1 if demo else 0,
        help="Colour-filled suits real-estate listing plans where rooms carry flat colour. "
             "Line drawing suits monochrome CAD-style plans.")
    colour_mode = mode == "Colour-filled plan"

    # ---- panel selection --------------------------------------------------
    full_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    panels = _panels(full_gray, 100, 40, 41)
    st.sidebar.header("Panel")
    if len(panels) > 1:
        st.sidebar.caption(f"{len(panels)} drawings detected. Listing images pair a floor plan "
                           "with a site plan; pick the floor plan.")
        idx = st.sidebar.radio("Which drawing",
                               list(range(len(panels))),
                               format_func=lambda i: f"{i + 1}: {panels[i][2] - panels[i][0]}"
                                                     f" x {panels[i][3] - panels[i][1]} px",
                               horizontal=True)
    else:
        idx = 0
    auto = panels[idx] if panels else (0, 0, bgr.shape[1] - 1, bgr.shape[0] - 1)
    with st.sidebar.expander("Adjust crop"):
        x0 = st.number_input("left", 0, bgr.shape[1] - 1, int(auto[0]))
        y0 = st.number_input("top", 0, bgr.shape[0] - 1, int(auto[1]))
        x1 = st.number_input("right", 1, bgr.shape[1], int(auto[2]) + 1)
        y1 = st.number_input("bottom", 1, bgr.shape[0], int(auto[3]) + 1)
    PAD = 20
    crop = cv2.copyMakeBorder(bgr[int(y0):int(y1), int(x0):int(x1)], PAD, PAD, PAD, PAD,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ---- wall detection ---------------------------------------------------
    st.sidebar.header("Wall detection")
    adaptive = st.sidebar.toggle("Adaptive threshold", value=False,
                                 help="For photos or scans with uneven lighting. Otsu is better for clean exports.")
    block = st.sidebar.slider("Adaptive window (px)", 11, 151, 35, 2, disabled=not adaptive)
    C = st.sidebar.slider("Adaptive bias", -20, 40, 10, disabled=not adaptive)
    axis_aligned = st.sidebar.toggle("Orthogonal walls only", value=True,
                                     help="Keeps horizontal and vertical runs only. Switch off for diagonal or curved walls.")
    min_line_px = st.sidebar.slider(
        "Shortest wall run (px)", 3, 200, 5 if colour_mode else 40, disabled=not axis_aligned,
        help="Drop to about 5 when windows are drawn as gaps in the outer wall: at 40 the short "
             "segments between them are rejected and room detection returns nothing.")
    min_thick_px = st.sidebar.slider(
        "Thinnest wall (px)", 0, 31, 0,
        help="Removes hatching and stair treads. Runs before gap closing, so leave at 0 for hollow "
             "double-line walls or it erases them. Useless when labels are as thick as walls.")
    close_px = st.sidebar.slider(
        "Gap closing (px)", 0, 41, 9 if colour_mode else 5,
        help="A little above the drawn wall thickness and no higher. Too high bridges dimension "
             "lines into the walls, which nothing downstream can undo.")
    min_blob_px = st.sidebar.slider("Smallest kept blob (px)", 0, 5000, 120 if colour_mode else 200, 20)

    wp = WallParams(dark_ink=True, adaptive=adaptive, block=block, C=C, axis_aligned=axis_aligned,
                    min_line_px=min_line_px, min_thick_px=min_thick_px, close_px=close_px,
                    min_blob_px=min_blob_px)

    # ---- colour fills -----------------------------------------------------
    regions: list[dict] = []
    fill_union = np.zeros(gray.shape, np.uint8)
    if colour_mode:
        st.sidebar.header("Room fills")
        tol = st.sidebar.slider("Colour tolerance", 4, 40, 14,
                                help="How far a pixel may sit from a fill's centre colour and still belong to it.")
        merge_dist = st.sidebar.slider(
            "Colour separation", 6, 40, 16,
            help="Fills closer than this are treated as shades of one colour. Too high merges "
                 "distinct room types together; too low splits one room type into several.")
        found = _fills(crop, 6, merge_dist, 0.005)
        chosen = []
        for i, (key, cnt) in enumerate(found):
            pct = 100 * cnt / (crop.shape[0] * crop.shape[1])
            r, g_, b = key[2], key[1], key[0]
            if st.sidebar.checkbox(f"Fill {i + 1} - {pct:.1f}% of plan", value=True, key=f"fill{i}"):
                chosen.append(key)
            st.sidebar.markdown(
                f"<div style='height:10px;margin:-8px 0 8px 26px;background:rgb({r},{g_},{b});"
                "border:1px solid #999;border-radius:2px'></div>", unsafe_allow_html=True)
        if not found:
            st.sidebar.warning("No flat fills found. This looks like a line drawing.")
        inc_white = st.sidebar.toggle("Include uncoloured enclosed rooms", value=False,
                                      help="Kitchens, robes and entries are often drawn white. "
                                           "Always used to place walls; this adds them as floors too.")
        band_px = st.sidebar.slider("Wall band (px)", 5, 31, 13,
                                    help="How far inside a room wall ink is still kept. This is what "
                                         "removes room labels, which no thickness filter can.")
        wall_blob2 = st.sidebar.slider("Wall blob floor (px)", 0, 2000, 250, 10)

        for k in chosen:
            fill_union = cv2.bitwise_or(fill_union, colour_mask(crop, k, tol))
        white_rooms = enclosed_uncoloured(crop)
        fill_union = cv2.bitwise_or(fill_union, white_rooms)
        keys = list(chosen) + ([None] if inc_white else [])
        regions = room_components(crop, chosen, tol, 500)
        if inc_white:
            n, lab, stt, _ = cv2.connectedComponentsWithStats(white_rooms, 4)
            for i in range(1, n):
                if stt[i, cv2.CC_STAT_AREA] < 500:
                    continue
                regions.append({"fill": -1, "key": (245, 245, 245),
                                "area_px": int(stt[i, cv2.CC_STAT_AREA]),
                                "x": int(stt[i, cv2.CC_STAT_LEFT]), "y": int(stt[i, cv2.CC_STAT_TOP]),
                                "w": int(stt[i, cv2.CC_STAT_WIDTH]), "h": int(stt[i, cv2.CC_STAT_HEIGHT]),
                                "mask": np.where(lab == i, 255, 0).astype(np.uint8)})
            regions.sort(key=lambda r: -r["area_px"])
        wall = extract_walls_colour(gray, fill_union, wp, band_px, wall_blob2)
    else:
        wall = extract_walls(binarize(gray, wp), wp)

    # ---- scale ------------------------------------------------------------
    st.sidebar.header("Scale")
    method = st.sidebar.radio("Calibrate by",
                              ["Printed room sizes", "Overall width", "Pixels per metre"],
                              index=0 if colour_mode and regions else 1)
    ys, xs = np.nonzero(wall)
    span_px = float(xs.max() - xs.min()) if xs.size else float(gray.shape[1])
    px_per_m, sd, nref = 0.0, 0.0, 0
    if method == "Printed room sizes" and regions:
        st.sidebar.caption("Fill the table on the Scale tab using the numbered regions.")
        refs = st.session_state.get("scale_refs")
        if refs is not None and len(refs):
            rows = []
            for r in refs.to_dict("records"):
                i = int(r.get("Region") or 0) - 1
                if 0 <= i < len(regions):
                    rows.append((regions[i]["w"], regions[i]["h"],
                                 float(r.get("Width (m)") or 0), float(r.get("Depth (m)") or 0)))
            px_per_m, sd, nref = fit_scale(rows)
        if px_per_m <= 0:
            px_per_m = 50.0
    elif method == "Overall width":
        known = st.sidebar.number_input("Real width of the plan (m)", 1.0, 500.0, 10.2, 0.1,
                                        help=f"Detected walls span {span_px:.0f} px horizontally.")
        px_per_m = span_px / max(known, 1e-6)
    else:
        px_per_m = st.sidebar.number_input("Pixels per metre", 1.0, 2000.0, 50.0, 1.0)
    st.sidebar.caption(f"{px_per_m:.2f} px/m" + (f"  sd {sd:.2f} ({100*sd/px_per_m:.1f}%), n={nref}" if nref else ""))

    st.sidebar.header("3D")
    wall_h = st.sidebar.slider("Wall height (m)", 2.0, 6.0, 2.55, 0.05)
    floor_t = st.sidebar.slider("Floor thickness (m)", 0.0, 0.5, 0.10, 0.01)
    eps_rel = st.sidebar.select_slider("Contour simplification",
                                       [0.0005, 0.001, 0.0015, 0.002, 0.004, 0.008], value=0.0015)
    min_room = st.sidebar.slider("Smallest room (m²)", 0.2, 20.0, 0.5, 0.1)
    show_floor = st.sidebar.toggle("Show floor slabs", value=True)

    gp = GeomParams(px_per_m=px_per_m, eps_rel=eps_rel, wall_height_m=wall_h,
                    floor_thickness_m=floor_t, min_wall_area_m2=0.04, min_room_area_m2=min_room)

    # ---- vectorise --------------------------------------------------------
    walls = mask_to_polygons(wall, gp)
    if colour_mode:
        rooms = []
        for r in regions:
            for poly in mask_to_polygons(r["mask"], gp, min_area_m2=min_room).geoms:
                rgb = (r["key"][2], r["key"][1], r["key"][0])
                rooms.append((poly, rgb))
    else:
        pal = _room_palette(len(rooms_from_walls(wall, gp)))
        rooms = [(p, tuple(int(c) for c in pal[i][:3]))
                 for i, (p, _a) in enumerate(rooms_from_walls(wall, gp))]

    total_area = sum(p.area for p, _ in rooms)
    t_m = estimate_wall_thickness_px(wall) / px_per_m

    st.title("Floor plan to 3D")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wall parts", len(getattr(walls, "geoms", [])))
    c2.metric("Rooms", len(rooms))
    c3.metric("Floor area", f"{total_area:.1f} m²")
    c4.metric("Wall thickness", f"{t_m:.2f} m",
              help="Modal scanline run length, reading 1-2 px high because thresholding widens the "
                   "ink edge. This is the drawing's line weight, not real construction. A wildly "
                   "wrong value means the scale is wrong.")

    t_mask, t_scale, t_3d, t_rooms, t_exp = st.tabs(
        ["Check the mask", "Scale", "3D model", "Rooms", "Export"])

    with t_mask:
        st.caption("Tune until the red overlay covers the walls and nothing else. "
                   "Everything downstream inherits this mask.")
        a, b = st.columns(2)
        a.image(overlay(crop, wall), caption="Detected walls", width="stretch")
        if colour_mode and regions:
            b.image(numbered(crop, regions), caption="Numbered room regions", width="stretch")
        else:
            b.image(binarize(gray, wp), caption="Threshold output, before the sieve", width="stretch")
        if len(getattr(walls, "geoms", [])) == 0:
            st.warning("No walls survived. Lower the blob and shortest-run sliders.")
        if not rooms:
            st.warning("No rooms. In colour mode check the fill selection; in line-drawing mode "
                       "raise gap closing until the wall loop is continuous.")

    with t_scale:
        if colour_mode and regions:
            st.caption("Enter dimensions printed on the plan for a few numbered regions. Each row "
                       "gives two independent estimates, one per axis. Leave a cell at 0 to skip it. "
                       "Skip any room whose two printed numbers disagree with each other.")
            seed = pd.DataFrame({"Region": [1, 2, 3], "Width (m)": [0.0] * 3, "Depth (m)": [0.0] * 3})
            st.data_editor(seed, num_rows="dynamic", width="stretch", key="scale_refs")
            if nref:
                st.success(f"{px_per_m:.2f} px/m from {nref} measurements, sd {sd:.2f} "
                           f"({100 * sd / px_per_m:.1f}%). Printed labels rounded to 0.1 m contribute "
                           "about ±1.5% of that on a 3.3 m room.")
            else:
                st.info("No references entered yet; using the fallback scale.")
        else:
            st.info("Printed-size calibration needs the colour-filled mode. "
                    "Use overall width or pixels per metre in the sidebar.")

    with t_3d:
        with st.expander("Doors and windows"):
            st.caption("Openings are cut, not detected. Centre in metres, angle 0 means the wall runs along x.")
            seed = pd.DataFrame([{"x": 0.0, "y": 0.0, "width": 0.9, "height": 2.05,
                                  "sill": 0.0, "angle_deg": 0.0}]).iloc[0:0]
            table = st.data_editor(seed, num_rows="dynamic", width="stretch", key="openings")
        scene = build_scene(walls, rooms, gp, include_floor=show_floor)
        if len(table):
            scene, warns = cut_openings(scene, [Opening(**r) for r in table.to_dict("records")])
            for w_ in warns:
                st.warning(f"Boolean cut skipped: {w_}")
        stats = scene_stats(scene)
        if stats["faces"] > MAX_FACES:
            st.warning(f"{stats['faces']:,} triangles will render slowly. Raise contour simplification.")
        if stats["parts"] == 0:
            st.info("Nothing to show yet. Fix the mask first.")
        else:
            st.plotly_chart(scene_to_plotly(scene), width="stretch")
            st.caption(f"{stats['parts']} parts · {stats['faces']:,} triangles · "
                       f"bounding box {stats['bbox_m']} m")

    with t_rooms:
        if rooms:
            st.dataframe(pd.DataFrame([
                {"Room": i + 1, "Area (m²)": round(p.area, 2), "Perimeter (m)": round(p.length, 2),
                 "Centroid x (m)": round(p.centroid.x, 2), "Centroid y (m)": round(p.centroid.y, 2)}
                for i, (p, _c) in enumerate(rooms)]), width="stretch", hide_index=True)
            st.caption("Areas are measured to the inside wall faces.")
        else:
            st.info("No rooms detected.")

    with t_exp:
        if scene_stats(scene)["parts"] == 0:
            st.info("Build a model first.")
        else:
            st.caption("glTF keeps the per-room colours and is right for Blender, three.js and most "
                       "viewers. OBJ and STL are geometry only.")
            e1, e2, e3 = st.columns(3)
            e1.download_button("Download .glb", scene.export(file_type="glb"),
                               "floorplan.glb", "model/gltf-binary", width="stretch")
            obj = scene.export(file_type="obj")
            e2.download_button("Download .obj", obj if isinstance(obj, bytes) else obj.encode(),
                               "floorplan.obj", "text/plain", width="stretch")
            e3.download_button("Download .stl", scene.export(file_type="stl"),
                               "floorplan.stl", "model/stl", width="stretch")


if __name__ == "__main__":
    main()
