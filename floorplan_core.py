"""
floorplan_core.py
=================
Raster floor plan  ->  vector wall/room polygons  ->  extruded 3D mesh.

Deliberately free of Streamlit imports so the pipeline can be unit-tested,
batch-run, or reused from a notebook.

Pipeline
--------
    grayscale
      -> binarise (Otsu | adaptive Gaussian)
      -> morphological line sieve (removes text, furniture, hatching)
      -> connected-component area filter
      -> contour tracing (RETR_CCOMP: shells + holes)
      -> Ramer-Douglas-Peucker simplification
      -> pixel -> metre affine map (with y-flip)
      -> shapely polygons -> trimesh prism extrusion
      -> optional boolean subtraction for door/window openings

Assumptions are stated in each function's docstring. Nothing here does
semantic understanding: it is a geometric vectoriser, not a floor-plan parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

__all__ = [
    "WallParams",
    "GeomParams",
    "Opening",
    "binarize",
    "extract_walls",
    "estimate_wall_thickness_px",
    "mask_to_polygons",
    "rooms_from_walls",
    "build_scene",
    "cut_openings",
    "scene_stats",
]

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WallParams:
    """Raster -> binary wall mask."""

    dark_ink: bool = True          # True: walls are dark on a light page
    adaptive: bool = False         # adaptive threshold (uneven scans/photos)
    block: int = 35                # adaptive window, forced odd
    C: int = 10                    # adaptive bias
    axis_aligned: bool = True      # morphological H/V line sieve
    min_line_px: int = 40          # shortest structure kept by the sieve
    close_px: int = 5              # fills hollow double-line walls
    min_thick_px: int = 0          # rejects dimension lines, hatching, leaders
    min_blob_px: int = 200         # drops leftover speckle / glyphs


@dataclass(frozen=True)
class GeomParams:
    """Vectorisation and extrusion."""

    px_per_m: float = 100.0        # scale calibration
    eps_rel: float = 0.002         # RDP tolerance as fraction of perimeter
    wall_height_m: float = 2.7
    floor_thickness_m: float = 0.12
    min_wall_area_m2: float = 0.02
    min_room_area_m2: float = 1.5


@dataclass(frozen=True)
class Opening:
    """A door or window cut out of the walls. Metres, world frame."""

    x: float                       # centre x
    y: float                       # centre y
    width: float = 0.9             # along the wall
    height: float = 2.05           # vertical extent
    sill: float = 0.0              # bottom of the opening above floor
    angle_deg: float = 0.0         # wall bearing; 0 = wall runs along +x
    depth: float = 1.0             # cut depth across the wall


# --------------------------------------------------------------------------- #
# Stage 1 - binarisation
# --------------------------------------------------------------------------- #


def binarize(gray: np.ndarray, p: WallParams) -> np.ndarray:
    """Return a uint8 {0,255} mask where 255 marks ink (candidate wall pixels).

    Otsu is the default because printed plans are strongly bimodal. Adaptive
    thresholding is the fallback for photographs and shaded scans where the
    background level drifts across the page.
    """
    if gray.ndim != 2:
        raise ValueError("binarize expects a single-channel image")
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    mode = cv2.THRESH_BINARY_INV if p.dark_ink else cv2.THRESH_BINARY
    if p.adaptive:
        blk = max(3, int(p.block) | 1)  # OpenCV requires odd >= 3
        return cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, blk, float(p.C)
        )
    _, bw = cv2.threshold(g, 0, 255, mode | cv2.THRESH_OTSU)
    return bw


# --------------------------------------------------------------------------- #
# Stage 2 - wall isolation
# --------------------------------------------------------------------------- #


def _remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def extract_walls(bw: np.ndarray, p: WallParams) -> np.ndarray:
    """Isolate wall structure from an ink mask.

    The morphological sieve is an opening with a 1 x L horizontal element and a
    L x 1 vertical element. Only runs at least L pixels long in one of those two
    directions survive, which is exactly the structure of an orthogonal wall and
    exactly not the structure of glyphs, dimension arrows, furniture icons and
    hatching. Limitation: it also erases diagonal and curved walls, so set
    ``axis_aligned=False`` for plans containing them and rely on the area filter.

    The closing merges the two faces of a hollow double-line wall into one solid
    slab, so ``close_px`` should be a little larger than the drawn wall thickness
    and no larger. It is the most consequential parameter here and it fails
    silently when overdriven: on a test plan with a dimension line 25 px clear of
    the envelope, raising ``close_px`` from 21 to 24 bridged the line into the
    wall and inflated the measured footprint by 4.8%. Nothing downstream can
    undo that, because once bridged the annotation is part of the wall blob.
    Below the bridging threshold the same line is harmless: it survives as a 1 px
    sliver that Douglas-Peucker and the minimum-area filter discard.

    The sieve cannot reject dimension lines, leader lines, hatching or stair
    treads: those are long straight runs, which is exactly what it keeps. What
    separates them from walls is thickness, so ``min_thick_px`` opens with a
    square element and drops anything thinner than that in both directions.

    Ordering was decided by measurement, not by taste. The thickness opening runs
    *before* the closing. Placed after it, it does nothing measurable, because
    the closing has already fused thin annotations into thick blobs: on a plan
    hatched with 5 px strokes, the hatched room read 39.6 m2 for every value of
    ``min_thick_px`` when the opening ran last, against a true 45.2 m2. Running
    it first at 9 px recovered 44.3 m2.

    The cost of that ordering is that a hollow double-line wall is drawn as two
    thin strokes and has not yet been merged into a slab, so any non-zero
    ``min_thick_px`` erases it. Leave the parameter at 0 for hollow-wall plans;
    it is only for solid-wall drawings carrying heavy annotation.
    """
    wall = bw
    if p.axis_aligned:
        L = max(3, int(p.min_line_px))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
        wall = cv2.bitwise_or(
            cv2.morphologyEx(bw, cv2.MORPH_OPEN, kh),
            cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv),
        )
    if p.min_thick_px > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(p.min_thick_px), int(p.min_thick_px))
        )
        wall = cv2.morphologyEx(wall, cv2.MORPH_OPEN, k)
    if p.close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(p.close_px), int(p.close_px))
        )
        wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k)
    return _remove_small_components(wall, int(p.min_blob_px))


def _run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Lengths of all maximal runs of foreground along the given axis."""
    a = (mask > 0).astype(np.int8)
    if axis == 0:
        a = a.T
    a = np.pad(a, ((0, 0), (1, 1)))
    d = np.diff(a, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    if len(starts) != len(ends):
        return np.empty(0, dtype=np.int64)
    return (ends[:, 1] - starts[:, 1]).astype(np.int64)


def estimate_wall_thickness_px(wall: np.ndarray) -> float:
    """Estimate wall thickness in pixels as the modal scanline run length.

    Scanning rows crosses every vertical wall at exactly its thickness, and
    scanning columns does the same for horizontal walls. Wall *lengths* also
    appear in the histogram but are spread over many distinct values, whereas
    the thickness piles up into a sharp mode. Taking the mode is therefore
    robust to junctions, stubs and long runs.

    Measured against synthetic ground truth in ``_selftest`` the bias is a fixed
    +1 to +2 px, contributed by the Gaussian pre-blur and threshold widening the
    anti-aliased ink edge, not by this estimator. It is an absolute offset, not
    a proportional one, so it matters only at low resolution.

    Rejected alternative: the distance transform gives t = 4 E[DT] for an
    infinite slab (DT is uniform on (0, t/2) there), but corners and T-junctions
    push the medial surface away from the background and inflate the mean. On
    the same synthetic plans that estimator ran 28-49% high, so it is not used.
    """
    if not np.any(wall):
        return 0.0
    lengths = np.concatenate([_run_lengths(wall, 0), _run_lengths(wall, 1)])
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return 0.0
    return float(np.bincount(lengths).argmax())


# --------------------------------------------------------------------------- #
# Stage 3 - vectorisation
# --------------------------------------------------------------------------- #


def _px_to_world(contour: np.ndarray, px_per_m: float, img_h: int) -> np.ndarray:
    """Map pixel coordinates to a right-handed metric frame.

    Image rows increase downwards; the world y axis increases upwards. Omitting
    the flip mirrors the whole building, which is easy to miss on a symmetric
    plan and impossible to fix downstream.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    s = 1.0 / float(px_per_m)
    return np.column_stack([pts[:, 0] * s, (img_h - pts[:, 1]) * s])


def _simplify(contour: np.ndarray, eps_rel: float) -> np.ndarray:
    """Ramer-Douglas-Peucker with tolerance proportional to perimeter.

    A relative tolerance keeps behaviour stable across plan sizes and DPI.
    Typical range 0.001-0.01: below that the mesh carries pixel staircase noise,
    above it short wall stubs are swallowed.
    """
    eps = float(eps_rel) * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, max(eps, 1e-9), True)


def _clean(poly: Polygon) -> list[Polygon]:
    """Repair self-intersections introduced by simplification."""
    if poly.is_valid:
        fixed = poly
    else:
        fixed = poly.buffer(0)
    if fixed.is_empty:
        return []
    if isinstance(fixed, Polygon):
        return [fixed]
    return [g for g in getattr(fixed, "geoms", []) if isinstance(g, Polygon)]


def mask_to_polygons(
    mask: np.ndarray, g: GeomParams, min_area_m2: float | None = None
) -> MultiPolygon:
    """Trace a binary mask into metric polygons, holes included.

    ``RETR_CCOMP`` returns a two-level hierarchy: rows with parent == -1 are
    outer boundaries, their children are holes. Building the shell/hole pairing
    from that hierarchy is what makes a courtyard or a stair void come out as a
    void instead of a solid block.
    """
    min_area = g.min_wall_area_m2 if min_area_m2 is None else min_area_m2
    h = mask.shape[0]
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return MultiPolygon()
    hierarchy = hierarchy[0]

    polys: list[Polygon] = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:  # a hole; consumed by its parent
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


def rooms_from_walls(
    wall: np.ndarray, g: GeomParams
) -> list[tuple[Polygon, float]]:
    """Segment enclosed free space into rooms, sorted by descending area.

    Free space is the complement of the wall mask. Components touching the image
    border are exterior and discarded; what remains is enclosed. This depends on
    the wall loop being topologically closed, so a doorway drawn as a full gap
    will merge two rooms, and a break in the envelope will leak every room to the
    outside. Increase ``close_px`` if the room count looks too low.
    """
    free = (wall == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    min_px = g.min_room_area_m2 * g.px_per_m**2

    out: list[tuple[Polygon, float]] = []
    for i in range(1, n):
        if i in border or stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        mp = mask_to_polygons(comp, g, min_area_m2=g.min_room_area_m2)
        for poly in mp.geoms:
            out.append((poly, float(poly.area)))
    return sorted(out, key=lambda t: -t[1])


# --------------------------------------------------------------------------- #
# Stage 4 - extrusion
# --------------------------------------------------------------------------- #


def _extrude(poly: Polygon, height: float) -> trimesh.Trimesh | None:
    """Prism extrusion from z=0 to z=height, with a triangulator fallback.

    ``earcut`` (mapbox_earcut, ISC licence) is preferred over the ``triangle``
    engine, which wraps Shewchuk's Triangle and is not licensed for commercial
    use.
    """
    for kwargs in ({"engine": "earcut"}, {}):
        try:
            return trimesh.creation.extrude_polygon(poly, height=float(height), **kwargs)
        except Exception:
            continue
    return None


def build_scene(
    walls: MultiPolygon,
    rooms: Sequence[tuple[Polygon, float]],
    g: GeomParams,
    include_floor: bool = True,
) -> trimesh.Scene:
    """Assemble wall prisms and per-room floor slabs into a trimesh Scene.

    Walls occupy z in [0, wall_height]; floor slabs occupy [-floor_thickness, 0]
    so the wall base and the floor top share the z=0 plane and no gap appears at
    the joint.
    """
    scene = trimesh.Scene()
    wall_rgba = [190, 190, 195, 255]

    for i, poly in enumerate(getattr(walls, "geoms", [])):
        m = _extrude(poly, g.wall_height_m)
        if m is None or m.is_empty:
            continue
        m.visual.face_colors = wall_rgba
        scene.add_geometry(m, node_name=f"wall_{i:03d}", geom_name=f"wall_{i:03d}")

    if include_floor:
        palette = _room_palette(len(rooms))
        for i, (poly, _area) in enumerate(rooms):
            m = _extrude(poly, g.floor_thickness_m)
            if m is None or m.is_empty:
                continue
            m.apply_translation([0.0, 0.0, -g.floor_thickness_m])
            m.visual.face_colors = palette[i]
            scene.add_geometry(m, node_name=f"room_{i:03d}", geom_name=f"room_{i:03d}")

    return scene


def _room_palette(n: int) -> np.ndarray:
    """Distinct, evenly spaced hues; deterministic across runs."""
    if n <= 0:
        return np.zeros((0, 4), dtype=np.uint8)
    hues = (np.arange(n) * (180.0 / max(n, 1))).astype(np.uint8).reshape(-1, 1, 1)
    hsv = np.concatenate(
        [hues, np.full_like(hues, 90), np.full_like(hues, 225)], axis=2
    )
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return np.hstack([rgb, np.full((n, 1), 255, np.uint8)])


# --------------------------------------------------------------------------- #
# Stage 5 - openings (optional)
# --------------------------------------------------------------------------- #


def cut_openings(
    scene: trimesh.Scene, openings: Iterable[Opening]
) -> tuple[trimesh.Scene, list[str]]:
    """Subtract door/window boxes from the wall geometry.

    Uses the ``manifold`` engine (manifold3d), which is pip-installable and does
    not require Blender. Boolean CSG on meshes produced by extruding simplified
    contours can fail on degenerate or non-manifold input, so every subtraction
    is attempted independently and failures are reported rather than raised.
    Returns the new scene and a list of warning strings.
    """
    ops = list(openings)
    warnings: list[str] = []
    if not ops:
        return scene, warnings

    cutters = []
    for o in ops:
        box = trimesh.creation.box(
            extents=[float(o.width), float(o.depth), float(o.height)]
        )
        T = trimesh.transformations.rotation_matrix(
            np.radians(float(o.angle_deg)), [0, 0, 1]
        )
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
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                warnings.append(f"opening {j} vs {name}: {type(exc).__name__}")
        mesh.visual.face_colors = [190, 190, 195, 255]
        out.add_geometry(mesh, node_name=name, geom_name=name)
    return out, warnings


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def scene_stats(scene: trimesh.Scene) -> dict:
    faces = sum(len(g.faces) for g in scene.geometry.values())
    verts = sum(len(g.vertices) for g in scene.geometry.values())
    b = scene.bounds
    return {
        "parts": len(scene.geometry),
        "vertices": verts,
        "faces": faces,
        "bbox_m": None if b is None else (b[1] - b[0]).round(3).tolist(),
    }


# --------------------------------------------------------------------------- #
# Self-test on a synthetic plan
# --------------------------------------------------------------------------- #


def synthetic_plan(px_per_m: float = 50.0) -> np.ndarray:
    """A 10 m x 7 m two-room plan with a courtyard void, plus decoy text.

    Ground truth: envelope 10 x 7 m, walls 0.2 m thick, one internal partition,
    a 2 x 2 m void. Used to verify the pipeline end to end without a real scan.
    """
    s = px_per_m
    H, W = int(7 * s) + 80, int(10 * s) + 80
    img = np.full((H, W), 255, np.uint8)
    t = max(2, int(round(0.2 * s)))
    x0, y0 = 40, 40
    x1, y1 = x0 + int(10 * s), y0 + int(7 * s)

    cv2.rectangle(img, (x0, y0), (x1, y1), 0, t)                 # envelope
    xm = x0 + int(6 * s)
    cv2.line(img, (xm, y0), (xm, y1), 0, t)                      # partition
    cx, cy = x0 + int(2 * s), y0 + int(2 * s)
    cv2.rectangle(img, (cx, cy), (cx + int(2 * s), cy + int(2 * s)), 0, t)  # void

    cv2.putText(img, "BEDROOM 3.5x4.0", (x0 + 20, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)             # decoy text
    cv2.circle(img, (x1 - 60, y0 + 60), 18, 0, 1)                # decoy furniture
    return img


def _selftest() -> int:
    """End-to-end check against a plan whose true dimensions are known.

    Note on ground truth: cv2.rectangle strokes are centred on the path, so a
    10.0 x 7.0 m centreline box drawn with 0.2 m walls has a 10.2 x 7.2 m outer
    envelope. The envelope, not the centreline, is what contour tracing recovers.
    """
    print("floorplan_core self-test")
    px_per_m = 50.0
    img = synthetic_plan(px_per_m)
    wp = WallParams(min_line_px=30, close_px=7, min_blob_px=150)
    gp = GeomParams(px_per_m=px_per_m, eps_rel=0.004, min_room_area_m2=1.0)

    wall = extract_walls(binarize(img, wp), wp)

    t_m = estimate_wall_thickness_px(wall) / px_per_m
    print(f"  thickness   {t_m:.3f} m            (drawn 0.200, +1-2 px raster bias)")

    polys = mask_to_polygons(wall, gp)
    minx, miny, maxx, maxy = polys.bounds
    w, h = maxx - minx, maxy - miny
    print(f"  footprint   {w:.2f} x {h:.2f} m     (true envelope 10.20 x 7.20)")

    holes = sum(len(p.interiors) for p in polys.geoms)
    print(f"  parts {len(polys.geoms)}, interior rings {holes}   (expect 2 cells + 1 void = 3)")

    rooms = rooms_from_walls(wall, gp)
    print("  rooms " + ", ".join(f"{a:.1f} m2" for _, a in rooms))

    scene = build_scene(polys, rooms, gp)
    before = scene_stats(scene)
    print(f"  scene       {before}")

    # The partition sits at pixel x = 40 + 6*px_per_m, i.e. world x = 6.8 m,
    # and spans world y in [0.8, 7.8]. Place the door on it, not beside it.
    door = Opening(x=6.8, y=4.3, width=0.9, height=2.05, angle_deg=90.0, depth=1.0)
    scene, warn = cut_openings(scene, [door])
    after = scene_stats(scene)
    print(f"  after door  {after}  warnings={warn}")

    glb = scene.export(file_type="glb")
    print(f"  glb export  {len(glb)} bytes")

    checks = {
        "footprint width": abs(w - 10.2) < 0.15,
        "footprint depth": abs(h - 7.2) < 0.15,
        "thickness": abs(t_m - 0.2) < 0.05,
        "void preserved": holes >= 3,
        "rooms found": len(rooms) >= 2,
        "door removed material": after["faces"] > before["faces"],
        "glb exported": len(glb) > 1000,
    }
    for k, v in checks.items():
        print(f"    {'ok ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
