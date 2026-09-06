"""
app.py - Floor plan to 3D model
===============================
Streamlit front end for floorplan_core. Run with:

    streamlit run app.py

The UI's job is calibration and inspection. Automatic vectorisation of a raster
plan is not a solved problem, so every stage is exposed and the wall mask is
shown overlaid on the source: tuning against that overlay is the workflow, not
an optional debugging step.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from floorplan_core import (
    GeomParams,
    Opening,
    WallParams,
    binarize,
    build_scene,
    cut_openings,
    estimate_wall_thickness_px,
    extract_walls,
    mask_to_polygons,
    rooms_from_walls,
    scene_stats,
    synthetic_plan,
)

st.set_page_config(page_title="Floor plan to 3D", layout="wide")

MAX_FACES = 150_000  # above this Plotly's WebGL path gets sluggish


# --------------------------------------------------------------------------- #
# Cached stages
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _load_gray(data: bytes, max_side: int) -> np.ndarray:
    """Decode to grayscale and cap the long side.

    Downscaling before morphology is not just for speed: structuring-element
    sizes are in pixels, so a stable working resolution keeps one set of
    parameters valid across a 150 DPI scan and a 600 DPI export.
    """
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return gray


@st.cache_data(show_spinner=False)
def _masks(gray: np.ndarray, wp: WallParams) -> tuple[np.ndarray, np.ndarray]:
    bw = binarize(gray, wp)
    return bw, extract_walls(bw, wp)


@st.cache_data(show_spinner=False)
def _vectorise(wall: np.ndarray, gp: GeomParams):
    return mask_to_polygons(wall, gp), rooms_from_walls(wall, gp)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def overlay(gray: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Source plan with detected walls tinted red."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    tint = rgb.copy()
    tint[wall > 0] = (220, 40, 40)
    return cv2.addWeighted(rgb, 0.45, tint, 0.55, 0)


def scene_to_plotly(scene) -> go.Figure:
    fig = go.Figure()
    for name, geom in scene.geometry.items():
        v, f = geom.vertices, geom.faces
        try:
            r, g, b = geom.visual.face_colors[0][:3]
        except Exception:
            r, g, b = 190, 190, 195
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=f"rgb({r},{g},{b})",
                flatshading=True,
                name=name,
                hoverinfo="name",
                lighting=dict(ambient=0.55, diffuse=0.85, specular=0.08, roughness=0.9),
                lightposition=dict(x=100, y=200, z=300),
            )
        )
    fig.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data",  # metres are metres on all three axes
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.header("Source")
upload = st.sidebar.file_uploader("Floor plan image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
demo = st.sidebar.toggle("Use the built-in demo plan", value=upload is None)

max_side = st.sidebar.select_slider("Working resolution (px)", [800, 1200, 1600, 2000, 2600], value=1600)

if demo:
    gray = synthetic_plan(50.0)
    if max(gray.shape) > max_side:
        s = max_side / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
elif upload is not None:
    gray = _load_gray(upload.getvalue(), max_side)
else:
    st.title("Floor plan to 3D")
    st.info("Upload a floor plan image, or switch on the demo plan to see the pipeline run.")
    st.stop()

H, W = gray.shape

st.sidebar.header("Wall detection")
dark_ink = st.sidebar.toggle("Walls are dark on a light page", value=True)
adaptive = st.sidebar.toggle("Adaptive threshold", value=False,
                             help="Turn on for photos or scans with uneven lighting. Otsu is better for clean exports.")
block = st.sidebar.slider("Adaptive window (px)", 11, 151, 35, 2, disabled=not adaptive)
C = st.sidebar.slider("Adaptive bias", -20, 40, 10, disabled=not adaptive)
axis_aligned = st.sidebar.toggle("Orthogonal walls only", value=True,
                                 help="Keeps horizontal and vertical runs and discards everything else, which removes text and furniture. Switch off for diagonal or curved walls.")
min_line_px = st.sidebar.slider("Shortest wall run (px)", 5, 200, 40, disabled=not axis_aligned)
min_thick_px = st.sidebar.slider("Thinnest wall (px)", 0, 31, 0,
                                 help="Discards strokes thinner than this, which removes hatching and stair treads. Leave at 0 for hollow double-line walls: it runs before gap closing and would erase them.")
close_px = st.sidebar.slider("Gap closing (px)", 0, 41, 5,
                             help="Set a little above the drawn wall thickness and no higher. Too high and dimension lines get bridged into the walls, which nothing downstream can undo.")
min_blob_px = st.sidebar.slider("Smallest kept blob (px)", 0, 5000, 200, 50)

wp = WallParams(dark_ink=dark_ink, adaptive=adaptive, block=block, C=C,
                axis_aligned=axis_aligned, min_line_px=min_line_px,
                close_px=close_px, min_thick_px=min_thick_px, min_blob_px=min_blob_px)

bw, wall = _masks(gray, wp)

# --- scale -----------------------------------------------------------------
st.sidebar.header("Scale")
ys, xs = np.nonzero(wall)
span_px = float(xs.max() - xs.min()) if xs.size else float(W)
mode = st.sidebar.radio("Calibrate by", ["Overall width", "Pixels per metre"], horizontal=True)
if mode == "Overall width":
    known_m = st.sidebar.number_input("Real width of the detected plan (m)", 1.0, 500.0, 10.2, 0.1,
                                      help=f"The detected walls span {span_px:.0f} px horizontally.")
    px_per_m = span_px / max(known_m, 1e-6)
    st.sidebar.caption(f"{px_per_m:.1f} px/m")
else:
    px_per_m = st.sidebar.number_input("Pixels per metre", 1.0, 2000.0, 50.0, 1.0)

st.sidebar.header("3D")
wall_h = st.sidebar.slider("Wall height (m)", 2.0, 6.0, 2.7, 0.05)
floor_t = st.sidebar.slider("Floor thickness (m)", 0.0, 0.5, 0.12, 0.01)
eps_rel = st.sidebar.select_slider("Contour simplification",
                                   [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016], value=0.002,
                                   help="Douglas-Peucker tolerance as a fraction of each contour's perimeter. Higher means fewer triangles and squarer walls.")
min_room = st.sidebar.slider("Smallest room (m²)", 0.5, 20.0, 1.5, 0.5)
show_floor = st.sidebar.toggle("Show floor slabs", value=True)

gp = GeomParams(px_per_m=px_per_m, eps_rel=eps_rel, wall_height_m=wall_h,
                floor_thickness_m=floor_t, min_room_area_m2=min_room)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

st.title("Floor plan to 3D")

walls, rooms = _vectorise(wall, gp)
t_m = estimate_wall_thickness_px(wall) / px_per_m
total_area = sum(a for _, a in rooms)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wall parts", len(getattr(walls, "geoms", [])))
c2.metric("Rooms", len(rooms))
c3.metric("Enclosed area", f"{total_area:.1f} m²")
c4.metric("Wall thickness", f"{t_m:.2f} m", help="Modal scanline run length. Reads 1-2 px high because thresholding widens the ink edge. A wildly wrong value means the scale is wrong.")

tab_check, tab_3d, tab_rooms, tab_export = st.tabs(["Check the mask", "3D model", "Rooms", "Export"])

with tab_check:
    st.caption("Tune the sidebar until the red overlay covers the walls and nothing else. Everything downstream inherits this mask.")
    a, b = st.columns(2)
    a.image(overlay(gray, wall), caption="Detected walls over the source", width="stretch")
    b.image(bw, caption="Threshold output, before the wall sieve", width="stretch")
    if len(getattr(walls, "geoms", [])) == 0:
        st.warning("No walls survived. Lower the smallest-blob and shortest-run sliders, or switch off orthogonal-walls-only.")
    if len(rooms) == 0:
        st.warning("No enclosed rooms. Raise gap closing until the wall loop is continuous, or lower the smallest-room threshold.")

with tab_3d:
    with st.expander("Doors and windows", expanded=False):
        st.caption("Openings are cut, not detected. Give the centre in metres in the same frame as the axes below; angle 0 means the wall runs along x.")
        seed = pd.DataFrame([{"x": 0.0, "y": 0.0, "width": 0.9, "height": 2.05,
                              "sill": 0.0, "angle_deg": 0.0}]).iloc[0:0]
        table = st.data_editor(seed, num_rows="dynamic", width="stretch", key="openings")

    scene = build_scene(walls, rooms, gp, include_floor=show_floor)
    warns: list[str] = []
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
        st.caption(f"{stats['parts']} parts · {stats['faces']:,} triangles · bounding box {stats['bbox_m']} m")

with tab_rooms:
    if rooms:
        df = pd.DataFrame(
            [{"Room": f"{i + 1}", "Area (m²)": round(a, 2),
              "Perimeter (m)": round(p.length, 2),
              "Centroid x (m)": round(p.centroid.x, 2),
              "Centroid y (m)": round(p.centroid.y, 2)}
             for i, (p, a) in enumerate(rooms)]
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption("Areas are of the enclosed free space, measured to the inside wall faces. A doorway drawn as a full gap merges two rooms into one entry.")
    else:
        st.info("No enclosed rooms detected. Raise gap closing in the sidebar.")

with tab_export:
    if stats["parts"] == 0:
        st.info("Build a model first.")
    else:
        st.caption("glTF keeps the per-room colours and is the right choice for Blender, three.js and most viewers. OBJ and STL are geometry only.")
        e1, e2, e3 = st.columns(3)
        e1.download_button("Download .glb", scene.export(file_type="glb"),
                           "floorplan.glb", "model/gltf-binary", width="stretch")
        obj = scene.export(file_type="obj")
        e2.download_button("Download .obj", obj if isinstance(obj, bytes) else obj.encode(),
                           "floorplan.obj", "text/plain", width="stretch")
        e3.download_button("Download .stl", scene.export(file_type="stl"),
                           "floorplan.stl", "model/stl", width="stretch")
