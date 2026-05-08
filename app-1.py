"""
Isovist Building Generator — Streamlit app.

Generates 3 random building masses (axis-aligned boxes + optional setbacks
and a cylindrical core), previews them in 3D with Plotly, and exports each
as an ASCII STL file for downstream isovist modeling.

Run:
    pip install streamlit numpy plotly
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import random
import struct
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    """Triangle soup. vertices: (N,3,3) array of triangle vertices."""
    triangles: np.ndarray  # shape (N, 3, 3)

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def bbox(self):
        v = self.triangles.reshape(-1, 3)
        return v.min(axis=0), v.max(axis=0)

    def merge(self, other: "Mesh") -> "Mesh":
        return Mesh(np.concatenate([self.triangles, other.triangles], axis=0))


def box_mesh(w: float, h: float, d: float,
             cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> Mesh:
    """Axis-aligned box centered at (cx, cy, cz)."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    z0, z1 = cz - d / 2, cz + d / 2

    # 8 corners
    p = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    # 12 triangles (CCW outward)
    faces = [
        (0, 2, 1), (0, 3, 2),  # -Z
        (4, 5, 6), (4, 6, 7),  # +Z
        (0, 1, 5), (0, 5, 4),  # -Y (bottom)
        (3, 6, 2), (3, 7, 6),  # +Y (top)
        (0, 4, 7), (0, 7, 3),  # -X
        (1, 2, 6), (1, 6, 5),  # +X
    ]
    tris = np.array([[p[a], p[b], p[c]] for (a, b, c) in faces])
    return Mesh(tris)


def cylinder_mesh(radius: float, height: float,
                  cx: float = 0.0, cz: float = 0.0,
                  y_base: float = 0.0, segments: int = 32) -> Mesh:
    """Cylinder along Y axis, base at y_base."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    bottom = np.stack([cx + radius * np.cos(angles),
                       np.full_like(angles, y_base),
                       cz + radius * np.sin(angles)], axis=1)
    top = bottom.copy()
    top[:, 1] = y_base + height

    tris = []
    cb = np.array([cx, y_base, cz])
    ct = np.array([cx, y_base + height, cz])
    for i in range(segments):
        j = (i + 1) % segments
        # side (two tris, outward CCW)
        tris.append([bottom[i], bottom[j], top[j]])
        tris.append([bottom[i], top[j], top[i]])
        # bottom cap (CCW seen from below = clockwise from above)
        tris.append([cb, bottom[j], bottom[i]])
        # top cap
        tris.append([ct, top[i], top[j]])
    return Mesh(np.array(tris))


# ---------------------------------------------------------------------------
# Random building generator
# ---------------------------------------------------------------------------

def generate_building(rng: random.Random) -> tuple[Mesh, dict]:
    """
    Build a random mass:
      - 1..3 base box blocks (overlapping is allowed)
      - 0..3 stacked setbacks on the tallest block
      - 45% chance of a cylindrical tower
    Returns merged mesh and a metadata dict.
    """
    base_extent = 12.0 + rng.random() * 18.0   # ~12..30 m
    parts: list[Mesh] = []
    block_info = []

    n_blocks = rng.randint(1, 3)
    for i in range(n_blocks):
        w = base_extent * (0.4 + rng.random() * 0.8)
        d = base_extent * (0.4 + rng.random() * 0.8)
        h = 8.0 + rng.random() * 40.0
        ox = 0.0 if i == 0 else (rng.random() - 0.5) * base_extent * 0.8
        oz = 0.0 if i == 0 else (rng.random() - 0.5) * base_extent * 0.8
        parts.append(box_mesh(w, h, d, ox, h / 2, oz))
        block_info.append({"w": w, "h": h, "d": d, "ox": ox, "oz": oz})

    tallest = max(block_info, key=lambda b: b["h"])
    cur_h = tallest["h"]
    n_setbacks = rng.randint(0, 3)
    for _ in range(n_setbacks):
        sw = tallest["w"] * (0.4 + rng.random() * 0.5)
        sd = tallest["d"] * (0.4 + rng.random() * 0.5)
        sh = 4.0 + rng.random() * 12.0
        sox = tallest["ox"] + (rng.random() - 0.5) * (tallest["w"] - sw) * 0.6
        soz = tallest["oz"] + (rng.random() - 0.5) * (tallest["d"] - sd) * 0.6
        parts.append(box_mesh(sw, sh, sd, sox, cur_h + sh / 2, soz))
        cur_h += sh

    has_cyl = rng.random() < 0.45
    if has_cyl:
        r = 2.0 + rng.random() * 4.0
        ch = 6.0 + rng.random() * 30.0
        cx = (rng.random() - 0.5) * base_extent * 0.5
        cz = (rng.random() - 0.5) * base_extent * 0.5
        parts.append(cylinder_mesh(r, ch, cx, cz, y_base=0.0, segments=28))

    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)

    mn, mx = mesh.bbox()
    meta = {
        "footprint": (float(mx[0] - mn[0]), float(mx[2] - mn[2])),
        "height": float(mx[1] - mn[1]),
        "blocks": n_blocks,
        "setbacks": n_setbacks,
        "cylinder": has_cyl,
        "triangles": mesh.num_triangles,
    }
    return mesh, meta


# ---------------------------------------------------------------------------
# STL writers
# ---------------------------------------------------------------------------

def _triangle_normals(tris: np.ndarray) -> np.ndarray:
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]
    n = np.cross(b - a, c - a)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return n / lengths


def mesh_to_stl_ascii(mesh: Mesh, name: str = "building") -> bytes:
    tris = mesh.triangles
    normals = _triangle_normals(tris)
    out = [f"solid {name}"]
    for n, t in zip(normals, tris):
        out.append(f"facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        out.append("  outer loop")
        for v in t:
            out.append(f"    vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        out.append("  endloop")
        out.append("endfacet")
    out.append(f"endsolid {name}")
    return ("\n".join(out)).encode("utf-8")


def mesh_to_stl_binary(mesh: Mesh) -> bytes:
    tris = mesh.triangles.astype(np.float32)
    normals = _triangle_normals(tris).astype(np.float32)
    n_tri = tris.shape[0]
    buf = io.BytesIO()
    buf.write(b"\x00" * 80)                       # header
    buf.write(struct.pack("<I", n_tri))           # triangle count
    for n, t in zip(normals, tris):
        buf.write(struct.pack("<3f", *n))
        buf.write(struct.pack("<3f", *t[0]))
        buf.write(struct.pack("<3f", *t[1]))
        buf.write(struct.pack("<3f", *t[2]))
        buf.write(struct.pack("<H", 0))           # attribute byte count
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plotly preview
# ---------------------------------------------------------------------------

def mesh_figure(mesh: Mesh, title: str) -> go.Figure:
    tris = mesh.triangles
    verts = tris.reshape(-1, 3)
    n = tris.shape[0]
    i = np.arange(0, 3 * n, 3)
    j = i + 1
    k = i + 2

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 2], z=verts[:, 1],   # swap to put Y up
            i=i, j=j, k=k,
            color="#e8e6df",
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1, roughness=0.8),
            lightposition=dict(x=100, y=100, z=200),
            opacity=1.0,
        )
    ])
    mn, mx = mesh.bbox()
    size = max(mx[0] - mn[0], mx[2] - mn[2], mx[1] - mn[1])
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        scene=dict(
            xaxis=dict(title="X (m)", showbackground=False),
            yaxis=dict(title="Z (m)", showbackground=False),
            zaxis=dict(title="Y / height (m)", showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        paper_bgcolor="#0d0e10",
        font=dict(color="#e8e6df"),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Isovist Building Generator", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem;}
      h1 {letter-spacing: .15em; font-size: 1.4rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ISOVIST · BUILDING GENERATOR")
st.caption("3 random building masses · STL export for isovist modeling")

with st.sidebar:
    st.header("Parameters")
    seed_input = st.text_input("Seed", value=st.session_state.get("seed", "ALPHA01"))
    fmt = st.radio("STL format", ["ASCII", "Binary"], index=1, horizontal=True)
    if st.button("↻ Regenerate (random seed)", use_container_width=True):
        st.session_state["seed"] = "".join(
            random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7)
        )
        st.rerun()
    st.session_state["seed"] = seed_input
    st.divider()
    st.markdown(
        "**Notes**\n\n"
        "- Units are meters. STL has no unit metadata; import as meters.\n"
        "- Geometry is a triangle soup of overlapping primitives (not boolean-unioned). "
        "Fine for 2D isovists at a given eye height; for strict watertight 3D "
        "isovists, run a boolean union in Blender/Rhino first.\n"
        "- Y is up. Plotly preview swaps axes so the building stands upright."
    )

seed = st.session_state["seed"] or "DEFAULT"

# Deterministic per-building RNGs
buildings = []
for i in range(3):
    rng = random.Random(f"{seed}-{i}")
    mesh, meta = generate_building(rng)
    buildings.append((mesh, meta))

cols = st.columns(3)
labels = ["A", "B", "C"]
for col, label, (mesh, meta) in zip(cols, labels, buildings):
    with col:
        st.plotly_chart(
            mesh_figure(mesh, f"Building {label}"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        fp = meta["footprint"]
        st.markdown(
            f"**Building {label}**  \n"
            f"Footprint: `{fp[0]:.1f} × {fp[1]:.1f} m`  \n"
            f"Height: `{meta['height']:.1f} m`  \n"
            f"Masses: {meta['blocks']} block(s), {meta['setbacks']} setback(s)"
            f"{', cylinder' if meta['cylinder'] else ''}  \n"
            f"Triangles: {meta['triangles']}"
        )
        if fmt == "ASCII":
            data = mesh_to_stl_ascii(mesh, name=f"building_{label}")
            mime = "model/stl"
        else:
            data = mesh_to_stl_binary(mesh)
            mime = "application/octet-stream"
        st.download_button(
            label=f"⬇ Download Building {label} STL",
            data=data,
            file_name=f"building_{label}_{seed}.stl",
            mime=mime,
            use_container_width=True,
        )

st.divider()

# Bundle all 3 as a zip
import zipfile
zbuf = io.BytesIO()
with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
    for label, (mesh, _) in zip(labels, buildings):
        if fmt == "ASCII":
            z.writestr(f"building_{label}_{seed}.stl",
                       mesh_to_stl_ascii(mesh, name=f"building_{label}"))
        else:
            z.writestr(f"building_{label}_{seed}.stl",
                       mesh_to_stl_binary(mesh))
st.download_button(
    "⬇ Download all 3 STL files (.zip)",
    data=zbuf.getvalue(),
    file_name=f"buildings_{seed}.zip",
    mime="application/zip",
)
