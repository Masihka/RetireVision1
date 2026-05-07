"""
3D Isovist Saliency Analyzer
============================
A Streamlit app that reproduces and extends the methodology of:
  Bhatia, Chalup & Ostwald (2012), "Analyzing architectural space:
  identifying salient regions by computing 3D isovists", ANZAScA 2012.

The app:
  1. Loads or generates a 3D scene (built-in test scenes or user-uploaded
     OBJ / PLY mesh).
  2. Lets the user place vantage points along a path or on a grid.
  3. Casts 3D isovist rays (azimuth 0-360 deg, polar 0-180 deg) at multiple
     eye heights -> Boundary-Length Isovists (BLIs).
  4. Reduces the per-vantage-point ray array to Z statistics (Zmin, Zmax,
     Zmean, Zvar) -> heat maps over (height, azimuth).
  5. Runs PCA on each heat map to obtain a principal-component subspace.
  6. Computes Global and Local saliency using:
        - Krzanowski-based subspace angle measure (paper Eq. 1)
        - Shannon entropy of subspace coefficients (paper Eq. 2)
        - A combined saliency score.
  7. ML extensions: clusters vantage points (KMeans), visualises them in
     2D via PCA / t-SNE, and reports cluster-level saliency.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# trimesh is used for ray-casting against meshes (built-in or uploaded).
# It is the most reliable pure-Python option for this scale of work.
import trimesh

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Isovist Saliency Analyzer",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ 3D Isovist Saliency Analyzer")
st.caption(
    "Reproduces Bhatia, Chalup & Ostwald (2012). 3D ray-cast isovists → "
    "Z-statistic heat maps → PCA subspaces → Global / Local saliency, "
    "plus ML clustering of vantage points."
)


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------
def make_box_room(size=(10.0, 10.0, 3.0)) -> trimesh.Trimesh:
    """Single rectangular room. Useful as a sanity check (low saliency
    everywhere). The mesh is left as a closed solid - the ray caster
    happily finds first-hit faces from interior origins, and trimesh's
    contains() works correctly for vantage-point filtering."""
    sx, sy, sz = size
    box = trimesh.creation.box(extents=(sx, sy, sz))
    # Move so the floor is at z=0
    box.apply_translation((0, 0, sz / 2))
    return box


def make_corridor_with_rooms() -> trimesh.Trimesh:
    """A corridor with two side rooms - has clear salient junctions
    where the corridor opens up."""
    parts = []
    # Main corridor (long, narrow)
    corridor = trimesh.creation.box(extents=(20.0, 3.0, 3.0))
    corridor.apply_translation((0, 0, 1.5))
    parts.append(corridor)
    # Two perpendicular rooms
    room_a = trimesh.creation.box(extents=(5.0, 5.0, 3.0))
    room_a.apply_translation((-4.0, 4.0, 1.5))
    parts.append(room_a)
    room_b = trimesh.creation.box(extents=(5.0, 5.0, 3.0))
    room_b.apply_translation((4.0, -4.0, 1.5))
    parts.append(room_b)
    scene = trimesh.util.concatenate(parts)
    return scene


def make_villa_like() -> trimesh.Trimesh:
    """A simplified two-storey, multi-room model loosely inspired by the
    Villa Savoye layout used in the paper. NOT a faithful reproduction -
    just enough geometric variety (corridor, opening, mezzanine, ramp)
    to make saliency analysis meaningful."""
    parts = []
    # Outer shell - 20 x 20 x 6
    shell = trimesh.creation.box(extents=(20.0, 20.0, 6.0))
    shell.apply_translation((0, 0, 3.0))
    parts.append(shell)
    # Mid floor slab with a stairwell hole
    slab = trimesh.creation.box(extents=(20.0, 20.0, 0.3))
    slab.apply_translation((0, 0, 3.0))
    parts.append(slab)
    # Interior partition walls (ground floor)
    w1 = trimesh.creation.box(extents=(0.3, 8.0, 3.0))
    w1.apply_translation((-3.0, -4.0, 1.5))
    parts.append(w1)
    w2 = trimesh.creation.box(extents=(8.0, 0.3, 3.0))
    w2.apply_translation((3.0, 2.0, 1.5))
    parts.append(w2)
    # Curved-ish partition (approximated by a cylinder slice)
    cyl = trimesh.creation.cylinder(radius=2.0, height=3.0, sections=24)
    cyl.apply_translation((-5.0, 5.0, 1.5))
    parts.append(cyl)
    # Spiral-stair stand-in: a tall thin cylinder
    stair = trimesh.creation.cylinder(radius=0.8, height=6.0, sections=16)
    stair.apply_translation((0.0, 0.0, 3.0))
    parts.append(stair)
    # Upper floor partition
    w3 = trimesh.creation.box(extents=(0.3, 6.0, 2.7))
    w3.apply_translation((4.0, 0.0, 4.65))
    parts.append(w3)

    scene = trimesh.util.concatenate(parts)
    return scene


SCENE_FACTORIES = {
    "Box room (sanity check)": make_box_room,
    "Corridor + 2 rooms": make_corridor_with_rooms,
    "Villa-like (multi-room, 2 storeys)": make_villa_like,
}


# ---------------------------------------------------------------------------
# Vantage point sampling
# ---------------------------------------------------------------------------
def _candidate_resolution(n_target: int, oversample: int = 8) -> int:
    """Choose a per-axis sample count that yields enough candidates to
    survive the inside-mesh filter even in narrow geometries."""
    # We want at least n_target * oversample candidate cells.
    return max(10, int(np.ceil(np.sqrt(n_target * oversample))))


def sample_grid_inside(mesh: trimesh.Trimesh,
                       n_target: int = 50,
                       z: float = 1.6) -> np.ndarray:
    """Sample vantage points on a regular 2D grid at height z, keeping only
    those that lie inside the mesh (i.e. inside the building)."""
    bounds = mesh.bounds  # (2,3)
    n_per = _candidate_resolution(n_target)
    xs = np.linspace(bounds[0, 0] + 0.3, bounds[1, 0] - 0.3, n_per)
    ys = np.linspace(bounds[0, 1] + 0.3, bounds[1, 1] - 0.3, n_per)
    pts = np.array([(x, y, z) for x in xs for y in ys])

    try:
        inside = mesh.contains(pts)
        if inside.any():
            pts = pts[inside]
    except Exception:
        pass

    if len(pts) > n_target:
        idx = np.linspace(0, len(pts) - 1, n_target).astype(int)
        pts = pts[idx]
    return pts


def sample_path_inside(mesh: trimesh.Trimesh,
                       n_points: int = 50,
                       z: float = 1.6) -> np.ndarray:
    """Sample an ordered snake-pattern path through the bounding box,
    keeping only points inside the mesh. Mirrors the 'walk through'
    workflow described in the paper, but generated deterministically
    rather than from a user's clicks."""
    bounds = mesh.bounds
    n_per = _candidate_resolution(n_points)
    xs = np.linspace(bounds[0, 0] + 0.3, bounds[1, 0] - 0.3, n_per)
    ys = np.linspace(bounds[0, 1] + 0.3, bounds[1, 1] - 0.3, n_per)
    pts = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        for x in row:
            pts.append((x, y, z))
    pts = np.array(pts)

    try:
        inside = mesh.contains(pts)
        if inside.any():
            pts = pts[inside]
    except Exception:
        pass

    if len(pts) > n_points:
        idx = np.linspace(0, len(pts) - 1, n_points).astype(int)
        pts = pts[idx]
    return pts


# ---------------------------------------------------------------------------
# 3D isovist ray casting
# ---------------------------------------------------------------------------
@dataclass
class IsovistConfig:
    n_azimuth: int = 60        # samples around 360 deg
    n_polar: int = 30          # samples 0..180 deg
    n_heights: int = 5         # eye-height layers
    h_min: float = 0.4         # m above floor
    h_max: float = 2.0         # m above floor
    max_ray: float = 50.0      # m, fallback if no hit (BLI: clipped)


def build_ray_directions(n_azimuth: int, n_polar: int) -> np.ndarray:
    """Build a (n_azimuth*n_polar, 3) array of unit direction vectors
    parameterised by azimuth theta (xy-plane) and polar phi (from +z).
    Matches the (theta, phi) convention used in the paper."""
    thetas = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    phis = np.linspace(0.01, np.pi - 0.01, n_polar)  # avoid exact poles
    tt, pp = np.meshgrid(thetas, phis, indexing='ij')  # (n_az, n_pol)
    sin_p = np.sin(pp)
    dirs = np.stack([sin_p * np.cos(tt),
                     sin_p * np.sin(tt),
                     np.cos(pp)], axis=-1)  # (n_az, n_pol, 3)
    return dirs.reshape(-1, 3)


def cast_isovist(mesh: trimesh.Trimesh,
                 origin: np.ndarray,
                 cfg: IsovistConfig,
                 ray_intersector=None) -> np.ndarray:
    """Cast rays from `origin` and return ray-length array of shape
    (n_heights, n_azimuth, n_polar). Hits beyond max_ray are clipped to
    max_ray (Boundary-Length Isovist behaviour for closed meshes)."""
    if ray_intersector is None:
        ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    dirs_unit = build_ray_directions(cfg.n_azimuth, cfg.n_polar)
    n_rays = dirs_unit.shape[0]

    heights = np.linspace(cfg.h_min, cfg.h_max, cfg.n_heights)
    out = np.zeros((cfg.n_heights, cfg.n_azimuth, cfg.n_polar), dtype=np.float32)

    for hi, dh in enumerate(heights):
        origin_h = np.array([origin[0], origin[1], origin[2] - cfg.h_max + dh])
        # Replicate origin for every ray
        origins = np.tile(origin_h, (n_rays, 1))
        # Find first hit per ray
        locations, index_ray, _ = ray_intersector.intersects_location(
            origins, dirs_unit, multiple_hits=False
        )
        lengths = np.full(n_rays, cfg.max_ray, dtype=np.float32)
        if len(index_ray) > 0:
            d = np.linalg.norm(locations - origins[index_ray], axis=1)
            # Keep nearest hit per ray (intersects_location with
            # multiple_hits=False already returns nearest, but be safe)
            for r, dist in zip(index_ray, d):
                if dist < lengths[r]:
                    lengths[r] = dist
        np.clip(lengths, 0.0, cfg.max_ray, out=lengths)
        out[hi] = lengths.reshape(cfg.n_azimuth, cfg.n_polar)

    return out  # (H, A, P)


# ---------------------------------------------------------------------------
# Z statistics & heat maps
# ---------------------------------------------------------------------------
def compute_z_matrices(rays: np.ndarray) -> dict:
    """Reduce (H, A, P) -> (H, A) by collapsing the polar dimension.
    Returns a dict of Z statistics: zmin, zmax, zmean, zvar."""
    return {
        "zmin":  rays.min(axis=2),
        "zmax":  rays.max(axis=2),
        "zmean": rays.mean(axis=2),
        "zvar":  rays.var(axis=2),
    }


# ---------------------------------------------------------------------------
# PCA subspace per vantage point
# ---------------------------------------------------------------------------
def heatmap_to_subspace(heatmap: np.ndarray,
                        var_target: float = 0.95
                        ) -> np.ndarray:
    """Run PCA on the heatmap (rows=heights, cols=azimuth). Return the
    matrix of principal-component vectors (cols are PCs) accounting for
    `var_target` of variance. The PC vectors live in azimuth-space."""
    # Treat heights as samples, azimuth as features (n_samples=H, n_features=A)
    X = heatmap
    if X.shape[0] < 2:
        # Not enough samples - return single mean direction
        return X.mean(axis=0, keepdims=True).T
    # Center
    Xc = X - X.mean(axis=0, keepdims=True)
    # Cap n_components at min(n_samples, n_features)
    n_max = min(Xc.shape)
    pca = PCA(n_components=n_max)
    pca.fit(Xc)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, n_max))
    return pca.components_[:k].T  # shape (A, k)


# ---------------------------------------------------------------------------
# Saliency measures (paper eqs. 1 and 2)
# ---------------------------------------------------------------------------
def krzanowski_distance(L: np.ndarray, M: np.ndarray) -> float:
    """Eq. 1 from the paper. L: (A, k1), M: (A, k2). Returns a positive
    'distance' = 1 / sum(cos^2 angles). Clipped to avoid div-by-zero."""
    # Normalise columns to unit length (PCA components already are, but
    # be defensive after slicing).
    Ln = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)
    # Trace(L^T M M^T L) = sum of squared cosines of principal angles
    LtM = Ln.T @ Mn
    sim = np.trace(LtM @ LtM.T)
    sim = max(sim, 1e-6)
    return float(1.0 / sim)


def subspace_entropy(L: np.ndarray, n_bins: int = 30) -> float:
    """Eq. 2 from the paper. Shannon entropy of the histogram of all
    subspace coefficients."""
    vals = L.flatten()
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=n_bins, density=False)
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def global_saliency(subspaces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each subspace i, compute:
        angle_sal[i]   = mean Krzanowski distance to all others
        entropy_sal[i] = subspace_entropy(subspaces[i])
        combined[i]    = scaled angle_sal + scaled entropy_sal
    """
    n = len(subspaces)
    angle_sal = np.zeros(n)
    entropy_sal = np.zeros(n)
    for i in range(n):
        ds = []
        for j in range(n):
            if i == j:
                continue
            ds.append(krzanowski_distance(subspaces[i], subspaces[j]))
        angle_sal[i] = float(np.mean(ds)) if ds else 0.0
        entropy_sal[i] = subspace_entropy(subspaces[i])

    a_scaled = _minmax(angle_sal)
    e_scaled = _minmax(entropy_sal)
    combined = a_scaled + e_scaled
    return a_scaled, e_scaled, combined


def local_saliency(subspaces: list[np.ndarray],
                   region_size: int) -> np.ndarray:
    """Compute local saliency by partitioning the ordered subspace list
    into contiguous regions of size `region_size` and running global
    saliency within each region. Mirrors paper section 4.2."""
    n = len(subspaces)
    out = np.zeros(n)
    for start in range(0, n, region_size):
        end = min(start + region_size, n)
        if end - start < 2:
            continue
        a, e, c = global_saliency(subspaces[start:end])
        out[start:end] = c
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Scene")
    scene_choice = st.selectbox(
        "Built-in scene",
        list(SCENE_FACTORIES.keys()),
        index=2,
        help="Choose a pre-built test scene. The Villa-like model is the closest "
             "analogue to the paper's Villa Savoye experiment.",
    )
    uploaded = st.file_uploader(
        "…or upload your own mesh (.obj / .ply / .stl)",
        type=["obj", "ply", "stl"],
    )

    st.header("2 · Vantage points")
    n_points = st.slider("Number of vantage points", 10, 200, 60, step=10,
                          help="Paper used 300; smaller is much faster.")
    sampling = st.radio("Sampling pattern",
                        ["Snake path (ordered)", "Grid (unordered)"],
                        index=0)
    eye_height = st.slider("Eye height (m)", 0.5, 2.5, 1.6, step=0.1)

    st.header("3 · Isovist resolution")
    n_az = st.select_slider("Azimuth samples",
                             options=[24, 36, 60, 90, 120], value=60)
    n_pol = st.select_slider("Polar samples",
                              options=[12, 18, 30, 60], value=30)
    n_h = st.slider("Height layers", 2, 10, 5)
    max_ray = st.slider("Max ray length (m, BLI clip)", 5, 100, 30)

    st.header("4 · Saliency / ML")
    region_size = st.slider("Local region size (sections)", 5, 60, 15)
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    do_tsne = st.checkbox("Add t-SNE 2D embedding", value=True)

    run = st.button("▶ Run analysis", type="primary", use_container_width=True)


# Lazy-load the mesh
@st.cache_data(show_spinner=False)
def load_mesh(scene_name: str, uploaded_bytes: Optional[bytes],
              uploaded_ext: Optional[str]) -> trimesh.Trimesh:
    if uploaded_bytes is not None:
        f = io.BytesIO(uploaded_bytes)
        m = trimesh.load(f, file_type=uploaded_ext, force="mesh")
    else:
        m = SCENE_FACTORIES[scene_name]()
    if not isinstance(m, trimesh.Trimesh):
        # If a Scene is returned, dump to a single mesh
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


up_bytes = uploaded.read() if uploaded is not None else None
up_ext = uploaded.name.split(".")[-1].lower() if uploaded is not None else None
mesh = load_mesh(scene_choice, up_bytes, up_ext)

col_overview_a, col_overview_b = st.columns([2, 1])
with col_overview_a:
    st.subheader("Scene overview")
    st.write(
        f"Mesh: **{len(mesh.vertices):,} vertices**, "
        f"**{len(mesh.faces):,} faces**. "
        f"Bounds (m): X {mesh.bounds[0,0]:.1f}..{mesh.bounds[1,0]:.1f}, "
        f"Y {mesh.bounds[0,1]:.1f}..{mesh.bounds[1,1]:.1f}, "
        f"Z {mesh.bounds[0,2]:.1f}..{mesh.bounds[1,2]:.1f}."
    )

with col_overview_b:
    # Quick top-down preview
    fig, ax = plt.subplots(figsize=(3, 3))
    # Use 2D silhouette of triangle vertices
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1],
               s=0.5, alpha=0.3, color="steelblue")
    ax.set_aspect("equal")
    ax.set_title("Top-down vertex scatter")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    st.pyplot(fig)

st.divider()

if not run:
    st.info("👈 Configure the run in the sidebar, then click **Run analysis**. "
            "Smaller resolutions finish in seconds; the defaults are a "
            "reasonable accuracy/speed compromise.")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
status = st.empty()
prog = st.progress(0.0)
t0 = time.time()

# 1. Vantage points
status.write("Sampling vantage points…")
if sampling.startswith("Snake"):
    vps = sample_path_inside(mesh, n_points, eye_height)
else:
    vps = sample_grid_inside(mesh, n_points, eye_height)

if len(vps) < 4:
    st.error(
        "Could not place enough vantage points inside the mesh. "
        "Try a different scene, lower eye height, or increase the "
        "vantage-point count."
    )
    st.stop()

st.write(f"Placed **{len(vps)}** vantage points.")
prog.progress(0.05)

# 2. Cast isovists
status.write("Casting 3D isovists (this is the slow part)…")
cfg = IsovistConfig(n_azimuth=n_az, n_polar=n_pol, n_heights=n_h,
                    h_min=0.4, h_max=eye_height, max_ray=float(max_ray))

intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
all_rays = []
for i, p in enumerate(vps):
    all_rays.append(cast_isovist(mesh, p, cfg, intersector))
    prog.progress(0.05 + 0.55 * (i + 1) / len(vps))
all_rays = np.stack(all_rays, axis=0)  # (N, H, A, P)

# 3. Z statistics
status.write("Computing Z statistics & heat maps…")
zvar_maps = []
zmean_maps = []
for rays in all_rays:
    Zs = compute_z_matrices(rays)
    zvar_maps.append(Zs["zvar"])
    zmean_maps.append(Zs["zmean"])
zvar_maps = np.stack(zvar_maps, axis=0)   # (N, H, A)
zmean_maps = np.stack(zmean_maps, axis=0) # (N, H, A)
prog.progress(0.7)

# 4. Per-vantage-point PCA subspaces
status.write("Computing PCA subspaces per vantage point…")
subspaces = [heatmap_to_subspace(z) for z in zvar_maps]
prog.progress(0.8)

# 5. Saliency
status.write("Computing global & local saliency…")
g_angle, g_entropy, g_combined = global_saliency(subspaces)
l_combined = local_saliency(subspaces, region_size=region_size)
prog.progress(0.9)

# 6. ML clustering on the heat-map feature vectors
status.write("Clustering vantage points…")
feat = zvar_maps.reshape(len(zvar_maps), -1)
feat_scaled = StandardScaler().fit_transform(feat)
pca2 = PCA(n_components=2).fit_transform(feat_scaled)
km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(feat_scaled)
labels = km.labels_

tsne2 = None
if do_tsne and len(feat_scaled) >= 5:
    perp = max(5, min(30, len(feat_scaled) // 3))
    tsne2 = TSNE(n_components=2, perplexity=perp, random_state=0,
                 init="pca", learning_rate="auto").fit_transform(feat_scaled)

prog.progress(1.0)
elapsed = time.time() - t0
status.success(f"Done in {elapsed:.1f} s")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
st.header("Results")

# --- (a) Saliency curves ----------------------------------------------------
st.subheader("Global & local saliency per vantage point")
df = pd.DataFrame({
    "index": np.arange(len(vps)),
    "angle_saliency": g_angle,
    "entropy_saliency": g_entropy,
    "global_combined": g_combined,
    "local_combined": l_combined,
    "cluster": labels,
    "x": vps[:, 0], "y": vps[:, 1], "z": vps[:, 2],
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["index"], df["global_combined"], label="Global combined",
        color="black", lw=1.5)
ax.plot(df["index"], df["local_combined"], label="Local combined",
        color="tab:red", lw=1.0, alpha=0.8)
ax.plot(df["index"], df["angle_saliency"], label="Angle (Krzanowski)",
        color="tab:blue", lw=0.8, ls="--")
ax.plot(df["index"], df["entropy_saliency"], label="Entropy",
        color="tab:green", lw=0.8, ls=":")
ax.set_xlabel("Vantage point index")
ax.set_ylabel("Saliency (min-max scaled)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Per-section saliency (analogue of paper Fig. 4a / 5a)")
st.pyplot(fig)

top_g = int(np.argmax(g_combined))
bot_g = int(np.argmin(g_combined))
c1, c2, c3 = st.columns(3)
c1.metric("Most globally salient index", top_g,
          f"score={g_combined[top_g]:.2f}")
c2.metric("Least globally salient index", bot_g,
          f"score={g_combined[bot_g]:.2f}")
c3.metric("Median global score", f"{np.median(g_combined):.3f}")

# --- (b) Floor-plan saliency map -------------------------------------------
st.subheader("Saliency on the floor plan")
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
sc1 = axs[0].scatter(df["x"], df["y"], c=df["global_combined"],
                     cmap="magma", s=60, edgecolors="k", linewidths=0.3)
axs[0].set_title("Global combined saliency")
axs[0].set_xlabel("X (m)"); axs[0].set_ylabel("Y (m)")
axs[0].set_aspect("equal")
plt.colorbar(sc1, ax=axs[0], shrink=0.8)

sc2 = axs[1].scatter(df["x"], df["y"], c=df["cluster"],
                     cmap="tab10", s=60, edgecolors="k", linewidths=0.3)
axs[1].set_title(f"KMeans clusters (k={n_clusters})")
axs[1].set_xlabel("X (m)"); axs[1].set_ylabel("Y (m)")
axs[1].set_aspect("equal")
plt.colorbar(sc2, ax=axs[1], shrink=0.8, ticks=range(n_clusters))

st.pyplot(fig)

# --- (c) Heat maps for top / bottom vantage points -------------------------
st.subheader("Z_var heat maps: most vs. least salient")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
im0 = axs[0].imshow(zvar_maps[top_g], aspect="auto", cmap="inferno",
                    origin="lower",
                    extent=[0, 360, cfg.h_min, cfg.h_max])
axs[0].set_title(f"Most salient (idx {top_g}) — Z_var")
axs[0].set_xlabel("Azimuth (deg)")
axs[0].set_ylabel("Eye height (m)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(zvar_maps[bot_g], aspect="auto", cmap="inferno",
                    origin="lower",
                    extent=[0, 360, cfg.h_min, cfg.h_max])
axs[1].set_title(f"Least salient (idx {bot_g}) — Z_var")
axs[1].set_xlabel("Azimuth (deg)")
axs[1].set_ylabel("Eye height (m)")
plt.colorbar(im1, ax=axs[1])
st.pyplot(fig)
st.caption("Analogue of paper Figs. 3 and 4c. Brighter / more textured "
          "heat maps correspond to richer geometric variation across "
          "azimuth and eye-height.")

# --- (d) Embedding plots ---------------------------------------------------
st.subheader("ML embedding of vantage-point feature space")
fig, axs = plt.subplots(1, 2 if tsne2 is not None else 1,
                         figsize=(11, 4),
                         squeeze=False)
ax = axs[0, 0]
sc = ax.scatter(pca2[:, 0], pca2[:, 1],
                c=g_combined, cmap="magma",
                s=60, edgecolors="k", linewidths=0.3)
ax.set_title("PCA(2) of Z_var heat maps, coloured by global saliency")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.colorbar(sc, ax=ax)
if tsne2 is not None:
    ax = axs[0, 1]
    sc = ax.scatter(tsne2[:, 0], tsne2[:, 1],
                    c=labels, cmap="tab10",
                    s=60, edgecolors="k", linewidths=0.3)
    ax.set_title("t-SNE(2), coloured by KMeans cluster")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.colorbar(sc, ax=ax, ticks=range(n_clusters))
st.pyplot(fig)

# --- (e) Cluster summary ----------------------------------------------------
st.subheader("Cluster-level saliency")
cluster_summary = (
    df.groupby("cluster")
      .agg(n=("index", "size"),
           mean_global=("global_combined", "mean"),
           mean_local=("local_combined", "mean"),
           mean_x=("x", "mean"),
           mean_y=("y", "mean"))
      .round(3)
)
st.dataframe(cluster_summary, use_container_width=True)

# --- (f) Raw data + download ------------------------------------------------
with st.expander("Per-vantage-point data table"):
    st.dataframe(df.round(4), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv,
                       file_name="vantage_saliency.csv",
                       mime="text/csv")

# ---------------------------------------------------------------------------
# Footnotes
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
**Method recap** (after Bhatia, Chalup & Ostwald 2012):

1. *3D isovist*: from each vantage point, cast rays across azimuth
   `[0°, 360°)` × polar `[0°, 180°]` at multiple eye heights.
2. *Heat map*: collapse the polar dimension via a statistic (here
   variance, `Z_var`) → `(height × azimuth)` matrix per vantage point.
3. *Subspace*: PCA on the heat map; keep components covering 95% of
   variance.
4. *Saliency*:
   - **Angle (Krzanowski 1979)** – inverse of `tr(LᵀMMᵀL)`, paper Eq. 1.
   - **Entropy (Shannon 1948)** – of the histogram of subspace
     coefficients, paper Eq. 2.
   - Min-max scale each, then sum → *combined saliency*.
5. *Global vs Local* – global compares each vantage point against all
   others; local restricts the comparison to a sliding region of `n`
   sections (paper used 50 and 100).

**Caveats / honest limits**
- The built-in 'Villa-like' scene is *not* a faithful Villa Savoye
  model; it is a simplified stand-in to demonstrate the method.
- Trimesh's Python ray-caster is much slower than the paper's
  SketchUp-API approach. Resolutions and vantage-point counts are
  therefore lower by default.
- Inside-the-mesh testing relies on `trimesh.contains`, which can fail
  silently on non-watertight uploads — in that case all sampled points
  are kept.
- BLI behaviour (clip at `max_ray`) is implemented; FLI variant is not
  yet exposed in the UI.
"""
)
