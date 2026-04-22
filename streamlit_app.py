"""
3D Spatial Visibility Analyzer with Machine Learning
=====================================================
A comprehensive Streamlit app for architectural spatial-visibility research.

Features
--------
1. Interactive 3D room builder with obstacles
2. Vectorized ray-casting engine (analytical AABB intersection)
3. 2D (planar) vs 3D (volumetric) isovist comparison
4. "Corner-overestimation" diagnostic map
5. ML surrogate model (MLP) that learns the expensive ray-cast function
6. Perception-proxy scoring (openness, enclosure, corner-bias index)
7. Export of grid results to CSV

Run with:
    pip install streamlit numpy pandas plotly scikit-learn
    streamlit run app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Spatial Visibility + ML",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .metric-card {background:#f5f7fa; border-radius:8px; padding:12px;}
        h1, h2, h3 {color:#1f2937;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """Axis-aligned bounding box."""
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    name: str = "obstacle"

    @property
    def mn(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def mx(self) -> np.ndarray:
        return np.array([self.xmax, self.ymax, self.zmax])


@dataclass
class Room:
    width: float   # x extent
    length: float  # y extent
    height: float  # z extent
    obstacles: List[Box] = field(default_factory=list)

    @property
    def mn(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    @property
    def mx(self) -> np.ndarray:
        return np.array([self.width, self.length, self.height])


# ---------------------------------------------------------------------------
# Ray-casting engine (vectorized)
# ---------------------------------------------------------------------------
def _inv_dir(directions: np.ndarray) -> np.ndarray:
    """Safe reciprocal of ray directions to avoid /0."""
    safe = np.where(np.abs(directions) > 1e-9, directions, 1e-9)
    return 1.0 / safe


def batch_ray_cast(origin: np.ndarray, directions: np.ndarray, room: Room) -> np.ndarray:
    """
    For each ray from `origin` in `directions`, return the distance to the
    closest intersection with either the room walls (exit) or an obstacle.
    directions: (N, 3) unit vectors
    returns:    (N,) distances
    """
    inv = _inv_dir(directions)

    # Room walls – we are INSIDE so take the exit distance
    t1 = (room.mn - origin) * inv
    t2 = (room.mx - origin) * inv
    wall_dist = np.min(np.maximum(t1, t2), axis=1)

    min_dist = wall_dist.copy()

    for obs in room.obstacles:
        t1 = (obs.mn - origin) * inv
        t2 = (obs.mx - origin) * inv
        tn = np.max(np.minimum(t1, t2), axis=1)
        tf = np.min(np.maximum(t1, t2), axis=1)
        hit = (tn <= tf) & (tf > 1e-6)
        # Use near intersection if we're outside the obstacle, far if inside.
        dist = np.where(tn > 1e-6, tn, np.where(hit, tf, np.inf))
        dist = np.where(hit, dist, np.inf)
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform unit vectors on the sphere."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    y = 1 - 2 * i / max(n - 1, 1)
    r = np.sqrt(np.maximum(0.0, 1 - y * y))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def fibonacci_circle(n: int) -> np.ndarray:
    """Unit vectors in the horizontal (xy) plane."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)


# ---------------------------------------------------------------------------
# Visibility metrics
# ---------------------------------------------------------------------------
def visibility_at_point(
    point: np.ndarray,
    room: Room,
    n_rays: int = 256,
    mode: str = "3d",
) -> Dict[str, float]:
    """
    Compute visibility descriptors at a single observer position.
    mode: '3d' (full sphere) or '2d' (horizontal slice at observer height)
    """
    if mode == "3d":
        dirs = fibonacci_sphere(n_rays)
        d = batch_ray_cast(point, dirs, room)
        solid = 4 * np.pi / n_rays
        vol = float(np.sum(d ** 3 / 3.0) * solid)          # isovist volume
        return {
            "mean_dist": float(np.mean(d)),
            "max_dist": float(np.max(d)),
            "min_dist": float(np.min(d)),
            "isovist": vol,
            "n_rays": n_rays,
        }
    else:
        dirs = fibonacci_circle(n_rays)
        d = batch_ray_cast(point, dirs, room)
        ang = 2 * np.pi / n_rays
        area = float(np.sum(d ** 2 / 2.0) * ang)           # isovist area
        return {
            "mean_dist": float(np.mean(d)),
            "max_dist": float(np.max(d)),
            "min_dist": float(np.min(d)),
            "isovist": area,
            "n_rays": n_rays,
        }


def visibility_grid(
    room: Room,
    resolution: int = 20,
    eye_height: float = 1.6,
    n_rays: int = 128,
    mode: str = "3d",
) -> Dict[str, np.ndarray]:
    """Dense grid of visibility values across the room floor."""
    xs = np.linspace(0.3, room.width - 0.3, resolution)
    ys = np.linspace(0.3, room.length - 0.3, resolution)
    X, Y = np.meshgrid(xs, ys)
    iso = np.zeros_like(X)
    mean_d = np.zeros_like(X)
    valid = np.ones_like(X, dtype=bool)

    for i in range(resolution):
        for j in range(resolution):
            p = np.array([X[i, j], Y[i, j], eye_height])

            # Skip points inside obstacles
            inside = False
            for o in room.obstacles:
                if (o.xmin <= p[0] <= o.xmax and
                        o.ymin <= p[1] <= o.ymax and
                        o.zmin <= p[2] <= o.zmax):
                    inside = True
                    break
            if inside:
                iso[i, j] = np.nan
                mean_d[i, j] = np.nan
                valid[i, j] = False
                continue

            r = visibility_at_point(p, room, n_rays=n_rays, mode=mode)
            iso[i, j] = r["isovist"]
            mean_d[i, j] = r["mean_dist"]

    return {"X": X, "Y": Y, "isovist": iso, "mean_dist": mean_d, "valid": valid}


# ---------------------------------------------------------------------------
# Room presets
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Room] = {
    "Empty hall (8×6×3)": Room(8.0, 6.0, 3.0, []),
    "L-shaped room": Room(
        10.0, 8.0, 3.0,
        [Box(6.0, 0.0, 0.0, 10.0, 4.0, 3.0, "wall")],
    ),
    "Office with central pillar": Room(
        10.0, 8.0, 3.0,
        [Box(4.5, 3.5, 0.0, 5.5, 4.5, 3.0, "pillar")],
    ),
    "Open-plan with furniture": Room(
        12.0, 9.0, 3.0,
        [
            Box(2.0, 2.0, 0.0, 4.0, 3.0, 1.2, "desk"),
            Box(7.0, 2.0, 0.0, 9.0, 3.0, 1.2, "desk"),
            Box(2.0, 5.5, 0.0, 4.0, 6.5, 1.2, "desk"),
            Box(7.0, 5.5, 0.0, 9.0, 6.5, 1.2, "desk"),
            Box(5.5, 3.8, 0.0, 6.5, 5.2, 1.8, "partition"),
        ],
    ),
    "Narrow corridor": Room(
        12.0, 3.0, 2.7,
        [
            Box(3.0, 0.0, 0.0, 4.0, 1.2, 2.7, "bump"),
            Box(8.0, 1.8, 0.0, 9.0, 3.0, 2.7, "bump"),
        ],
    ),
    "Atrium with low ceiling corner": Room(
        10.0, 10.0, 4.0,
        [
            Box(0.0, 0.0, 2.2, 3.0, 3.0, 4.0, "dropped_ceiling"),
            Box(7.0, 7.0, 0.0, 10.0, 10.0, 1.0, "platform"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def box_mesh(b: Box, color: str, name: str, opacity: float = 0.5) -> go.Mesh3d:
    x = [b.xmin, b.xmax, b.xmax, b.xmin, b.xmin, b.xmax, b.xmax, b.xmin]
    y = [b.ymin, b.ymin, b.ymax, b.ymax, b.ymin, b.ymin, b.ymax, b.ymax]
    z = [b.zmin, b.zmin, b.zmin, b.zmin, b.zmax, b.zmax, b.zmax, b.zmax]
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 2, 3]
    j = [1, 2, 3, 2, 5, 3, 5, 7, 6, 7, 6, 7]
    k = [2, 3, 4, 5, 6, 7, 6, 6, 1, 3, 7, 4]
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        flatshading=True, showscale=False,
    )


def plot_room_3d(room: Room, observer: np.ndarray | None = None,
                 ray_dirs: np.ndarray | None = None,
                 ray_dists: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()

    # Floor & ceiling as light meshes
    fig.add_trace(go.Mesh3d(
        x=[0, room.width, room.width, 0],
        y=[0, 0, room.length, room.length],
        z=[0, 0, 0, 0],
        i=[0], j=[1], k=[2], color="#d5dbe5", opacity=0.35, name="floor",
    ))
    fig.add_trace(go.Mesh3d(
        x=[0, room.width, room.width, 0],
        y=[0, 0, room.length, room.length],
        z=[room.height] * 4,
        i=[0, 0], j=[1, 2], k=[2, 3], color="#e5e7eb", opacity=0.15, name="ceiling",
    ))

    # Obstacles
    palette = ["#ef4444", "#f59e0b", "#10b981", "#6366f1", "#ec4899", "#14b8a6"]
    for idx, obs in enumerate(room.obstacles):
        fig.add_trace(box_mesh(obs, palette[idx % len(palette)], obs.name, 0.55))

    # Rays
    if observer is not None and ray_dirs is not None and ray_dists is not None:
        xs, ys, zs = [], [], []
        for d, r in zip(ray_dirs, ray_dists):
            end = observer + d * r
            xs += [observer[0], end[0], None]
            ys += [observer[1], end[1], None]
            zs += [observer[2], end[2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(59,130,246,0.35)", width=1),
            name="rays", hoverinfo="skip",
        ))

    # Observer
    if observer is not None:
        fig.add_trace(go.Scatter3d(
            x=[observer[0]], y=[observer[1]], z=[observer[2]],
            mode="markers", marker=dict(size=7, color="#111827", symbol="diamond"),
            name="observer",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, room.width], title="X (m)"),
            yaxis=dict(range=[0, room.length], title="Y (m)"),
            zaxis=dict(range=[0, room.height], title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=560,
        showlegend=False,
    )
    return fig


def plot_heatmap(grid: Dict[str, np.ndarray], title: str, colorscale: str = "Viridis") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        x=grid["X"][0], y=grid["Y"][:, 0], z=grid["isovist"],
        colorscale=colorscale, colorbar=dict(title="isovist"),
    ))
    fig.update_layout(
        title=title, height=420,
        xaxis_title="X (m)", yaxis_title="Y (m)",
        margin=dict(l=10, r=10, b=10, t=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------------
# ML surrogate
# ---------------------------------------------------------------------------
def build_training_set(room: Room, n_samples: int, n_rays: int, mode: str,
                       eye_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Random-sample the room and compute ground-truth visibility."""
    rng = np.random.default_rng(42)
    Xs, ys = [], []
    attempts = 0
    while len(Xs) < n_samples and attempts < n_samples * 5:
        attempts += 1
        x = rng.uniform(0.3, room.width - 0.3)
        y = rng.uniform(0.3, room.length - 0.3)
        p = np.array([x, y, eye_height])
        inside = any(
            o.xmin <= x <= o.xmax and o.ymin <= y <= o.ymax
            for o in room.obstacles
        )
        if inside:
            continue
        r = visibility_at_point(p, room, n_rays=n_rays, mode=mode)
        Xs.append([x, y])
        ys.append(r["isovist"])
    return np.array(Xs), np.array(ys)


def train_surrogate(X: np.ndarray, y: np.ndarray,
                    hidden=(64, 64, 32), seed=0):
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        learning_rate_init=1e-3,
        max_iter=800,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(scaler_x.transform(X), scaler_y.transform(y.reshape(-1, 1)).ravel())
    return model, scaler_x, scaler_y


def surrogate_grid(room: Room, model, scaler_x, scaler_y, resolution: int = 60) -> Dict[str, np.ndarray]:
    xs = np.linspace(0.3, room.width - 0.3, resolution)
    ys = np.linspace(0.3, room.length - 0.3, resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pred = model.predict(scaler_x.transform(pts))
    pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
    Z = pred.reshape(X.shape)
    # Mask obstacle interior
    valid = np.ones_like(Z, dtype=bool)
    for i in range(resolution):
        for j in range(resolution):
            x_, y_ = X[i, j], Y[i, j]
            for o in room.obstacles:
                if o.xmin <= x_ <= o.xmax and o.ymin <= y_ <= o.ymax:
                    Z[i, j] = np.nan
                    valid[i, j] = False
                    break
    return {"X": X, "Y": Y, "isovist": Z, "valid": valid}


# ---------------------------------------------------------------------------
# Sidebar – global config
# ---------------------------------------------------------------------------
st.sidebar.title("🏛️ Spatial Visibility Lab")
st.sidebar.caption("3D isovist analysis · ML surrogate · perception proxy")

preset_name = st.sidebar.selectbox("Room preset", list(PRESETS.keys()), index=3)
room = PRESETS[preset_name]

with st.sidebar.expander("Room dimensions (override)"):
    w = st.number_input("Width (m)", 2.0, 30.0, float(room.width), 0.5)
    l = st.number_input("Length (m)", 2.0, 30.0, float(room.length), 0.5)
    h = st.number_input("Height (m)", 2.0, 8.0, float(room.height), 0.1)
    room = Room(w, l, h, room.obstacles)

with st.sidebar.expander("Analysis parameters", expanded=True):
    n_rays = st.select_slider("Rays per point", [64, 128, 256, 512, 1024], 256)
    eye_h = st.slider("Eye height (m)", 0.5, 2.2, 1.6, 0.05)
    grid_res = st.slider("Grid resolution", 10, 40, 22, 1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built as a demo of the ideas in the 3D-visibility literature: "
    "planar metrics overestimate connectivity at corners; "
    "3D ray-casting corrects for ceiling, furniture, and vertical obstruction."
)


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("3D Spatial Visibility Analyzer with ML")
st.markdown(
    "Compare **planar (2D)** and **volumetric (3D)** isovists, diagnose where "
    "2D metrics over-state spatial connectivity, and train a neural **surrogate** "
    "that replaces expensive ray-casting with a fast regression model."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Point Analysis",
    "🌡️ Grid & 2D-vs-3D",
    "🧠 ML Surrogate",
    "🫥 Perception Proxy",
    "ℹ️ Theory",
])


# ---------------------------------------------------------------------------
# Tab 1 – Point analysis
# ---------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Observer position")
        ox = st.slider("X (m)", 0.1, float(room.width) - 0.1, float(room.width) / 2, 0.1)
        oy = st.slider("Y (m)", 0.1, float(room.length) - 0.1, float(room.length) / 2, 0.1)
        oz = st.slider("Z / eye height (m)", 0.2, float(room.height) - 0.1, eye_h, 0.05)
        obs = np.array([ox, oy, oz])

        mode = st.radio("Mode", ["3D (sphere)", "2D (horizontal)"], horizontal=True)
        mode_key = "3d" if mode.startswith("3D") else "2d"

        show_rays = st.checkbox("Draw rays", True)
        n_show = st.slider("Rays to draw", 32, 400, 120)

    # Compute
    dirs_full = fibonacci_sphere(n_rays) if mode_key == "3d" else fibonacci_circle(n_rays)
    t0 = time.perf_counter()
    dists = batch_ray_cast(obs, dirs_full, room)
    dt = (time.perf_counter() - t0) * 1000

    mean_d = float(np.mean(dists))
    max_d = float(np.max(dists))
    min_d = float(np.min(dists))
    if mode_key == "3d":
        solid = 4 * np.pi / n_rays
        isovist = float(np.sum(dists ** 3 / 3.0) * solid)
        iso_label = "Isovist volume (m³)"
    else:
        ang = 2 * np.pi / n_rays
        isovist = float(np.sum(dists ** 2 / 2.0) * ang)
        iso_label = "Isovist area (m²)"

    with c1:
        st.markdown("### Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean ray dist", f"{mean_d:.2f} m")
        m2.metric("Max ray dist", f"{max_d:.2f} m")
        m3.metric("Min ray dist", f"{min_d:.2f} m")
        st.metric(iso_label, f"{isovist:.2f}")
        st.caption(f"Computed {n_rays} rays in {dt:.1f} ms")

    with c2:
        show_dirs = dirs_full[:n_show] if show_rays else None
        show_dists = dists[:n_show] if show_rays else None
        fig = plot_room_3d(room, obs, show_dirs, show_dists)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 – Grid & 2D vs 3D comparison
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Dense visibility field — the key comparison")
    st.caption(
        "Compute isovist values on a grid of floor points. The "
        "**difference map** highlights regions (especially corners) where "
        "the 2D planar metric overstates spatial openness."
    )

    if st.button("▶ Run grid analysis", type="primary"):
        with st.spinner(f"Ray-casting {grid_res*grid_res} points × {n_rays} rays × 2 modes…"):
            t0 = time.perf_counter()
            g3 = visibility_grid(room, grid_res, eye_h, n_rays, "3d")
            g2 = visibility_grid(room, grid_res, eye_h, n_rays, "2d")
            dt = time.perf_counter() - t0
        st.session_state["grid_3d"] = g3
        st.session_state["grid_2d"] = g2
        st.success(f"Grid ready in {dt:.2f} s")

    if "grid_3d" in st.session_state:
        g3 = st.session_state["grid_3d"]
        g2 = st.session_state["grid_2d"]

        # Normalize both for fair comparison
        def _norm(a):
            a = a.copy()
            m = np.nanmax(a)
            return a / m if m > 0 else a

        n3 = _norm(g3["isovist"])
        n2 = _norm(g2["isovist"])
        diff = n2 - n3          # positive ⇒ 2D overstates

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                plot_heatmap({"X": g2["X"], "Y": g2["Y"], "isovist": n2},
                             "2D isovist (normalised)"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_heatmap({"X": g3["X"], "Y": g3["Y"], "isovist": n3},
                             "3D isovist (normalised)"),
                use_container_width=True,
            )
        with c3:
            fig = go.Figure(go.Heatmap(
                x=g3["X"][0], y=g3["Y"][:, 0], z=diff,
                colorscale="RdBu_r", zmid=0,
                colorbar=dict(title="2D − 3D"),
            ))
            fig.update_layout(
                title="Overestimation map (red = 2D overstates)",
                height=420,
                xaxis_title="X (m)", yaxis_title="Y (m)",
                margin=dict(l=10, r=10, b=10, t=40),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        mean_over = float(np.nanmean(diff))
        pos_frac = float(np.nanmean(diff > 0.05))
        st.markdown(
            f"**Summary:** Mean 2D-3D difference = `{mean_over:+.3f}` · "
            f"`{pos_frac*100:.1f}%` of points show 2D-overestimation > 0.05."
        )

        # Export
        df = pd.DataFrame({
            "x": g3["X"].ravel(),
            "y": g3["Y"].ravel(),
            "isovist_3d": g3["isovist"].ravel(),
            "isovist_2d_slice": g2["isovist"].ravel(),
            "mean_ray_dist_3d": g3["mean_dist"].ravel(),
            "mean_ray_dist_2d": g2["mean_dist"].ravel(),
        })
        st.download_button(
            "⬇ Download grid as CSV",
            df.to_csv(index=False).encode(),
            file_name="visibility_grid.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab 3 – ML surrogate
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Neural surrogate model for fast visibility prediction")
    st.caption(
        "Train an MLP on a small set of ground-truth points, then use it to "
        "render a dense field almost instantly. Demonstrates the speed-up ML "
        "can bring to generative and real-time design tools."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        n_train = st.slider("Training samples", 50, 1000, 250, 50)
        n_rays_train = st.select_slider("Rays per training sample", [64, 128, 256, 512], 256)
        mode_train = st.radio("Target metric", ["3d", "2d"], horizontal=True, index=0)
        hidden_str = st.text_input("Hidden layers (comma-sep)", "64,64,32")
        surrogate_res = st.slider("Surrogate render resolution", 30, 120, 80)

        run = st.button("🧪 Train surrogate", type="primary")

    if run:
        hidden = tuple(int(s.strip()) for s in hidden_str.split(",") if s.strip())
        with st.spinner("Generating ground-truth training set…"):
            t0 = time.perf_counter()
            Xtr, ytr = build_training_set(room, n_train, n_rays_train, mode_train, eye_h)
            gt_time = time.perf_counter() - t0

        with st.spinner("Fitting MLP…"):
            t0 = time.perf_counter()
            model, sx, sy = train_surrogate(Xtr, ytr, hidden=hidden)
            fit_time = time.perf_counter() - t0

        with st.spinner("Predicting dense field…"):
            t0 = time.perf_counter()
            gs = surrogate_grid(room, model, sx, sy, resolution=surrogate_res)
            pred_time = time.perf_counter() - t0

        # Evaluate on a held-out grid
        val_grid = visibility_grid(room, 16, eye_h, 128, mode_train)
        valid = val_grid["valid"]
        pred_val = model.predict(sx.transform(
            np.stack([val_grid["X"].ravel(), val_grid["Y"].ravel()], axis=1)
        ))
        pred_val = sy.inverse_transform(pred_val.reshape(-1, 1)).ravel().reshape(val_grid["X"].shape)
        true_val = val_grid["isovist"]
        mask = valid & ~np.isnan(true_val)
        r2 = r2_score(true_val[mask], pred_val[mask])
        mae = mean_absolute_error(true_val[mask], pred_val[mask])

        with c2:
            fig = go.Figure(go.Heatmap(
                x=gs["X"][0], y=gs["Y"][:, 0], z=gs["isovist"],
                colorscale="Viridis",
            ))
            fig.add_trace(go.Scatter(
                x=Xtr[:, 0], y=Xtr[:, 1], mode="markers",
                marker=dict(color="red", size=5, line=dict(width=0.5, color="white")),
                name="training pts",
            ))
            fig.update_layout(
                title=f"Surrogate prediction ({surrogate_res}×{surrogate_res})",
                height=500,
                xaxis_title="X (m)", yaxis_title="Y (m)",
                margin=dict(l=10, r=10, b=10, t=40),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (hold-out)", f"{r2:.3f}")
        m2.metric("MAE", f"{mae:.3f}")
        m3.metric("Ray-cast time", f"{gt_time:.1f} s")
        m4.metric("Surrogate time", f"{pred_time*1000:.0f} ms",
                  delta=f"{gt_time/max(pred_time,1e-6):.0f}× faster")

        st.info(
            f"Surrogate evaluated the entire {surrogate_res}×{surrogate_res} "
            f"grid in **{pred_time*1000:.0f} ms** — a speed-up useful for "
            f"real-time design feedback and generative optimisation loops."
        )


# ---------------------------------------------------------------------------
# Tab 4 – Perception proxy
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Perception proxy: openness, enclosure, corner bias")
    st.caption(
        "A simple heuristic combining 3D ray statistics into perceptual "
        "descriptors. In a real study you would regress these features "
        "against human ratings collected in VR."
    )

    if "grid_3d" not in st.session_state:
        st.warning("Run the grid analysis in Tab 2 first.")
    else:
        g3 = st.session_state["grid_3d"]
        g2 = st.session_state["grid_2d"]
        iso3 = g3["isovist"]
        iso2 = g2["isovist"]

        # Normalised descriptors
        openness = iso3 / np.nanmax(iso3)
        enclosure = 1 - openness
        # Corner bias = planar overstates relative to 3D
        overest = (iso2 / np.nanmax(iso2)) - openness

        comfort = 0.55 * openness + 0.25 * (1 - np.abs(openness - 0.6)) - 0.3 * np.clip(overest, 0, None)

        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            "Openness", "Enclosure", "Predicted comfort"
        ))
        fig.add_trace(go.Heatmap(z=openness, x=g3["X"][0], y=g3["Y"][:, 0],
                                 colorscale="Viridis", showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=enclosure, x=g3["X"][0], y=g3["Y"][:, 0],
                                 colorscale="Magma", showscale=False), 1, 2)
        fig.add_trace(go.Heatmap(z=comfort, x=g3["X"][0], y=g3["Y"][:, 0],
                                 colorscale="RdYlGn", showscale=True), 1, 3)
        fig.update_layout(height=420, margin=dict(l=10, r=10, b=10, t=40))
        for i in range(1, 4):
            fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Comfort heuristic** = 0.55·openness + 0.25·(1 − |openness − 0.6|) "
            "− 0.3·max(0, 2D−3D). The middle term rewards *moderate* openness "
            "(neither claustrophobic nor agoraphobic), and the penalty "
            "captures the corner-over-estimation effect."
        )


# ---------------------------------------------------------------------------
# Tab 5 – Theory & references
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Why this matters")
    st.markdown(
        """
Planar isovists and space-syntax measures have been the workhorse of spatial
analysis for decades, but they treat every floor point as if the ceiling and
vertical obstructions did not exist. In real buildings, the **3D isovist
volume** diverges from the **2D isovist area × ceiling height** — most
noticeably in:

* **Corner regions** — where upper walls and soffits occlude rays the 2D
  method counts as "open".
* **Spaces with dropped ceilings, mezzanines, or partial-height furniture.**
* **Transition zones** between tall and low volumes (atria ↔ corridors).

### What this app demonstrates
1. A vectorised ray-casting engine that computes both planar and volumetric
   isovists on a common grid.
2. A difference map that localises *where* the 2D metric is unreliable.
3. A neural surrogate that learns the isovist field from a modest set of
   samples — the seed of a real-time, ML-powered design tool.
4. A perception proxy combining geometric descriptors into a comfort score,
   ready to be regressed against VR-collected human ratings.

### Extending the app
* Swap the 2-feature input `(x, y)` for a voxelised room tensor and train a
  small 3D-CNN — now the surrogate generalises across *geometries*, not just
  positions.
* Replace the heuristic comfort score with one trained on a small VR study
  (50–100 participants × 20 scenes is enough for a pilot).
* Couple with a generative layout model (diffusion over floor plans) and use
  the surrogate as the reward to optimise visibility-aware designs.
* Add semantic labels to obstacles (window, greenery, person) and compute
  *view-content* ratios, not just geometric openness.
        """
    )
    st.caption("Built as a demonstration. All computations run locally — no data is sent to servers.")
