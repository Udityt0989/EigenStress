"""
EigenStress - upload a 2D part, click edges to add supports and forces, run.
Start with:   streamlit run app.py
"""
import io
import hashlib
import json
from pathlib import Path

import joblib
import logging
import matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fe2d import interior_angles, signed_area
from geometry_io import read_dxf, write_sample_dxfs
from corner_tools import corner_features
from sing_tool import solve_two_meshes, analyse, max_regular_stress, predict, scan_holes
from topopt import optimise, extract_shape, write_dxf

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"
# Neon Brutalist x Terminal-Core palette. ACCENT is used in exactly three places:
# the Run button, the selected edge, and sharp-corner markers in the results.
BG, SURFACE, BORDER, FG, MUTED, ACCENT = "#0e0e0e", "#171717", "rgba(255,255,255,0.07)", "#f2f2f2", "#5a5a5a", "#e8ff5a"
INK, INK2, GRID = FG, "#8c8c8c", "rgba(255,255,255,0.07)"
C_EDGE, C_SUPPORT, C_LOAD, C_PICK, C_SING = MUTED, FG, FG, ACCENT, ACCENT
MONO = "Courier New, Courier, monospace"
SMOOTH_TURN = 10.0   # joints that turn less than this belong to the same "edge" (e.g. a drawn arc)

st.set_page_config(page_title="EigenStress", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;500&display=swap');
:root { --bg:#0e0e0e; --surface:#171717; --border:rgba(255,255,255,0.07); --fg:#f2f2f2; --muted:#5a5a5a; --accent:#e8ff5a; }
html, body, .stApp, .stMarkdown, p, li, input, textarea, select, button, [data-testid="stSidebar"] {
  font-family: 'Syne', 'Helvetica Neue', sans-serif !important; }
.stApp { background: var(--bg); }
[data-testid="stSidebar"] { background: var(--bg); border-right: 1px solid var(--border); }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
  font-family: 'Bebas Neue', 'Oswald', 'Arial Narrow', sans-serif !important;
  font-weight: 400 !important; text-transform: uppercase; letter-spacing: 0.01em; color: var(--fg); }
h1 { font-size: 96px !important; line-height: 0.9 !important; padding: 0 !important; }
h2 { font-size: 40px !important; }
h3 { font-size: 30px !important; }
[data-testid="stSidebar"] h2 { font-size: 26px !important; margin-top: 12px; }
[data-testid="stMetricValue"] { font-size: 56px !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"] p, .stCaption, label, .label, [data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stExpander"] summary p, [data-testid="stDataFrame"] {
  font-family: 'Courier New', Courier, monospace !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p, .label {
  font-size: 12px !important; letter-spacing: 0.08em; text-transform: uppercase; color: #8c8c8c !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 12px !important; color: #8c8c8c !important; }
/* square, border-only surfaces */
*, *::before, *::after { border-radius: 0 !important; }
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"], [data-testid="stNumberInputContainer"],
[data-testid="stFileUploaderDropzone"] {
  background: var(--bg) !important; border: 1px solid var(--border) !important; box-shadow: none !important; }
[data-testid="stForm"], [data-testid="stExpander"] details, [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; background: transparent !important; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stFileUploaderDropzone"] button {
  background: transparent !important; color: var(--fg) !important; border: 1px solid rgba(255,255,255,0.18) !important;
  font-family: 'Courier New', Courier, monospace !important; text-transform: uppercase; letter-spacing: 0.06em; font-size: 12px !important; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover { border-color: var(--fg) !important; }
.stButton button p, .stDownloadButton button p, .stFormSubmitButton button p { font-family: inherit !important; font-size: 12px !important; }
/* the one CTA: Run analysis */
.stButton button[kind="primary"] { background: var(--accent) !important; color: #0e0e0e !important; border-color: var(--accent) !important;
  font-weight: 700; padding: 12px 28px; }
.stButton button[kind="primary"]:disabled { background: transparent !important; color: var(--muted) !important; border-color: var(--border) !important; }
hr { border-color: var(--border) !important; }
input, [data-baseweb="select"] div, [data-testid="stDataFrame"] * { font-family: 'Courier New', Courier, monospace !important; }
.stMarkdown p.eyebrow, .eyebrow { font-family: 'Courier New', Courier, monospace !important; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #8c8c8c; }
.card { border: 1px solid var(--border); padding: 14px 16px; margin: 10px 0 14px; }
.card .hd { font-family: 'Bebas Neue', 'Oswald', sans-serif; font-size: 24px; letter-spacing: 0.02em; color: var(--fg); }
.card .sg { font-family: 'Courier New', Courier, monospace; font-size: 12px; color: #8c8c8c; margin: 6px 0 10px;
  padding-left: 10px; border-left: 2px solid #5a5a5a; }
.card .ml { font-family: 'Courier New', Courier, monospace; font-size: 12px; color: var(--fg); letter-spacing: 0.05em;
  padding-left: 10px; border-left: 2px solid var(--accent); }
.card .ml .big { display: block; font-family: 'Bebas Neue', 'Oswald', sans-serif; font-size: 44px; line-height: 1;
  letter-spacing: 0.01em; margin-top: 2px; }
.card .yl { font-family: 'Courier New', Courier, monospace; font-size: 12px; color: var(--fg); margin-top: 10px;
  padding-left: 10px; border-left: 2px solid var(--fg); }
.note { font-family: 'Courier New', Courier, monospace; font-size: 13px; color: var(--fg); border: 1px solid var(--border); padding: 10px 14px; margin: 8px 0; }
/* main Analyse / Optimise tabs: big section-style labels (only the top-level tab bar, not nested ones) */
.st-key-main_tabs > div > div > [data-testid="stTab"], .st-key-main_tabs > div > div > [data-testid="stTab"] * {
  font-family: 'Bebas Neue', 'Oswald', sans-serif !important; font-size: 30px !important; letter-spacing: 0.02em; }
.st-key-main_tabs > div > div > [data-testid="stTab"] { padding: 6px 26px 10px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model():
    return joblib.load(HERE / "singularity_model.joblib")


def edge_groups(outline):
    """Group polygon segments into user-facing edges: split only at real corners."""
    n = len(outline)
    beta = interior_angles(outline)
    seg_len = [np.linalg.norm(np.subtract(outline[(i + 1) % n], outline[i])) for i in range(n)]
    corner = []
    for i, b in enumerate(beta):           # vertex i joins segment i-1 and segment i
        sharp = abs(b - 180) >= SMOOTH_TURN
        a, c = seg_len[i - 1], seg_len[i]
        arc_to_line = abs(b - 180) > 0.1 and max(a, c) > 3 * min(a, c)   # tangent arc meets a straight edge
        corner.append(sharp or arc_to_line)
    if not any(corner):
        corner[0] = True
    start = corner.index(True)
    groups, cur = [], []
    for k in range(n):
        i = (start + k) % n
        if corner[i] and cur:
            groups.append(cur)
            cur = []
        cur.append(i)               # segment i runs from vertex i to vertex i+1
    groups.append(cur)
    out = []
    for g in groups:
        segs = [(np.asarray(outline[i], float), np.asarray(outline[(i + 1) % n], float)) for i in g]
        L = sum(np.linalg.norm(q - p) for p, q in segs)
        pts = np.array([segs[0][0]] + [q for _, q in segs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        k = np.searchsorted(cum, L / 2)
        k = min(max(k, 1), len(pts) - 1)
        t = (L / 2 - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-12)
        mid = pts[k - 1] + t * (pts[k] - pts[k - 1])
        out.append({"segs": segs, "length": L, "mid": mid, "curved": len(segs) > 1})
    return out


def _arc_xy(c, r, a0, a1, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return c[0] + r * np.cos(t), c[1] + r * np.sin(t)


def geometry_figure(geom, edges, supports, loads, selection, hole_bc=None, height=520):
    hole_bc = hole_bc or {}
    selection = set(selection or ())
    n_e = len(edges)
    fig = go.Figure()
    for idx, e in enumerate(edges):
        pts = np.array([e["segs"][0][0]] + [q for _, q in e["segs"]])
        if idx in supports:
            color, width, name = C_SUPPORT, 7, "support"
        elif idx in loads and idx not in selection:
            color, width, name = C_LOAD, 3, "load"
        elif idx in selection:
            color, width, name = C_PICK, 6, "selected"
        else:
            color, width, name = C_EDGE, 2.5, "edge"
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines",
                                 line=dict(color=color, width=width, dash="dot" if name == "load" else "solid"),
                                 hoverinfo="skip", showlegend=False))
    size = np.ptp(np.array(geom["outline"]), axis=0).max()
    for j, (c, r) in enumerate(geom["circles"]):
        sel = (n_e + j) in selection
        x, y = _arc_xy(c, r, 0, 360, 90)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C_PICK if sel else C_EDGE, width=4 if sel else 2),
                                 hoverinfo="skip", showlegend=False))
        bc = hole_bc.get(j)
        if bc is None:
            continue
        if bc["type"] == "fixed":
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C_SUPPORT, width=7),
                                     hoverinfo="skip", showlegend=False))
        elif bc["type"] == "pin":
            ax_, ay_ = _arc_xy(c, r, bc["angle"] - 90, bc["angle"] + 90)
            fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color=C_SUPPORT, width=7),
                                     hoverinfo="skip", showlegend=False))
        elif bc["type"] == "bearing":
            fx, fy = bc["F"]
            mag = np.hypot(fx, fy)
            if mag > 0:
                th = np.degrees(np.arctan2(fy, fx))
                ax_, ay_ = _arc_xy(c, r, th - 90, th + 90)
                fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", line=dict(color=C_LOAD, width=3, dash="dot"),
                                         hoverinfo="skip", showlegend=False))
                d = np.array([fx, fy]) / mag * max(size * 0.10, 2.5 * r)
                fig.add_annotation(x=c[0] + d[0], y=c[1] + d[1], ax=c[0], ay=c[1], xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4,
                                   arrowwidth=2.5, arrowcolor=C_LOAD, text="")
                fig.add_annotation(x=c[0] + 1.25 * d[0], y=c[1] + 1.25 * d[1], showarrow=False,
                                   text=f"{mag:.0f} N", font=dict(color=FG, size=11, family=MONO))
    for h in geom["poly_holes"]:
        p = np.array(h + [h[0]])
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=C_EDGE, width=2),
                                 hoverinfo="skip", showlegend=False))
    for idx, (fx, fy) in loads.items():
        m = edges[idx]["mid"]
        mag = np.hypot(fx, fy)
        if mag > 0:
            d = np.array([fx, fy]) / mag * size * 0.12
            fig.add_annotation(x=m[0], y=m[1], ax=m[0] - d[0], ay=m[1] - d[1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4,
                               arrowwidth=2.5, arrowcolor=C_LOAD,
                               text=f"{mag:.0f} N", font=dict(color=FG, size=12, family=MONO))
    mids = [e["mid"] for e in edges] + [np.asarray(c, float) for c, _ in geom["circles"]]
    mids = np.array(mids)
    labels = [f"E{i}" for i in range(n_e)] + [f"H{j}" for j in range(len(geom["circles"]))]
    ids = list(range(len(labels)))
    fig.add_trace(go.Scatter(
        x=mids[:, 0], y=mids[:, 1], mode="markers+text", text=labels,
        textfont=dict(color=FG, size=12, family=MONO),
        textposition=["top center"] * n_e + ["bottom right"] * (len(ids) - n_e),
        marker=dict(size=14, symbol=["square"] * n_e + ["circle"] * (len(ids) - n_e),
                    color=[C_PICK if i in selection else BG for i in ids],
                    line=dict(color=[C_PICK if i in selection else "#8c8c8c" for i in ids], width=1.5)),
        customdata=ids,
        hovertext=labels, hovertemplate="%{hovertext}: click to add / remove from selection<extra></extra>",
        showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                      font=dict(family=MONO, color="#8c8c8c", size=11),
                      hoverlabel=dict(bgcolor=SURFACE, bordercolor=FG, font=dict(family=MONO, color=FG)),
                      clickmode="event+select", dragmode="pan")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, title="X (MM)")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, scaleanchor="x", scaleratio=1,
                     title="Y (MM)")
    return fig


def edge_loads_from(forces, edges):
    """Total force (N) on each edge from all force groups. 'split' shares a total by edge length."""
    out = {}
    for f in forces:
        tg = [t for t in f["targets"] if t < len(edges)]
        if not tg:
            continue
        Ltot = sum(edges[t]["length"] for t in tg)
        for t in tg:
            k = edges[t]["length"] / Ltot if f["mode"] == "split" else 1.0
            fx, fy = out.get(t, (0.0, 0.0))
            out[t] = (fx + k * f["F"][0], fy + k * f["F"][1])
    return out


def hole_forces_from(forces, n_e):
    """List of bearing forces (N) per hole from all force groups. 'split' shares a total equally (bolt group)."""
    out = {}
    for f in forces:
        tg = [t - n_e for t in f["targets"] if t >= n_e]
        if not tg:
            continue
        k = 1.0 / len(tg) if f["mode"] == "split" else 1.0
        for j in tg:
            out.setdefault(j, []).append((k * f["F"][0], k * f["F"][1], f["name"]))
    return out


def hole_bc_full(hole_fix, hole_forces):
    """Fixtures + bearing resultants (a bolt carries the vector sum of all forces put through it)."""
    hb = {j: dict(b) for j, b in hole_fix.items()}
    for j, fl in hole_forces.items():
        if j in hb:                                   # fixed / pinned hole: bearing ignored
            continue
        R_ = (sum(f[0] for f in fl), sum(f[1] for f in fl))
        hb[j] = {"type": "bearing", "F": R_, "parts": fl}
    return hb


def build_part(geom, edges, supports, loads, thickness, E, nu, rho, hole_bc=None):
    fixed, trac = [], []
    for idx in supports:
        fixed += [(tuple(p), tuple(q)) for p, q in edges[idx]["segs"]]
    for idx, (fx, fy) in loads.items():
        L = edges[idx]["length"]
        tx, ty = fx / (L * thickness), fy / (L * thickness)          # N / mm^2 = MPa
        trac += [(tuple(p), tuple(q), (tx, ty)) for p, q in edges[idx]["segs"]]
    hb = {}
    for j, bc in (hole_bc or {}).items():
        bc = dict(bc)
        if bc["type"] == "bearing":
            bc["F"] = (bc["F"][0] / thickness, bc["F"][1] / thickness)  # solver works per unit thickness
        hb[j] = bc
    return {"poly": [np.asarray(p, float) for p in geom["outline"]], "circles": geom["circles"],
            "poly_holes": geom["poly_holes"], "fixed": fixed, "traction": trac, "hole_bc": hb,
            "E": E, "nu": nu, "corner_radius": rho}


def map_edges_to(edges, segs, tol=None):
    """Edges of a (new) outline that lie on any of the given old segments [(p, q), ...]."""
    out = []
    for idx, e in enumerate(edges):
        pts = np.array([e["segs"][0][0]] + [q for _, q in e["segs"]])
        L = e["length"]
        tl = tol if tol is not None else max(0.02 * L, 1e-3)
        for p, q in segs:
            p, q = np.asarray(p, float), np.asarray(q, float)
            d = q - p
            Ld = np.linalg.norm(d)
            tt = np.clip(((pts - p) @ d) / (Ld * Ld), 0, 1)
            dist = np.linalg.norm(pts - (p + tt[:, None] * d), axis=1)
            if dist.max() < tl:
                out.append(idx)
                break
    return out


def support_problem(supports, hole_bc):
    """Return a message if the supports cannot hold the part still (rigid-body mechanism), else None."""
    full = len(supports) + sum(1 for b in hole_bc.values() if b["type"] == "fixed")
    pins = sum(1 for b in hole_bc.values() if b["type"] == "pin")
    if full == 0 and pins == 0:
        return "Add a support: fix an edge, fix a hole, or pin a hole."
    if full == 0 and pins == 1:
        return ("A single pin lets the part swing around it. Add a second pin, a fixed hole, or a fixed edge.")
    return None


def result_figure(part, sols, flags, regular, hole_summary=(), mat=None):
    from matplotlib.colors import LinearSegmentedColormap
    f = sols["fine"]
    m, vm = f["m"], f["vm"]
    cap = regular["fine"][0] * 1.05 if regular["fine"][0] > 0 else np.percentile(vm, 98)
    mono = ["Courier New", "Courier", "DejaVu Sans Mono", "monospace"]
    cmap = LinearSegmentedColormap.from_list("mono", ["#1c1c1c", "#5a5a5a", "#f2f2f2"])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(BG)
    tc = ax.tripcolor(m.p[0], m.p[1], m.t.T, np.minimum(vm, cap), shading="gouraud", cmap=cmap, vmin=0, vmax=cap)
    cb = fig.colorbar(tc, ax=ax, shrink=0.8)
    cb.outline.set_edgecolor("#2a2a2a")
    cb.ax.tick_params(colors="#8c8c8c", labelsize=8)
    for t in cb.ax.get_yticklabels():
        t.set_fontfamily(mono)
    cb.set_label("VON MISES (MPA) · BRIGHTEST = AT/ABOVE MAX CONVERGED", color="#8c8c8c", fontsize=8, family=mono)
    ax.set_aspect("equal")
    ax.axis("off")
    box = dict(boxstyle="square,pad=0.35", fc=BG, ec="#2a2a2a", lw=1)
    sing = flags[flags.singular == True]
    sh = sing[sing.type == "sharp re-entrant corner"]
    pr_col = sh.get("predicted_real_stress", pd.Series(0.0, index=sh.index)).fillna(0)
    top = set(pr_col.sort_values(ascending=False).index[:5])       # label only the 5 most stressed corners
    many = len(sing) > 8
    for idx, r in sing.iterrows():
        geo = r.type == "sharp re-entrant corner"
        ax.plot(r.x, r.y, "o", ms=10 if many else 14, mfc="none", mec=C_SING if geo else "#8c8c8c",
                mew=1.6 if many else 2)
        if many and idx not in top:
            continue
        lab = r.label
        if geo:
            lab = f"{r.label} SINGULARITY · FE {r.fe_stress_fine:.0f} (NOT REAL)"
            if np.isfinite(r.get("predicted_real_stress", np.nan)):
                lab += f"\nML REAL STRESS → {r.predicted_real_stress:.0f} MPA"
                if mat and mat.get("on") and r.predicted_real_stress > mat["sy"]:
                    sg, _, ep = neuber(r.predicted_real_stress, mat["E"], mat["sy"], mat["n"])
                    lab += f"\nAFTER YIELD → {sg:.0f} MPA · εp {100 * ep:.2f}%"
        ax.annotate(lab, (r.x, r.y), xytext=(12, -22) if geo else (10, 10), textcoords="offset points",
                    fontsize=9, color=FG, family=mono, bbox=box, zorder=6 if geo else 4)
    for h in hole_summary:
        ax.plot(h["x"], h["y"], "D", ms=7, mfc="none", mec=FG, mew=1.4)
        ax.annotate(f"{h['hole']} PEAK {h['peak_stress']:.0f}", (h["x"], h["y"]), xytext=(-10, -16),
                    textcoords="offset points", fontsize=8, color=FG, family=mono, bbox=box, ha="right")
    v, p = regular["fine"]
    ax.plot(*p, "s", ms=9, mfc="none", mec=FG, mew=1.6)
    right_half = p[0] > (m.p[0].min() + m.p[0].max()) / 2
    ax.annotate(f"MAX ELSEWHERE {v:.0f} MPA", p, xytext=(-12, 10) if right_half else (12, 10),
                textcoords="offset points", fontsize=9, color=FG, family=mono, bbox=box, zorder=5,
                ha="right" if right_half else "left")
    ax.plot([], [], "o", ms=10, mfc="none", mec=C_SING, mew=2, label="SHARP CORNER · REAL STRESS PREDICTED")
    ax.plot([], [], "o", ms=10, mfc="none", mec="#8c8c8c", mew=2, label="SUPPORT / LOAD ARTEFACT")
    ax.plot([], [], "s", ms=9, mfc="none", mec=FG, mew=1.6, label="MAX CONVERGED STRESS ELSEWHERE")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
                    prop={"family": mono, "size": 8})
    for t in leg.get_texts():
        t.set_color("#8c8c8c")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ plasticity (Neuber)
def neuber(sig_e, E, sy, n):
    """
    Local stress/strain after yielding at a notch, from the linear-elastic peak sig_e (Neuber's rule):
        sigma * eps = sig_e^2 / E,  with a Ramberg-Osgood curve  eps = sigma/E + 0.002 (sigma/sy)^(1/n).
    Returns (sigma_actual, total_strain, plastic_strain). Below yield the elastic answer is returned.
    """
    from scipy.optimize import brentq
    if not np.isfinite(sig_e) or sig_e <= sy:
        return sig_e, sig_e / E, 0.0
    target = sig_e ** 2 / E
    f = lambda sg: sg * (sg / E + 0.002 * (sg / sy) ** (1 / n)) - target
    sg = brentq(f, 1e-6, sig_e)
    ep = 0.002 * (sg / sy) ** (1 / n)
    return sg, sg / E + ep, ep


# ------------------------------------------------------------------ contour fields + probe
FIELDS = {
    "vm": ("von Mises stress", "MPa"), "s1": ("Max principal stress σ1", "MPa"),
    "s2": ("Min principal stress σ2", "MPa"), "tmax": ("Max in-plane shear stress", "MPa"),
    "sx": ("Normal stress σx", "MPa"), "sy": ("Normal stress σy", "MPa"), "txy": ("Shear stress τxy", "MPa"),
    "e1": ("Max principal strain ε1", "µε"), "ex": ("Normal strain εx", "µε"),
    "ey": ("Normal strain εy", "µε"), "gxy": ("Shear strain γxy", "µε"),
}


def field_values(s, E, nu):
    """All contour fields at the mesh nodes (plane stress, linear elastic). Strains in microstrain."""
    sx, sy, txy = s["sx"], s["sy"], s["txy"]
    c, r = (sx + sy) / 2, np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    ex, ey, gxy = (sx - nu * sy) / E, (sy - nu * sx) / E, 2 * (1 + nu) * txy / E
    e1 = (ex + ey) / 2 + np.sqrt(((ex - ey) / 2) ** 2 + (gxy / 2) ** 2)
    return {"vm": s["vm"], "s1": c + r, "s2": c - r, "tmax": r, "sx": sx, "sy": sy, "txy": txy,
            "e1": 1e6 * e1, "ex": 1e6 * ex, "ey": 1e6 * ey, "gxy": 1e6 * gxy}


def singular_mask(xy, flags, frac=0.3):
    """True for nodes within frac*l of any flagged singular point."""
    near = np.zeros(len(xy), bool)
    for _, r in flags[flags.singular].iterrows():
        near |= np.linalg.norm(xy - np.array([r.x, r.y]), axis=1) < frac * r.ell
    return near


def contour_grid(px, py, tri, vals, n=220):
    """Interpolate a nodal field onto a regular grid (NaN outside the part, incl. holes)."""
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    T = Triangulation(np.asarray(px), np.asarray(py), np.asarray(tri))
    x0, x1, y0, y1 = float(np.min(px)), float(np.max(px)), float(np.min(py)), float(np.max(py))
    L = max(x1 - x0, y1 - y0)
    nx = max(20, int(n * (x1 - x0) / L))
    ny = max(20, int(n * (y1 - y0) / L))
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(gx, gy)
    Z = LinearTriInterpolator(T, np.asarray(vals))(X, Y)
    return gx, gy, np.ma.filled(Z, np.nan)


def probe_values(m, fields, pts):
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    T = Triangulation(m.p[0], m.p[1], m.t.T)
    out = []
    for (x, y) in pts:
        row = {"x": x, "y": y}
        for k, v in fields.items():
            val = LinearTriInterpolator(T, v)(x, y)
            row[k] = float(val) if not np.ma.is_masked(val) else np.nan
        out.append(row)
    return out


CMAPS = {"Mono": [[0, "#1c1c1c"], [0.5, "#5a5a5a"], [1, "#f2f2f2"]],
         "ANSYS-style": [[0, "#0000ff"], [0.25, "#00b4ff"], [0.5, "#00ff00"], [0.75, "#ffff00"], [1, "#ff0000"]]}


def contour_figure(gx, gy, Z, lo, hi, levels, cmap, unit, probes, extremes, height=560, marks=()):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z, zmin=lo, zmax=hi, colorscale=CMAPS[cmap], connectgaps=False,
        contours=dict(start=lo, end=hi, size=(hi - lo) / max(levels, 1), coloring="fill", showlines=False),
        colorbar=dict(title=dict(text=unit, font=dict(family=MONO, color="#8c8c8c", size=11)),
                      tickfont=dict(family=MONO, color="#8c8c8c", size=10), outlinewidth=0, thickness=14),
        hoverinfo="skip"))
    # invisible click / hover layer: one point per grid cell inside the part
    X, Y = np.meshgrid(gx, gy)
    ok = np.isfinite(Z)
    fig.add_trace(go.Scattergl(
        x=X[ok], y=Y[ok], mode="markers", marker=dict(size=7, color="rgba(0,0,0,0)"),
        customdata=Z[ok], hovertemplate=f"x %{{x:.2f}} · y %{{y:.2f}}<br>%{{customdata:.4g}} {unit}<extra></extra>",
        showlegend=False))
    for kind, (x, y, v) in extremes.items():
        side = "right" if x < (gx[0] + gx[-1]) / 2 else "left"
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"{kind} {v:.4g}"],
                                 textposition=("top " if kind == "MAX" else "bottom ") + side, textfont=dict(family=MONO, color=FG, size=11),
                                 marker=dict(symbol="triangle-up" if kind == "MAX" else "triangle-down", size=13,
                                             color=BG, line=dict(color=FG, width=2)),
                                 hoverinfo="skip", showlegend=False))
    for (x, y, txt, real) in marks:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[txt], textposition="bottom right",
                                 textfont=dict(family=MONO, color=FG, size=10),
                                 marker=dict(symbol="circle-open", size=16, color=C_SING if real else "#8c8c8c",
                                             line=dict(width=2.2)),
                                 hoverinfo="skip", showlegend=False))
    for i, pr in enumerate(probes):
        fig.add_trace(go.Scatter(x=[pr["x"]], y=[pr["y"]], mode="markers+text", text=[f"P{i + 1}"],
                                 textposition="bottom right", textfont=dict(family=MONO, color=FG, size=11),
                                 marker=dict(symbol="x-thin", size=14, line=dict(color=FG, width=2.5)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                      font=dict(family=MONO, color="#8c8c8c", size=11), dragmode="pan", clickmode="event+select",
                      hoverlabel=dict(bgcolor=SURFACE, bordercolor=FG, font=dict(family=MONO, color=FG)))
    pad = 0.06 * max(gx[-1] - gx[0], gy[-1] - gy[0])
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, title="X (MM)", range=[gx[0] - pad, gx[-1] + pad])
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=BORDER, scaleanchor="x", scaleratio=1, title="Y (MM)",
                     range=[gy[0] - pad, gy[-1] + pad])
    return fig


def mesh_stats(m):
    e = np.concatenate([m.t[[0, 1]], m.t[[1, 2]], m.t[[2, 0]]], axis=1)
    L = np.linalg.norm(m.p[:, e[0]] - m.p[:, e[1]], axis=0)
    return {"nodes": m.p.shape[1], "elements": m.t.shape[1], "dof": 2 * (m.p.shape[1] + m.facets.shape[1]),
            "h_min": float(L.min()), "h_max": float(L.max())}


def mesh_figure(m, flags, hole_summary, center=None, half=None, title=""):
    mono = ["Courier New", "Courier", "DejaVu Sans Mono", "monospace"]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(BG)
    lw = 0.35 if center is None else 0.7
    ax.triplot(m.p[0], m.p[1], m.t.T, color="#8c8c8c", lw=lw)
    for _, r in flags[flags.singular].iterrows():
        geo = r.type == "sharp re-entrant corner"
        ax.plot(r.x, r.y, "o", ms=12, mfc="none", mec=C_SING if geo else FG, mew=1.8, zorder=5)
        if center is None or (abs(r.x - center[0]) < half and abs(r.y - center[1]) < half):
            ax.annotate(r.label, (r.x, r.y), xytext=(9, 9), textcoords="offset points", fontsize=8, color=FG,
                        family=mono, bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec="#2a2a2a"), zorder=6)
    ax.set_aspect("equal")
    if center is not None:
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    ax.axis("off")
    if title:
        ax.set_title(title, color="#8c8c8c", fontsize=9, family=mono, loc="left")
    fig.tight_layout()
    return fig


def _corner_ring(part, r):
    """Ring and vertex index for a flagged corner row (outline Vi or cutout Ck.i)."""
    from sing_tool import part_holes
    lab = str(r.label)
    if lab.startswith("C"):
        k, i = (int(v) for v in lab[1:].split("."))
        return part_holes(part)[1][k], i
    return part["poly"], int(r.vertex)


def deformation_tab(part, sols, flags, thickness):
    from deformation import DispSampler, corner_opening, deformation_summary
    from views import deformed_figure, auto_exaggeration, outline_segments
    summ = deformation_summary(sols)
    if summ is None:
        st.info("The uploaded ANSYS results carry stresses only. Export UX / UY as well to see deformation; "
                "the built-in solver always has them.")
        return
    E, nu = part.get("E", 210000.0), part.get("nu", 0.3)
    c, f = summ["coarse"], summ["fine"]
    sharp = flags[flags.singular & (flags.type == "sharp re-entrant corner")]
    rise = sharp.refine_ratio.max() if len(sharp) else np.nan
    d1, d2, d3 = st.columns(3)
    d1.metric("Max |u| (mm)", f"{f['umax']:.3g}", f"{100 * (f['umax'] / c['umax'] - 1):+.2f} % on refining",
              delta_color="off", help="Fine mesh; the change from the coarse mesh shows it has converged.")
    d2.metric("Energy (N·mm)", f"{f['energy'] * thickness:.3g}",
              f"{100 * (f['energy'] / c['energy'] - 1):+.2f} % on refining", delta_color="off",
              help="U = ½ uᵀKu. Finite even at a sharp corner: the energy density ~ r^(2λ−2) is integrable.")
    d3.metric("Corner stress", f"×{rise:.2f}" if np.isfinite(rise) else "—", "on refining", delta_color="off",
              help="Same refinement, highest rise of a sharp-corner stress. Stress diverges; deformation does not.")
    k1, k2, k3 = st.columns([2, 2, 1])
    m_ = sols["fine"]["m"]
    s_ = sols["fine"]["s"]
    auto = auto_exaggeration(m_.p.T, np.hypot(s_["ux"], s_["uy"]))
    mult = k1.number_input("Exaggeration (× auto)", min_value=0.0, value=1.0, step=0.5, key="def_mult",
                           help=f"Auto = {auto:.3g}×, so the largest displacement is 8 % of the part size.")
    show_mesh = k3.checkbox("Mesh", value=False, key="def_mesh")
    from sing_tool import part_holes
    rings = [part["poly"]] + list(part_holes(part)[1])
    rings += [np.c_[cc[0] + rr * np.cos(np.linspace(0, 2 * np.pi, 60)), cc[1] + rr * np.sin(np.linspace(0, 2 * np.pi, 60))]
              for cc, rr in part.get("circles", [])]
    marks = [(r.x, r.y, r.label) for _, r in sharp.iterrows()]
    st.pyplot(deformed_figure(m_.p.T, m_.t.T, s_["ux"], s_["uy"], auto * mult, outline_segments(rings),
                              show_mesh, marks, title=f"DEFORMED SHAPE · ×{auto * mult:.3g}"), width="stretch")
    rows = []
    ds = {t: DispSampler(sols[t]["m"], sols[t]["s"]) for t in ("coarse", "fine")}
    for _, r in sharp.iterrows():
        try:
            ring, i = _corner_ring(part, r)
            oc = corner_opening(ds["coarse"], ring, i, r.ell, E, nu)
            of = corner_opening(ds["fine"], ring, i, r.ell, E, nu)
        except Exception:
            oc = of = None
        if not of:
            continue
        rows.append({"corner": r.label, "angle": f"{r.angle_deg:.0f}°",
                     "K1 from displacement": f"{of['K1_u']:.4g}",
                     "change on refining": f"{100 * (of['K1_u'] / oc['K1_u'] - 1):+.1f} %" if oc else "—",
                     "stress change on refining": f"{100 * (r.refine_ratio - 1):+.0f} %",
                     "λ from opening": f"{of['lam_u']:.3f}", "Williams λ": f"{of['lam']:.3f}",
                     "flank opening at 0.1 l (µm)": f"{1000 * of['opening_at_01l']:.3g}"})
    if rows:
        st.markdown("**Corner intensity from deformation**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Near a sharp corner the stress goes as r^(λ−1) (infinite) but the displacement as r^λ (finite). "
               "The flank opening δ(r) between the two corner edges is fitted to a·r^λ + b·r + c·r² over "
               "0.1–0.4 l; Williams' displacement field turns a into K1, in MPa·mm^(1−λ). Rigid motion and the "
               "mode II part cancel in δ, so this is mode I only.")


def photo_tab(part, sols, flags, thickness):
    from deformation import grid_fields
    from views import photo_controls, photo_panel
    opts = photo_controls("ph", thickness)
    m_ = sols["fine"]["m"]
    g, ext = grid_fields(m_, sols["fine"]["s"], n_px=760)
    ny, nx = g["sx"].shape
    X, Y = np.meshgrid(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny))
    ok = ~singular_mask(np.c_[X.ravel(), Y.ravel()], flags, 0.3).reshape(X.shape)
    sharp = flags[flags.singular]
    top = sharp.assign(_k=sharp.get("predicted_real_stress", pd.Series(0.0, index=sharp.index)).fillna(0)) \
        .sort_values("_k", ascending=False).head(5)
    marks = [(r.x, r.y, r.label) for _, r in top.iterrows()]
    F = sum(np.hypot(*fo["F"]) for fo in st.session_state.get("forces", [])) if st.session_state.get("forces") else 0.0
    lf_holder = {}
    rows = []
    lf = photo_panel(g, ext, ok, opts, F, marks, (), title="SYNTHETIC PHOTOELASTICITY · " +
                     ("CIRCULAR" if opts["pol"] == "circular" else "PLANE") + " POLARISCOPE")
    for _, r in sharp[sharp.type == "sharp re-entrant corner"].iterrows():
        pr = r.get("predicted_real_stress", np.nan)
        rows.append({"corner": r.label,
                     "fringe order at corner node, coarse → fine": f"{thickness * r.fe_stress_coarse * lf / opts['f']:.1f} → "
                                                                  f"{thickness * r.fe_stress_fine * lf / opts['f']:.1f}",
                     f"with the real radius": f"{thickness * pr * lf / opts['f']:.1f}" if np.isfinite(pr) else "—"})
    if rows:
        st.markdown("**Fringe order at the sharp corners**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("At a free edge σ2 = 0, so the fringe order is t·σ/f_σ. The sharp-corner count grows with "
                   "every refinement (the model is wrong there); with the real radius it is finite and is "
                   "what a photoelastic test of the real part would show.")


def report_text(name, settings, supports, loads, edges, flags, regular, hole_bc=None, hole_summary=(), forces=()):
    hole_bc = hole_bc or {}
    hole_lines = []
    n_e = len(edges)
    lab = lambda t: f"E{t}" if t < n_e else f"H{t - n_e}"
    for f in forces:
        hole_lines.append(f"- Force {f['name']}: Fx = {f['F'][0]:.1f} N, Fy = {f['F'][1]:.1f} N on "
                          f"{', '.join(lab(t) for t in f['targets'])} "
                          f"({'total split across these' if f['mode'] == 'split' else 'applied to each'})")
    for j, b in sorted(hole_bc.items()):
        if b["type"] == "fixed":
            hole_lines.append(f"- H{j}: fixed hole (bolted / press-fit)")
        elif b["type"] == "pin":
            hole_lines.append(f"- H{j}: pin support, contact side centred at {b['angle']:g} deg (180 deg patch)")
        else:
            hole_lines.append(f"- H{j}: bolt bearing, resultant Fx = {b['F'][0]:.1f} N, Fy = {b['F'][1]:.1f} N")
    lines = [f"# EigenStress report: {name}", "",
             f"Thickness {settings['t']} mm, E = {settings['E']:.0f} MPa, nu = {settings['nu']}, "
             f"plane stress, linear elastic.", "",
             "## Boundary conditions",
             *[f"- E{i}: fixed support (length {edges[i]['length']:.1f} mm)" for i in supports],
             *hole_lines, "",
             "## Singular points", "",
             "| Point | Angle (deg) | Type | FE coarse (MPa) | FE fine (MPa) | Rise on refining | "
             "Corner radius (mm) | Predicted real stress (MPa) | Note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in flags[flags.singular].iterrows():
        pr = r.get("predicted_real_stress", np.nan)
        ang = "-" if not np.isfinite(r.angle_deg) else f"{r.angle_deg:.0f}"
        lines.append(f"| {r.label} ({r.x:.1f}, {r.y:.1f}) | {ang} | {r.type} | "
                     f"{r.fe_stress_coarse:.1f} | {r.fe_stress_fine:.1f} | x{r.refine_ratio:.2f} | "
                     f"{r.get('corner_radius', np.nan) if r.type.startswith('sharp') else '-'} | "
                     f"{'-' if not np.isfinite(pr) else f'{pr:.1f}'} | {r.get('note', '') if isinstance(r.get('note', ''), str) else ''} |")
    if hole_summary:
        lines += ["", "## Holes", "", "| Hole | Fixture | Dia (mm) | Peak stress (MPa) | At angle (deg) | "
                  "Change vs coarse mesh | Singular points |", "|---|---|---|---|---|---|---|"]
        for h in hole_summary:
            lines.append(f"| {h['hole']} | {h['fixture']} | {2 * h['radius_mm']:.1f} | {h['peak_stress']:.1f} | "
                         f"{h['at_angle_deg']:.0f} | {h['change_pct']:+.1f}% | {h['singular_points']} |")
    v_c, v_f = regular["coarse"][0], regular["fine"][0]
    lines += ["", f"Highest converged stress away from singular points: {v_f:.1f} MPa at "
                  f"({regular['fine'][1][0]:.1f}, {regular['fine'][1][1]:.1f}) "
                  f"(coarse mesh {v_c:.1f} MPa, change {100 * (v_f - v_c) / v_c:+.1f}%).", "",
              "Method: FE solved at two meshes; points whose stress rises >5% when the local mesh is halved are "
              "singular. Sharp corners: the singular field 0.1-0.3 l from the corner (mesh-independent) is fed to "
              "a model trained on ~74,000 sharp-vs-filleted FE pairs to predict the finite peak stress for the "
              "given corner radius. Typical accuracy within the trained range: ~2.6% mean, ~9.6% at the 95th "
              "percentile. Support/load artefacts are not physical stresses and are not predicted."]
    return "\n".join(lines)


# ------------------------------------------------------------------ state
ss = st.session_state
for k, v in {"supports": set(), "forces": [], "hole_fix": {}, "geom_key": None, "results": None,
             "pending_toggle": None, "force_count": 0}.items():
    ss.setdefault(k, v)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("00 · Mode")
    mode = st.radio("How to get the stresses", ["sur", "fe"], key="app_mode",
                    format_func={"sur": "Instant: parametric surrogate", "fe": "Any drawing: DXF + FE"}.get,
                    help="Instant: pick an L-bracket or stepped bar, type its dimensions, and a reduced-order "
                         "model trained offline on FE gives stress, deformation and fringes in milliseconds. "
                         "Any drawing: upload a DXF; FE runs on two meshes and the singular-corner model "
                         "predicts the real peak at every sharp corner.")
    up, sample, src, ans_f, ans_c, scale = None, "—", "builtin", None, None, 1.0
    if mode == "fe":
        st.header("01 · Part")
        if not SAMPLES.exists() or not any(SAMPLES.glob("*.dxf")):
            SAMPLES.mkdir(exist_ok=True)
            write_sample_dxfs(str(SAMPLES))
        up = st.file_uploader("Upload a 2D drawing (DXF)", type=["dxf"])
        sample_names = sorted(p.stem for p in SAMPLES.glob("*.dxf"))
        sample = st.selectbox("…or try a sample", ["—"] + sample_names, index=0 if up else 1)
        src = st.radio("Stresses from", ["builtin", "ansys"], horizontal=True,
                       format_func={"builtin": "Built-in solver", "ansys": "ANSYS results"}.get, key="src",
                       help="ANSYS results: upload the result file of the same part (.rst, or a CSV with X, Y, SX, SY, "
                            "SXY) and EigenStress analyses ANSYS's own stress field. Supports and loads then come "
                            "from your ANSYS model.")
        ans_f = ans_c = None
        if src == "ansys":
            _af = st.file_uploader("ANSYS result, fine mesh (.rst / .csv)", type=["rst", "csv", "txt"], key="ans_f",
                                   help="Workbench: <project>_files/dp0/SYS/MECH/file.rst. 2D or 3D plate models; "
                                        "the face matching the DXF is found automatically.")
            _ac = st.file_uploader("ANSYS result, coarse mesh (optional)", type=["rst", "csv", "txt"], key="ans_c",
                                   help="Same model with about twice the element size at the corners. With both, "
                                        "singular points are confirmed by mesh refinement; with one, every sharp "
                                        "re-entrant corner is treated as singular from its geometry.")
            ans_f = (_af.getvalue(), _af.name) if _af else None
            ans_c = (_ac.getvalue(), _ac.name) if _ac else None
        scale = st.number_input("Drawing units → mm (multiply by)", value=1.0, min_value=1e-6, format="%g",
                                help="1 if the DXF is in mm, 25.4 if in inches, 1000 if in metres")
    st.header("02 · Material")
    thickness = st.number_input("Thickness t (mm)", value=10.0, min_value=0.01)
    E = st.number_input("Young's modulus E (MPa)", value=210000.0, min_value=1.0, format="%g")
    nu = st.number_input("Poisson's ratio (nu)", value=0.30, min_value=0.0, max_value=0.49)
    rho = st.number_input("Real corner radius at sharp corners (mm)", value=0.5, min_value=0.001,
                          help="Machined inside corners are never perfectly sharp - typically 0.2-1 mm. "
                               "The real stress depends on this radius.")
    sy = st.number_input("Yield strength σy (MPa)", value=250.0, min_value=1.0, format="%g",
                         help="Mild steel ~250, S355 ~355, 6061-T6 aluminium ~275.")
    plast = st.checkbox("Correct for yielding (Neuber's rule)", value=True,
                        help="If the predicted elastic peak is above yield, the real material yields locally. "
                             "Neuber's rule + a Ramberg-Osgood curve estimate the actual stress and plastic strain.")
    n_ro = st.number_input("Hardening exponent n (Ramberg-Osgood)", value=0.15, min_value=0.02, max_value=0.5,
                          format="%g", help="~0.1-0.2 for steels and aluminium alloys. Only used above yield.") \
        if plast else 0.15
    st.header("Design checks")
    sf = st.number_input("Safety factor on yield", value=1.5, min_value=1.0, step=0.1, format="%g",
                         help="Allowable stress = σy / safety factor. Every sharp corner gets the smallest fillet "
                              "radius that keeps its real peak stress below this.")
    su = st.number_input("Ultimate strength σu (MPa)", value=400.0, min_value=1.0, format="%g",
                         help="For the fatigue check. Mild steel ~400, S355 ~510, 6061-T6 ~310.")
    se_in = st.number_input("Endurance limit Se (MPa), 0 = auto", value=0.0, min_value=0.0, format="%g",
                            help="Fatigue strength at 10⁶ cycles. 0 = 0.5 σu (polished steel). Lower it for rough "
                                 "surfaces, size or reliability (e.g. 0.35-0.4 σu for machined parts).")
    conf_r = st.checkbox("Required radius with 90 % confidence", value=True,
                         help="Size each fillet so that the TOP of its calibrated 90 % prediction interval stays "
                              "below the allowable stress (safer). Off: size on the central prediction.")
    cycle = st.radio("Load cycle", ["R=-1", "R=0"], horizontal=True,
                     format_func={"R=-1": "Reversed (±F)", "R=0": "On/off (0→F)"}.get,
                     help="How the applied forces repeat in service: fully reversed (+F to −F) or zero to max.")
    se = se_in if se_in > 0 else 0.5 * su

if mode == "sur":
    st.markdown('<p class="eyebrow">Real stress at sharp corners · deformation · photoelasticity</p>',
                unsafe_allow_html=True)
    st.title("EigenStress")
    st.markdown('<p style="font-size:18px;max-width:680px;margin-top:-4px">Stress at a sharp corner is infinite in '
                'theory and mesh-dependent in FE; deformation is not. EigenStress predicts the real stress, the '
                'deformation and the photoelastic fringe pattern of a part from a surrogate model.</p>',
                unsafe_allow_html=True)
    from surrogate_app import surrogate_page
    surrogate_page(thickness, E, sy, sf, su, se, cycle)
    st.stop()

# ------------------------------------------------------------------ geometry
try:
    if ss.get("opt_geom") is not None:
        geom = ss.opt_geom["geom"]
        name = ss.opt_geom["name"]
        gkey = ss.opt_geom["key"]
        with st.sidebar:
            st.markdown('<p class="eyebrow">→ showing the OPTIMISED shape</p>', unsafe_allow_html=True)
            if st.button("Back to the original drawing", width="stretch"):
                ss.opt_geom = None
                ss.geom_key = None
                st.rerun()
    elif up is not None:
        raw = up.getvalue()
        geom = read_dxf(io.BytesIO(raw), scale)
        name = up.name
        gkey = hashlib.md5(raw).hexdigest() + str(scale)
    elif sample != "—":
        geom = read_dxf(str(SAMPLES / f"{sample}.dxf"), scale)
        name = sample
        gkey = sample + str(scale)
    else:
        st.markdown('<p class="eyebrow">→ Upload a DXF or pick a sample in the sidebar</p>', unsafe_allow_html=True)
        st.stop()
except Exception as e:
    st.error(f"Could not read the drawing: {e}")
    st.stop()

if ss.geom_key != gkey:
    ss.supports, ss.forces, ss.hole_fix, ss.results, ss.geom_key = set(), [], {}, None, gkey
    ss.sel_ms, ss.pending_toggle, ss.force_count = [], None, 0
    ss.topo = None
edges = edge_groups(geom["outline"])
if ss.get("pending_bc") is not None:                       # supports / loads carried over to an optimised shape
    pb = ss.pending_bc
    ss.pending_bc = None
    n_e_new = len(edges)
    ss.hole_fix = {pb["hole_map"][j]: b for j, b in pb["hole_fix"].items() if j in pb["hole_map"]}
    ss.supports = set(map_edges_to(edges, pb["fixed_segs"]))
    ss.forces, ss.force_count = [], 0
    for fo in pb["forces"]:
        tg = [n_e_new + pb["hole_map"][t] for t in fo["holes"] if t in pb["hole_map"]]
        tg += map_edges_to(edges, fo["edge_segs"])
        if tg:
            ss.force_count += 1
            ss.forces.append({"name": f"F{ss.force_count}", "targets": tg, "F": fo["F"], "mode": fo["mode"]})

st.markdown('<p class="eyebrow">FEM singularity finder · real-stress predictor</p>', unsafe_allow_html=True)
st.title("EigenStress")
st.markdown('<p style="font-size:18px;max-width:640px;margin-top:-4px">Finds the points where FE stress never converges, '
            'and predicts the real peak stress at every sharp corner.</p>', unsafe_allow_html=True)

col_plot, col_ctrl = st.columns([3, 2], gap="large")
with col_ctrl:
    st.subheader("03 · Supports + forces")
    n_e, n_h = len(edges), len(geom["circles"])
    lab = lambda i: f"E{i}" if i < n_e else f"H{i - n_e}"
    opts = list(range(n_e + n_h))
    if ss.pending_toggle == "clear":
        ss.sel_ms, ss.pending_toggle = [], None
    if ss.pending_toggle is not None:               # a click on the drawing adds / removes that item
        cur = list(ss.get("sel_ms", []))
        t = ss.pending_toggle
        ss.sel_ms = [x for x in cur if x != t] if t in cur else cur + [t]
        ss.pending_toggle = None
    ss.sel_ms = [x for x in ss.get("sel_ms", []) if x in opts]

    def _fmt(i):
        if i < n_e:
            return f"E{i} · {edges[i]['length']:.0f} mm{' arc' if edges[i]['curved'] else ''}"
        return f"H{i - n_e} · hole Ø{2 * geom['circles'][i - n_e][1]:.1f}"
    st.write("Click edges (E…) and holes (H…) on the drawing to build a selection, then apply a support, "
             "fixture or force to all of it at once.")
    selection = st.multiselect("Selection", opts, format_func=_fmt, key="sel_ms", placeholder="Nothing selected")
    sel_e = [i for i in selection if i < n_e]
    sel_h = [i - n_e for i in selection if i >= n_e]

    if selection:
        c1, c2, c3 = st.columns(3)
        if c1.button("Fix selected", width="stretch",
                     help="Edges become clamped supports; holes become fixed (bolted / press-fit)."):
            ss.supports |= set(sel_e)
            for j in sel_h:
                ss.hole_fix[j] = {"type": "fixed"}
            ss.results = None
            ss.pending_toggle = "clear"                # done with this selection: deselect
            st.rerun()
        if c2.button("Clear selected", width="stretch",
                     help="Remove supports, hole fixtures and forces from the selected items."):
            ss.supports -= set(sel_e)
            for j in sel_h:
                ss.hole_fix.pop(j, None)
            for f in ss.forces:
                f["targets"] = [t for t in f["targets"] if t not in selection]
            ss.forces = [f for f in ss.forces if f["targets"]]
            ss.results = None
            ss.pending_toggle = "clear"                # done with this selection: deselect
            st.rerun()
        if c3.button("Deselect all", width="stretch"):
            ss.pending_toggle = "clear"
            st.rerun()

        with st.form("force_form", border=True):
            what = ("edges" if not sel_h else "holes (bolt bearing)" if not sel_e else "edges + holes")
            st.write(f"Force on the selection: {', '.join(lab(i) for i in selection)}  ·  {what}")
            f1, f2 = st.columns(2)
            fx = f1.number_input("Fx (N)", value=0.0, format="%g")
            fy = f2.number_input("Fy (N)", value=-1000.0, format="%g")
            mode = st.radio("How", ["split", "each"], horizontal=True,
                            format_func={"split": "Total, split across selection",
                                         "each": "Same force on each item"}.get)
            st.caption("Split: edges share it by length (uniform pressure), holes share it equally (bolt group). "
                       "Add several forces to the same holes to load a bolt in several directions.")
            if st.form_submit_button("Add force", width="stretch"):
                if fx == 0 and fy == 0:
                    st.warning("Force is zero.")
                else:
                    ss.force_count += 1
                    ss.forces.append({"name": f"F{ss.force_count}", "targets": list(selection),
                                      "F": (float(fx), float(fy)), "mode": mode})
                    ss.supports -= set(sel_e)
                    for j in sel_h:
                        ss.hole_fix.pop(j, None)
                    ss.results = None
                    ss.pending_toggle = "clear"                # done with this selection: deselect
                    st.rerun()

        if sel_h:
            with st.form("pin_form", border=True):
                st.write(f"Pin support on {', '.join(f'H{j}' for j in sel_h)} (frictionless pin, 180° contact)")
                ang = st.number_input("Contact side (deg)  0 = right · 90 = top · 180 = left · 270 = bottom",
                                      value=90.0, step=15.0, format="%g",
                                      help="The side of the hole that presses on the pin. A part hanging from a "
                                           "pin rests on the TOP of the hole (90).")
                if st.form_submit_button("Apply pin support", width="stretch"):
                    for j in sel_h:
                        ss.hole_fix[j] = {"type": "pin", "angle": float(ang)}
                    ss.results = None
                    ss.pending_toggle = "clear"                # done with this selection: deselect
                    st.rerun()

    # ---- summary of everything applied
    rows = [{"what": "support", "on": ", ".join(f"E{i}" for i in sorted(ss.supports)), "detail": "clamped edges"}] \
        if ss.supports else []
    for j, b in sorted(ss.hole_fix.items()):
        rows.append({"what": "fixed hole" if b["type"] == "fixed" else "pin", "on": f"H{j}",
                     "detail": "" if b["type"] == "fixed" else f"contact side {b['angle']:g}°"})
    for f in ss.forces:
        rows.append({"what": f["name"], "on": ", ".join(lab(t) for t in f["targets"]),
                     "detail": f"Fx {f['F'][0]:g} · Fy {f['F'][1]:g} N · {'split' if f['mode'] == 'split' else 'each'}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if ss.forces:
            with st.form("edit_force_form", border=True):
                st.write("Edit or delete an applied force")
                names = [f["name"] for f in ss.forces]
                d0, d1, d2 = st.columns([1, 1, 1])
                pick = d0.selectbox("Force", names)
                cur = next(f for f in ss.forces if f["name"] == pick)
                efx = d1.number_input("New Fx (N)", value=None, format="%g", placeholder=f"{cur['F'][0]:g}")
                efy = d2.number_input("New Fy (N)", value=None, format="%g", placeholder=f"{cur['F'][1]:g}")
                st.caption("Leave a box empty to keep its current value (shown in grey).")
                b1, b2 = st.columns(2)
                upd = b1.form_submit_button("Update force", width="stretch")
                dele = b2.form_submit_button("Delete force", width="stretch")
            if upd or dele:
                f_ = next(f for f in ss.forces if f["name"] == pick)
                if dele:
                    ss.forces = [f for f in ss.forces if f["name"] != pick]
                else:
                    nf = (float(efx) if efx is not None else f_["F"][0], float(efy) if efy is not None else f_["F"][1])
                    if nf == (0.0, 0.0):
                        st.warning("Force is zero; use Delete instead.")
                        st.stop()
                    f_["F"] = nf
                ss.results = None
                st.rerun()
        hf = hole_forces_from(ss.forces, n_e)
        clash = [j for j in hf if j in ss.hole_fix]
        if clash:
            st.markdown(f'<div class="note">NOTE → {", ".join(f"H{j}" for j in clash)}: hole is fixed / pinned, '
                        'so bearing forces on it are ignored</div>', unsafe_allow_html=True)
    else:
        st.caption("No supports or forces yet.")

loads = edge_loads_from(ss.forces, edges)
hole_bc = hole_bc_full(ss.hole_fix, hole_forces_from(ss.forces, len(edges)))

with col_plot:
    st.markdown(f'<p class="eyebrow">Part → {name}</p>', unsafe_allow_html=True)
    fig = geometry_figure(geom, edges, ss.supports, loads, ss.sel_ms, hole_bc)
    fig_key = "geom_" + hashlib.md5(str((sorted(ss.supports), ss.forces, sorted(ss.hole_fix.items()),
                                          ss.sel_ms)).encode()).hexdigest()[:10]
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                         key=fig_key,
                         config={"displaylogo": False, "scrollZoom": True})
    try:
        pts = ev.selection.points if ev else []
    except AttributeError:
        pts = (ev or {}).get("selection", {}).get("points", [])
    if pts:
        p = pts[0]
        cd = p.get("customdata")
        if isinstance(cd, list):
            cd = cd[0]
        if cd is None:
            mids = np.array([e["mid"] for e in edges] + [np.asarray(c, float) for c, _ in geom["circles"]])
            cd = int(np.argmin(np.hypot(mids[:, 0] - p["x"], mids[:, 1] - p["y"])))
        if ss.get("last_click") != (int(cd), fig_key):
            ss.last_click = (int(cd), fig_key)
            ss.pending_toggle = int(cd)
            st.rerun()
    holes = len(geom["circles"]) + len(geom["poly_holes"])
    st.caption(f"{len(edges)} edges · {holes} hole(s) · solid white = support / fixture · dotted = force · yellow = selected")

# ------------------------------------------------------------------ 04 analyse / 05 optimise (tabs)
st.divider()
TAB_AN, TAB_OPT = "04 · Analyse", "05 · Optimise shape"
if ss.get("goto_tab"):
    ss["main_tabs"] = ss.pop("goto_tab")
try:
    tab_an, tab_opt = st.tabs([TAB_AN, TAB_OPT], key="main_tabs", on_change="rerun")
except TypeError:                                            # older Streamlit: tabs without a key
    tab_an, tab_opt = st.tabs([TAB_AN, TAB_OPT])

with tab_an:
    # ------------------------------------------------------------------ run
    has_load = bool(loads) or any(b["type"] == "bearing" and any(b["F"]) for b in hole_bc.values())
    problem = support_problem(ss.supports, ss.hole_fix)
    ready = has_load and problem is None                     # (built-in solver; also gates the Optimise tab)
    if src == "ansys":
        st.markdown('<div class="note">ANSYS MODE → stresses come from your uploaded ANSYS result, so supports and '
                    'forces are already in it. Marking supports in 03 is optional (their ends are then flagged as '
                    'support artefacts instead of sharp corners).</div>', unsafe_allow_html=True)
        if ans_f is None:
            st.markdown('<p class="eyebrow">→ Upload the ANSYS result file in the sidebar (01 · Part)</p>',
                        unsafe_allow_html=True)
        run = st.button("Analyse ANSYS result", type="primary", disabled=ans_f is None)
    else:
        if problem:
            st.markdown(f'<p class="eyebrow">→ {problem}</p>', unsafe_allow_html=True)
        if not has_load:
            st.markdown('<p class="eyebrow">→ Add a force on an edge, or a bearing load on a hole</p>', unsafe_allow_html=True)
        run = st.button("Run analysis", type="primary", disabled=not ready)

    if run and src == "ansys":
        part = build_part(geom, edges, ss.supports, loads, thickness, E, nu, rho, hole_bc)
        with st.spinner("Reading the ANSYS result…"):
            try:
                from ansys_io import sols_from_results
                sols, info_ = sols_from_results(part, ans_f, ans_c)
            except Exception as e:
                st.error(f"Could not use the ANSYS result: {e}")
                st.stop()
        ss.results = {"part": part, "sols": sols,
                      "used": ["ANSYS " + i_ for i_ in info_] +
                              (["one result: sharp corners treated as singular from their geometry"]
                               if sols.get("geometric_only") else [])}
        ss.probes = []
        ss.run_id = ss.get("run_id", 0) + 1
    elif run:
        part = build_part(geom, edges, ss.supports, loads, thickness, E, nu, rho, hole_bc)
        with st.spinner("Meshing and solving at two mesh sizes…"):
            try:
                sols = solve_two_meshes(part)
            except Exception as e:
                st.error(f"Solve failed: {e}")
                st.stop()
        ss.results = {"part": part, "sols": sols,
                      "used": [f"{f['name']} on {', '.join(lab(t) for t in f['targets'])}: "
                               f"Fx {f['F'][0]:g} N, Fy {f['F'][1]:g} N ({f['mode']})" for f in ss.forces]}
        ss.probes = []
        ss.run_id = ss.get("run_id", 0) + 1

    if ss.results:
        part, sols = ss.results["part"], ss.results["sols"]
        part["corner_radius"] = rho
        if ss.results.get("used"):
            st.markdown('<div class="note">SOLVED WITH → ' + " · ".join(ss.results["used"]) + "</div>", unsafe_allow_html=True)
        model = load_model()
        flags = analyse(part, sols, model)
        hole_rows, hole_summary = scan_holes(part, sols)
        if hole_rows:
            flags = pd.concat([flags, pd.DataFrame(hole_rows)], ignore_index=True)
        regular = max_regular_stress(part, sols, flags)
        mat = {"E": E, "sy": sy, "n": n_ro, "on": plast}
        if "predicted_real_stress" in flags:
            nb = [neuber(v, E, sy, n_ro) if (plast and np.isfinite(v)) else (v, np.nan, 0.0)
                  for v in flags["predicted_real_stress"]]
            flags["actual_after_yield"] = [x[0] for x in nb]
            flags["plastic_strain_pct"] = [100 * x[2] for x in nb]
        # ---- design answers: required radius (sharp corners) + fatigue (every hot spot)
        from design_tools import required_radius, fatigue, life_text, interval
        allow = sy / sf
        req_r, req_s, lo_, hi_, wd_ = [], [], [], [], []
        for _, r_ in flags.iterrows():
            fe_ = r_.get("feats")
            kind_ = "cutout" if str(r_.get("label", "")).startswith("C") else "outline"
            pk_ = r_.get("predicted_real_stress", np.nan)
            if r_.get("type") == "sharp re-entrant corner" and np.isfinite(pk_ if pk_ is not None else np.nan):
                a_, b_, w_ = interval(pk_, r_.corner_radius / r_.ell, kind_)
            else:
                a_, b_, w_ = np.nan, np.nan, np.nan
            lo_.append(a_)
            hi_.append(b_)
            wd_.append(w_)
            if r_.get("type") == "sharp re-entrant corner" and isinstance(fe_, dict):
                rr_, st_, _ = required_radius(model, fe_, allow, kind=kind_ if conf_r else None)
            else:
                rr_, st_ = np.nan, ""
            req_r.append(rr_)
            req_s.append(st_)
        flags["pred_low_90"], flags["pred_high_90"], flags["pred_band_pct"] = lo_, hi_, wd_
        flags["required_radius"] = req_r
        flags["required_status"] = req_s
        sharp = flags[(flags.type == "sharp re-entrant corner")]
        artef = flags[(flags.type == "support / load artefact")]
        fat_rows = []
        for _, r_ in sharp.iterrows():
            pk_ = r_.get("predicted_real_stress", np.nan)
            if np.isfinite(pk_):
                fz = fatigue(pk_, r_.get("nominal_stress", pk_), r_.corner_radius, su, se, cycle)
                fat_rows.append({"where": f"{r_.label} · corner r {r_.corner_radius:g} mm", "peak": pk_, **fz})
        for h_ in hole_summary:
            fz = fatigue(h_["peak_stress"], h_["peak_stress"], h_["radius_mm"], su, se, cycle)
            fat_rows.append({"where": f"{h_['hole']} · hole edge", "peak": h_["peak_stress"], **fz})
        fz = fatigue(regular["fine"][0], regular["fine"][0], 1e3, su, se, cycle)
        fat_rows.append({"where": "max elsewhere", "peak": regular["fine"][0], **fz})
        fat = pd.DataFrame(fat_rows).sort_values("life_cycles", kind="stable").reset_index(drop=True) \
            if fat_rows else pd.DataFrame()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharp corners", len(sharp), help="Geometric singularities: the FE value there is not real.")
        m2.metric("BC artefacts", len(artef))
        preds = sharp.get("predicted_real_stress", pd.Series(dtype=float)).dropna()
        m3.metric("Highest ML real stress", f"{preds.max():.0f} MPa" if len(preds) else "—",
                  help="Largest predicted real (linear-elastic) peak stress over all sharp corners.")
        v_c, v_f = regular["coarse"][0], regular["fine"][0]
        m4.metric("Max converged elsewhere", f"{v_f:.0f} MPa",
                  help=f"Away from all singular points (holes, real fillets, straight edges). Coarse mesh gave "
                       f"{v_c:.0f} MPa ({100 * (v_f - v_c) / v_c:+.1f}%), so this value is mesh-converged.")

        left, right = st.columns([3, 2], gap="large")
        with left:
            tab_s, tab_c, tab_d, tab_p, tab_m = st.tabs(["Stress map", "Contours + probe", "Deformation",
                                                         "Photoelastic", "Mesh"])
            with tab_c:
                mfine = sols["fine"]["m"]
                fields = field_values(sols["fine"]["s"], part.get("E", 210000.0), part.get("nu", 0.3))
                k1, k2 = st.columns([3, 2])
                fkey = k1.selectbox("Field", list(FIELDS), format_func=lambda k: FIELDS[k][0], key="ct_field")
                cmap = k2.selectbox("Colours", list(CMAPS), key="ct_cmap")
                vals = fields[fkey]
                unit = FIELDS[fkey][1]
                near = singular_mask(mfine.p.T, flags)
                k3, k4 = st.columns([3, 2])
                rmode = k3.radio("Range", ["auto", "full", "manual"], horizontal=True, key="ct_range",
                                 format_func={"auto": "Auto (ignore singular spikes)", "full": "Auto (full)",
                                              "manual": "Manual"}.get)
                levels = k4.slider("Bands", 4, 30, 12, key="ct_levels")
                v_ok = vals[~near] if (~near).any() else vals
                if rmode == "auto":
                    lo, hi = float(v_ok.min()), float(v_ok.max())
                else:
                    lo, hi = float(vals.min()), float(vals.max())
                if rmode == "manual":
                    k5, k6 = st.columns(2)
                    lo = k5.number_input(f"Min ({unit})", value=float(f"{lo:.4g}"), format="%g", key=f"ct_lo_{fkey}")
                    hi = k6.number_input(f"Max ({unit})", value=float(f"{hi:.4g}"), format="%g", key=f"ct_hi_{fkey}")
                if hi <= lo:
                    hi = lo + max(abs(lo), 1.0) * 1e-3
                excl = st.checkbox("Max / min markers: exclude flagged singular points", value=True, key="ct_excl",
                                   help="At a sharp corner or clamp end the raw FE value is set by the mesh, "
                                        "not the physics. Excluding them shows the real extremes.")
                pool = ~near if (excl and (~near).any()) else np.ones(len(vals), bool)
                imax = np.flatnonzero(pool)[np.argmax(vals[pool])]
                imin = np.flatnonzero(pool)[np.argmin(vals[pool])]
                extremes = {"MAX": (mfine.p[0, imax], mfine.p[1, imax], vals[imax]),
                            "MIN": (mfine.p[0, imin], mfine.p[1, imin], vals[imin])}
                gx, gy, Z = contour_grid(mfine.p[0], mfine.p[1], mfine.t.T, vals)
                ss.setdefault("probes", [])
                probes = probe_values(mfine, fields, ss.probes)
                marks = []
                for _, r in flags[flags.singular].iterrows():
                    if r.type == "sharp re-entrant corner":
                        pr_ = r.get("predicted_real_stress", np.nan)
                        marks.append((r.x, r.y, f"{r.label} singularity · ML real {pr_:.0f} MPa"
                                      if np.isfinite(pr_) else f"{r.label} singularity", True))
                    else:
                        marks.append((r.x, r.y, f"{r.label} artefact", False))
                cfig = contour_figure(gx, gy, Z, lo, hi, levels, cmap, unit, probes, extremes, marks=marks)
                ckey = f"ct_{fkey}_{len(ss.probes)}_{ss.get('run_id', 0)}"
                cev = st.plotly_chart(cfig, width="stretch", on_select="rerun", selection_mode="points", key=ckey,
                                      config={"displaylogo": False, "scrollZoom": True})
                try:
                    cpts = cev.selection.points if cev else []
                except AttributeError:
                    cpts = (cev or {}).get("selection", {}).get("points", [])
                if cpts:
                    px_, py_ = float(cpts[0]["x"]), float(cpts[0]["y"])
                    if ss.get("last_probe") != (px_, py_, ckey):
                        ss.last_probe = (px_, py_, ckey)
                        ss.probes.append((px_, py_))
                        st.rerun()
                st.caption("Hover for a live reading · click to pin a probe (P1, P2, …) · scroll to zoom, drag to pan. "
                           f"▲ MAX {vals[imax]:.4g} {unit} at ({mfine.p[0, imax]:.1f}, {mfine.p[1, imax]:.1f}) · "
                           f"▼ MIN {vals[imin]:.4g} {unit} at ({mfine.p[0, imin]:.1f}, {mfine.p[1, imin]:.1f})")
                if probes:
                    pt = pd.DataFrame(probes)
                    show = pd.DataFrame({"probe": [f"P{i + 1}" for i in range(len(pt))],
                                         "x, y (mm)": [f"{a:.2f}, {b:.2f}" for a, b in zip(pt.x, pt.y)],
                                         "von Mises": pt.vm.round(1), "σ1": pt.s1.round(1), "σ2": pt.s2.round(1),
                                         "τmax": pt.tmax.round(1), "ε1 (µε)": pt.e1.round(0)})
                    st.dataframe(show, hide_index=True, width="stretch")
                    warn = [f"P{i + 1}" for i, (x, y) in enumerate(ss.probes)
                            if singular_mask(np.array([[x, y]]), flags, 0.15)[0]]
                    if warn:
                        st.markdown(f'<div class="note">NOTE → {", ".join(warn)}: next to a flagged singular point, '
                                    'so the FE value there depends on the mesh. Use the predicted real stress instead.'
                                    '</div>', unsafe_allow_html=True)
                    if st.button("Clear probes", key="clear_probes"):
                        ss.probes = []
                        st.rerun()
            with tab_s:
                st.pyplot(result_figure(part, sols, flags, regular, hole_summary, mat), width="stretch")
            with tab_m:
                mc1, mc2 = st.columns(2)
                which = mc1.radio("Mesh", ["fine", "coarse"], horizontal=True, key="mesh_which",
                                  format_func={"fine": "Fine (results)", "coarse": "Coarse (check)"}.get)
                targets = {"Whole part": None}
                for _, r in flags[flags.singular].iterrows():
                    targets[f"{r.label} · {r.type.split(' (')[0]}"] = (r.x, r.y, max(0.35 * r.ell, 1e-3))
                for h in hole_summary:
                    c, rr = part["circles"][int(h["hole"][1:])]
                    targets[f"{h['hole']} · hole"] = (c[0], c[1], 2.2 * rr)
                zoom = mc2.selectbox("Zoom to", list(targets), key="mesh_zoom")
                m_ = sols[which]["m"]
                tg = targets[zoom]
                st.pyplot(mesh_figure(m_, flags, hole_summary,
                                      None if tg is None else tg[:2], None if tg is None else tg[2],
                                      title=f"{which.upper()} MESH · {zoom.upper()}"), width="stretch")
                ms_c, ms_f = mesh_stats(sols["coarse"]["m"]), mesh_stats(sols["fine"]["m"])
                st.dataframe(pd.DataFrame({
                    "": ["nodes", "elements", "degrees of freedom", "smallest element (mm)", "largest element (mm)"],
                    "coarse": [f"{ms_c['nodes']:,}", f"{ms_c['elements']:,}", f"{ms_c['dof']:,}",
                               f"{ms_c['h_min']:.3f}", f"{ms_c['h_max']:.2f}"],
                    "fine": [f"{ms_f['nodes']:,}", f"{ms_f['elements']:,}", f"{ms_f['dof']:,}",
                             f"{ms_f['h_min']:.3f}", f"{ms_f['h_max']:.2f}"]}),
                    hide_index=True, width="stretch")
                st.caption("6-node quadratic triangles (like ANSYS PLANE183); drawn as their corner triangles. "
                           "The fine mesh halves the element size at every corner, support end and hole; "
                           "singular points are those whose stress keeps rising between the two.")
            with tab_d:
                deformation_tab(part, sols, flags, thickness)
            with tab_p:
                photo_tab(part, sols, flags, thickness)
        with right:
            st.markdown("### Sharp corners → real stress")
            if len(sharp):
                order = sharp.assign(_k=sharp.get("predicted_real_stress", pd.Series(0.0, index=sharp.index)).fillna(0)) \
                    .sort_values("_k", ascending=False)
                MAXCARDS = 5
                if len(sharp) > MAXCARDS:
                    st.caption(f"{len(sharp)} sharp corners. Showing the {MAXCARDS} with the highest real stress; "
                               "all are in the table below.")
                for _, r in order.head(MAXCARDS).iterrows():
                    pr_ = r.get("predicted_real_stress", np.nan)
                    rng_ = (f'<br>90 % RANGE {r.pred_low_90:.0f}–{r.pred_high_90:.0f} MPa (±{r.pred_band_pct:.0f} %, calibrated)'
                            if np.isfinite(r.get("pred_low_90", np.nan)) else '')
                    ml_line = (f'<div class="ml">ML PREDICTED REAL STRESS · r = {r.corner_radius:g} mm'
                               f'<span class="big">{pr_:.0f} MPa</span>{rng_}</div>') if np.isfinite(pr_) else \
                        '<div class="ml">ML PREDICTED REAL STRESS<span class="big">—</span></div>'
                    yld = ""
                    if np.isfinite(pr_):
                        if pr_ > sy and plast:
                            yld = (f'<div class="yl">ABOVE YIELD ({sy:g} MPa) → ACTUAL AFTER LOCAL YIELDING '
                                   f'<b>{r.actual_after_yield:.0f} MPa</b> · plastic strain '
                                   f'<b>{r.plastic_strain_pct:.2f}%</b> (Neuber)'
                                   + (' · <b>very large plastic strain: cracking likely, redesign this corner</b>'
                                      if r.plastic_strain_pct > 5 else '') + '</div>')
                        elif pr_ > sy:
                            yld = f'<div class="yl">ABOVE YIELD ({sy:g} MPa): the part yields locally here</div>'
                        else:
                            yld = f'<div class="yl">Below yield ({sy:g} MPa): elastic, prediction is the actual stress</div>'
                    rq_, rs_ = r.get("required_radius", np.nan), r.get("required_status", "")
                    if rs_ == "ok":
                        req = (f'<div class="yl">REQUIRED RADIUS for ≤ {allow:.0f} MPa (σy/{sf:g}'
                           + (', 90 % confidence' if conf_r else '') + ') → '
                               f'<b>r ≥ {rq_:.2f} mm</b>' + (' · current radius is enough' if r.corner_radius >= rq_
                                                              else ' · <b>increase the radius</b>') + '</div>')
                    elif rs_ == "any":
                        req = (f'<div class="yl">REQUIRED RADIUS for ≤ {allow:.0f} MPa → any radius ≥ {rq_:.2f} mm '
                               'works (low-stress corner)</div>')
                    elif rs_ == "impossible":
                        req = (f'<div class="yl">REQUIRED RADIUS for ≤ {allow:.0f} MPa → <b>a fillet alone cannot '
                               f'get there</b> (still above it at r = l = {r.ell:.1f} mm): add material or reduce '
                               'the load here</div>')
                    else:
                        req = ""
                    fp_ = r.get("fingerprint", "n/a")
                    if fp_ in ("ok", "check", "fail") and np.isfinite(r.get("lambda_measured", np.nan)):
                        fpl = (f'<div class="sg">SINGULARITY FINGERPRINT · measured exponent λ = {r.lambda_measured:.2f}, '
                               f'Williams theory λ = {r.williams_lambda:.2f} (growth ×{r.refine_ratio:.2f} vs ×{r.growth_theory:.2f}) → '
                               + ('<b>matches: a genuine corner singularity</b>' if fp_ == "ok" else
                                  '<b>does not match theory: the mesh may not resolve this corner, or a nearby feature '
                                  'dominates it; treat the prediction with care</b>') + '</div>')
                    else:
                        fpl = ""
                    st.markdown(
                        f'<div class="card"><div class="hd">{r.label} · sharp corner {r.angle_deg:.0f}°</div>'
                        + (f'<div class="sg">SINGULARITY (FE value at the point) · {r.fe_stress_fine:.0f} MPa in the '
                           f'ANSYS result. A sharp re-entrant corner (λ = {r.williams_lambda:.3f} < 1): theory says this '
                           f'value grows without limit as the mesh is refined, so it is not a real stress. Upload a '
                           f'coarse result too to see it rise.</div>' if sols.get("geometric_only") else
                           f'<div class="sg">SINGULARITY (FE value at the point) · {r.fe_stress_coarse:.0f} → '
                           f'{r.fe_stress_fine:.0f} MPa as the mesh is halved (×{r.refine_ratio:.2f}). '
                           f'Not a real stress: it keeps rising with every refinement.</div>') +
                        f'{fpl}{ml_line}{yld}{req}</div>', unsafe_allow_html=True)
                sharp = order.drop(columns="_k")
                t = pd.DataFrame({
                    "point": list(sharp.label),
                    "FE singular": sharp.fe_stress_fine.round(0),
                    "ML real (MPa)": sharp.get("predicted_real_stress", pd.Series(np.nan, index=sharp.index)).round(0),
                    "90 % range": [f"{a:.0f}–{b:.0f}" if np.isfinite(a) else "—"
                                   for a, b in zip(sharp.pred_low_90, sharp.pred_high_90)],
                    "λ meas/theory": [f"{m:.2f}/{t:.2f}" if np.isfinite(m) else "—" for m, t in
                                      zip(sharp.get("lambda_measured", pd.Series(np.nan, index=sharp.index)),
                                          sharp.get("williams_lambda", pd.Series(np.nan, index=sharp.index)))],
                    "after yield": sharp.get("actual_after_yield", pd.Series(np.nan, index=sharp.index)).round(0),
                    "r (mm)": sharp.corner_radius,
                    f"r needed ≤{allow:.0f}": [f"{x:.2f}" if s_ == "ok" else ("any" if s_ == "any" else
                                               ("> l" if s_ == "impossible" else "—"))
                                               for x, s_ in zip(sharp.required_radius, sharp.required_status)],
                })
                st.dataframe(t, hide_index=True, width="stretch")
                st.caption("FE singular = what an FE contour shows at the corner (mesh-dependent, not real). ML real = "
                           "the model's linear-elastic peak for the corner radius in the sidebar. After yield = Neuber "
                           "estimate when the elastic peak exceeds σy. Change radius / σy and everything updates "
                           "without re-solving.")
                for _, r in sharp.iterrows():
                    if isinstance(r.get("note"), str) and r.note:
                        st.markdown(f'<div class="note">NOTE → {r.label}: {r.note}</div>', unsafe_allow_html=True)
            else:
                st.write("No sharp re-entrant corners with singular stress.")
            if len(artef):
                st.markdown("### Support / load artefacts")
                st.dataframe(pd.DataFrame({"point": list(artef.label),
                                           "FE fine (MPa)": artef.fe_stress_fine.round(0),
                                           "rise on refining": [f"×{x:.2f}" for x in artef.refine_ratio]}),
                             hide_index=True, width="stretch")
                st.caption("These come from idealised boundary conditions: a perfectly rigid clamp, a load that "
                           "starts abruptly, or the sharp edge of an idealised pin contact (H…@ angle). They are not "
                           "physical stresses. Read stresses away from them, or model the real support / contact.")
            if hole_summary:
                st.markdown("### Holes → peak stress")
                hs = pd.DataFrame(hole_summary)
                st.dataframe(pd.DataFrame({
                    "hole": hs.hole, "fixture": hs.fixture, "Ø": (2 * hs.radius_mm).round(1),
                    "peak (MPa)": hs.peak_stress.round(0), "at °": hs.at_angle_deg.round(0),
                    "Δ coarse": [f"{x:+.1f}%" for x in hs.change_pct],
                    "sing.": hs.singular_points}), hide_index=True, width="stretch")
                st.caption("Highest stress on each hole edge away from flagged points. A small change vs the coarse "
                           "mesh means it is converged and real (this is the hole's stress concentration).")
                over = [h["hole"] for h in hole_summary if h["peak_stress"] > sy]
                if over:
                    st.markdown(f'<div class="note">NOTE → {", ".join(over)}: hole peak above yield ({sy:g} MPa); '
                                'the hole edge yields locally.</div>', unsafe_allow_html=True)
            if v_f > sy:
                st.markdown(f'<div class="note">NOTE → max converged stress elsewhere ({v_f:.0f} MPa) is above yield '
                            f'({sy:g} MPa): the part yields there too.</div>', unsafe_allow_html=True)

        if len(fat):
            st.markdown("### Fatigue → where it cracks first")
            top = fat.iloc[0]
            st.markdown(
                f'<div class="card"><div class="hd">Cracks first: {top["where"]}</div>'
                f'<div class="ml">ESTIMATED LIFE ({"reversed ±F" if cycle == "R=-1" else "on/off 0→F"} at the '
                f'applied load)<span class="big">{life_text(top.life_cycles)} cycles</span></div>'
                f'<div class="yl">fatigue stress {top.fatigue_stress:.0f} MPa (Kf {top.Kf:.2f} from Kt {top.Kt:.2f}, '
                f'notch sensitivity q {top.q:.2f}) · equivalent amplitude {top.eq_amplitude:.0f} MPa vs '
                f'Se {se:.0f} MPa</div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "rank": range(1, len(fat) + 1), "where": fat["where"], "elastic peak": fat.peak.round(0),
                "Kf/Kt": [f"{a:.2f}/{b:.2f}" for a, b in zip(fat.Kf, fat.Kt)],
                "fatigue σ": fat.fatigue_stress.round(0), "life (cycles)": [life_text(x) for x in fat.life_cycles]}),
                hide_index=True, width="stretch")
            st.caption("Ranks every hot spot by fatigue life if the applied loads repeat. Sharp corners use the ML "
                       "real peak, softened by Peterson's notch sensitivity (small radii hurt fatigue less than their "
                       "peak suggests); holes and the max elsewhere use their converged peak. S-N line through "
                       "0.9 σu at 10³ and Se at 10⁶ cycles, Goodman for on/off loading (Shigley). Steel-type "
                       "estimate for comparing locations, not a certified life.")
        # DXF with every sharp corner given its required radius (never below the real radius in the sidebar)
        fix_r = {}
        for _, r_ in sharp.iterrows():
            if r_.required_status in ("ok", "any"):
                fix_r[r_.label] = max(float(r_.required_radius), rho)
        if fix_r:
            from design_tools import apply_fillets, max_fillet
            from sing_tool import part_holes as _ph
            out_poly = [tuple(map(float, p)) for p in part["poly"]]
            rv = {int(l_[1:]): r_ for l_, r_ in fix_r.items() if l_.startswith("V")}
            rv = {i: min(r_, max_fillet(out_poly, i)) for i, r_ in rv.items()}
            new_out = apply_fillets(out_poly, rv)
            _, cw = _ph(part)
            new_holes = []
            for k_, ring_ in enumerate(cw):
                rh = {int(l_.split(".")[1]): r_ for l_, r_ in fix_r.items() if l_.startswith(f"C{k_}.")}
                rh = {i: min(r_, max_fillet(ring_, i)) for i, r_ in rh.items()}
                new_holes.append(apply_fillets([tuple(map(float, p)) for p in ring_], rh))
            import tempfile, os
            from topopt import write_dxf as _wd
            with tempfile.TemporaryDirectory() as td_:
                pth = os.path.join(td_, "fixed.dxf")
                _wd({"outline": new_out, "poly_holes": new_holes,
                     "circles": [(tuple(c_), r_) for c_, r_ in part.get("circles", [])]}, pth)
                dxf_bytes = open(pth, "rb").read()
            st.download_button(f"Download DXF with required fillets ({len(fix_r)} corners, ≤ {allow:.0f} MPa)",
                               dxf_bytes, file_name=f"{Path(name).stem}_required_fillets.dxf", width="stretch")
            st.caption("Every sharp corner rounded to its required radius. Upload this DXF and run it to check: the "
                       "fillets are drawn arcs, so the FE stresses there are real and converged.")

        rep = report_text(name, {"t": thickness, "E": E, "nu": nu}, sorted(ss.supports), loads, edges, flags, regular,
                          hole_bc, hole_summary, ss.forces)
        rep += (f"\n\n## Design checks\nAllowable stress σy/SF = {sy:g}/{sf:g} = {allow:.0f} MPa.\n\n"
                "| corner | ML real peak (MPa) at r | required radius (mm) |\n|---|---|---|\n" +
                "".join(f"| {r_.label} | {r_.get('predicted_real_stress', np.nan):.0f} at {r_.corner_radius:g} mm | "
                        f"{('%.2f' % r_.required_radius) if r_.required_status == 'ok' else r_.required_status} |\n"
                        for _, r_ in sharp.iterrows()))
        if len(fat):
            rep += (f"\n## Fatigue ranking ({'fully reversed' if cycle == 'R=-1' else 'zero to max'}, σu {su:g} MPa, "
                    f"Se {se:.0f} MPa)\n\n| rank | where | fatigue stress (MPa) | life (cycles) |\n|---|---|---|---|\n" +
                    "".join(f"| {i + 1} | {r_['where']} | {r_.fatigue_stress:.0f} | {life_text(r_.life_cycles)} |\n"
                            for i, r_ in fat.iterrows()))
        d1, d2 = st.columns(2)
        d1.download_button("Download report (Markdown)", rep, file_name=f"{Path(name).stem}_singularity_report.md")
        d2.download_button("Download results (CSV)", flags.drop(columns=["feats"], errors="ignore").to_csv(index=False),
                           file_name=f"{Path(name).stem}_singular_points.csv")


with tab_opt:
    # ------------------------------------------------------------------ 05 · topology optimisation
    st.markdown('<p class="eyebrow">Topology optimisation (SIMP, minimum compliance): keeps the stiffest '
                'layout of material for a target volume, using the supports and loads above. Material around '
                'loaded / fixed holes and edges is always kept.</p>', unsafe_allow_html=True)
    if not ready:
        st.markdown('<p class="eyebrow">→ Set supports and loads first (section 03)</p>', unsafe_allow_html=True)
    else:
        o1, o2, o3, o4 = st.columns(4)
        vf_pct = o1.number_input("Material to keep (%)", min_value=5.0, max_value=95.0, value=40.0, step=5.0,
                                 format="%g", key="to_vf",
                                 help="Percentage of the design area kept after optimisation (5-95).")
        vf = vf_pct / 100.0
        n_target = int(o2.number_input("Resolution (elements)", min_value=1000, max_value=200000, value=15000,
                                       step=1000, key="to_res",
                                       help="Number of triangles in the optimisation mesh. More = finer struts and "
                                            "smoother shape, but slower: roughly 10 s per 15,000 elements on a PC "
                                            "(longer online). 6,000 = quick look, 15,000 = default, 30,000+ = fine."))
        n_iter = int(o3.number_input("Iterations", min_value=10, max_value=500, value=80, step=10, key="to_iter",
                                     help="Optimisation steps. It stops earlier once the design stops changing."))
        start = o4.radio("Start from", ["solid", "current"], key="to_start", horizontal=True,
                         format_func={"solid": "Solid outline", "current": "Current shape"}.get,
                         help="Solid outline ignores existing cutouts (bolt holes are always kept).")
        fil_r = o1.number_input("Fillet radius on new shape (mm)", value=3.0, min_value=0.0, step=0.5, format="%g",
                                key="to_fil",
                                help="Every corner of the optimised shape is rounded with this radius, as you would "
                                     "do in CAD before making it (no singularities left; the stress at each fillet is "
                                     "then a real, converged FE value). 0 keeps sharp corners so EigenStress predicts "
                                     "their real stress for the corner radius set in the sidebar.")
        run_opt = st.button("Optimise", key="to_run")
        prog = st.empty()                    # fixed slot for the progress bar (keeps the layout stable)
        if run_opt:
            part0 = build_part(geom, edges, ss.supports, loads, thickness, E, nu, rho, hole_bc)
            if start == "solid":
                part0 = dict(part0)
                part0["poly_holes"] = []
            from shapely.geometry import Polygon as _Pg
            design_area = _Pg(geom["outline"]).area - sum(np.pi * r_ ** 2 for _, r_ in geom["circles"])
            hsel = float(np.sqrt(max(design_area, 1e-9) / (0.43 * n_target)))   # ~equilateral triangles
            bar = prog.progress(0.0, text="Optimising…")

            def cb(i, n, c, v):
                bar.progress(i / n, text=f"Iteration {i}/{n} · compliance {c:.4g} · volume {v:.2f}")
            try:
                from sing_tool import min_wall
                if part0.get("poly_holes"):
                    hsel = min(hsel, max(min_wall(part0) / 2.5, hsel / 3))
                tres = optimise(part0, volfrac=vf, h=hsel, iters=n_iter, callback=cb)
                tgeo, hmap = extract_shape(tres, part0, round_r=float(fil_r))
                if not tgeo.get("connected", True):
                    st.warning("The optimised layout splits into separate pieces at this resolution / material "
                               "fraction, so it cannot be analysed as one part. Try more material or a finer resolution.")
                ss.topo = {"res": tres, "geo": tgeo, "hole_map": hmap, "part": part0, "vf": vf, "fil_r": float(fil_r)}
            except Exception as ex:
                st.error(f"Optimisation failed: {ex}")
            prog.empty()
        if ss.get("topo"):
            T = ss.topo
            tres, tgeo = T["res"], T["geo"]
            hst = pd.DataFrame(tres["history"])
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Material kept", f"{100 * tgeo['volume_kept']:.0f}%")
            t2.metric("Mass saved", f"{100 * (1 - tgeo['volume_kept']):.0f}%")
            rel = tres["c_full"] / hst.compliance.iloc[-1]
            t3.metric("Stiffness vs solid", f"{100 * rel:.0f}%",
                      help="Compliance of the full solid part divided by the optimised compliance "
                           "(100% = as stiff as the solid part).")
            t4.metric("Iterations", len(hst))
            cA, cB = st.columns(2)
            mo = tres["mesh"]
            figd, axd = plt.subplots(figsize=(7, 5), facecolor=BG)
            axd.set_facecolor(BG)
            axd.tripcolor(mo.p[0], mo.p[1], mo.t.T, facecolors=tres["density"], cmap="gray", vmin=0, vmax=1)
            axd.set_aspect("equal"); axd.axis("off")
            axd.set_title("DENSITY (WHITE = MATERIAL)", color="#8c8c8c", fontsize=9, family="monospace", loc="left")
            cA.pyplot(figd, width="stretch")
            figs, axs = plt.subplots(figsize=(7, 5), facecolor=BG)
            axs.set_facecolor(BG)
            o_ = np.array(tgeo["outline"] + [tgeo["outline"][0]])
            axs.fill(o_[:, 0], o_[:, 1], color="#5a5a5a"); axs.plot(o_[:, 0], o_[:, 1], color=FG, lw=1)
            for hh in tgeo["poly_holes"]:
                hh = np.array(hh + [hh[0]]); axs.fill(hh[:, 0], hh[:, 1], color=BG); axs.plot(hh[:, 0], hh[:, 1], color=FG, lw=1)
            for c_, r_ in tgeo["circles"]:
                tt = np.linspace(0, 2 * np.pi, 60)
                axs.fill(c_[0] + r_ * np.cos(tt), c_[1] + r_ * np.sin(tt), color=BG)
                axs.plot(c_[0] + r_ * np.cos(tt), c_[1] + r_ * np.sin(tt), color=FG, lw=1)
            axs.set_aspect("equal"); axs.axis("off")
            axs.set_title(f"EXTRACTED SHAPE · {len(tgeo['poly_holes'])} CUTOUTS", color="#8c8c8c", fontsize=9,
                          family="monospace", loc="left")
            cB.pyplot(figs, width="stretch")
            if ss.topo.get("fil_r", 0) > 0:
                st.caption(f"The density field is smoothed into a clean outline and every corner is filleted with "
                           f"r = {ss.topo['fil_r']:g} mm, so the new part has no stress singularities: EigenStress reports "
                           "real, converged stresses at the fillets. Set the fillet radius to 0 to keep sharp corners "
                           "and have EigenStress predict their real stress instead.")
            else:
                st.caption("Sharp corners kept where struts meet. Send it to EigenStress to see which are singular and "
                           "what their real stress is for the corner radius in the sidebar.")
            import tempfile, os
            fd, tmpdxf = tempfile.mkstemp(suffix=".dxf"); os.close(fd)
            write_dxf(tgeo, tmpdxf)
            dxf_bytes = open(tmpdxf, "rb").read(); os.remove(tmpdxf)
            b1, b2 = st.columns(2)
            b1.download_button("Download optimised shape (DXF)", dxf_bytes,
                               file_name=f"{Path(name).stem}_optimised_{100 * T['vf']:g}pct.dxf")
            if b2.button("Analyse optimised shape with EigenStress →", width="stretch"):
                n_e_old = len(edges)
                seg_of = lambda i: [(tuple(p), tuple(q)) for p, q in edges[i]["segs"]]
                fo_list = []
                for fo in ss.forces:
                    e_t = [t for t in fo["targets"] if t < n_e_old]
                    h_t = [t - n_e_old for t in fo["targets"] if t >= n_e_old]
                    F_ = fo["F"]
                    mode = fo["mode"]
                    if mode == "each" and e_t:                 # keep the same total force on the edges
                        F_ = (F_[0] * len(e_t), F_[1] * len(e_t)) if not h_t else F_
                        mode = "split" if not h_t else mode
                    fo_list.append({"F": F_, "mode": mode, "holes": h_t,
                                    "edge_segs": [sg for i in e_t for sg in seg_of(i)]})
                ss.pending_bc = {"hole_map": T["hole_map"], "hole_fix": dict(ss.hole_fix),
                                 "fixed_segs": [sg for i in ss.supports for sg in seg_of(i)], "forces": fo_list}
                ss.opt_geom = {"geom": tgeo, "name": f"{Path(name).stem} · optimised {100 * T['vf']:g}%",
                               "key": f"opt_{gkey}_{T['vf']}_{len(tgeo['outline'])}_{len(tgeo['poly_holes'])}"}
                ss.goto_tab = TAB_AN
            st.rerun()

with st.expander("How it works"):
    st.markdown(
        "1. **Solve twice.** The part is meshed with quadratic triangles (like ANSYS PLANE183), refined at every "
        "corner and support end, and solved at two mesh sizes.\n"
        "2. **Flag.** A point whose stress rises by more than 5% when the local mesh is halved is singular. At a "
        "sharp corner of angle β, elastic stress follows σ ∝ r^(λ−1) (Williams), so it goes to infinity and the FE "
        "value is set only by the mesh.\n"
        "3. **Classify.** Sharp re-entrant corners (angle > 190°) are geometric singularities. Ends of fixed or "
        "loaded edges are boundary-condition artefacts.\n"
        "4. **Predict.** The stresses at 0.1–0.3 l from each sharp corner converge on any reasonable mesh. "
        "A model trained on ~91,000 sharp-vs-filleted FE pairs maps them, plus the real corner radius, to the "
        "finite peak stress.\n\n"
        "5. **Holes.** Each circular hole can be fixed (bolted), rest on a frictionless pin (radial contact on "
        "one half), or carry a bolt bearing load (cosine pressure on the loaded half). The hole edge is scanned "
        "at both meshes: points that keep rising (e.g. edges of a pin contact patch) are flagged; the converged "
        "hole peak is reported.\n\n"
        "**Limits:** 2D plane stress, linear elastic. Pin contact is modelled as bonded-radial on a fixed 180° "
        "patch (no gap/friction). Corner angles 190–330°, corner radius 1–100% of the local feature size (best below 30%).")
