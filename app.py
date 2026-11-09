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
from sing_tool import solve_two_meshes, analyse, max_regular_stress, predict

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
.note { font-family: 'Courier New', Courier, monospace; font-size: 13px; color: var(--fg); border: 1px solid var(--border); padding: 10px 14px; margin: 8px 0; }
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


def geometry_figure(geom, edges, supports, loads, picked, height=520):
    fig = go.Figure()
    for idx, e in enumerate(edges):
        pts = np.array([e["segs"][0][0]] + [q for _, q in e["segs"]])
        if idx in supports:
            color, width, name = C_SUPPORT, 7, "support"
        elif idx in loads:
            color, width, name = C_LOAD, 3, "load"
        elif idx == picked:
            color, width, name = C_PICK, 6, "selected"
        else:
            color, width, name = C_EDGE, 2.5, "edge"
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines",
                                 line=dict(color=color, width=width, dash="dot" if name == "load" else "solid"),
                                 hoverinfo="skip", showlegend=False))
    for (c, r) in geom["circles"]:
        t = np.linspace(0, 2 * np.pi, 80)
        fig.add_trace(go.Scatter(x=c[0] + r * np.cos(t), y=c[1] + r * np.sin(t), mode="lines",
                                 line=dict(color=C_EDGE, width=2), hoverinfo="skip", showlegend=False))
    for h in geom["poly_holes"]:
        p = np.array(h + [h[0]])
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=C_EDGE, width=2),
                                 hoverinfo="skip", showlegend=False))
    size = np.ptp(np.array(geom["outline"]), axis=0).max()
    for idx, (fx, fy) in loads.items():
        m = edges[idx]["mid"]
        mag = np.hypot(fx, fy)
        if mag > 0:
            d = np.array([fx, fy]) / mag * size * 0.12
            fig.add_annotation(x=m[0], y=m[1], ax=m[0] - d[0], ay=m[1] - d[1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4,
                               arrowwidth=2.5, arrowcolor=C_LOAD,
                               text=f"{mag:.0f} N", font=dict(color=FG, size=12, family=MONO))
    mids = np.array([e["mid"] for e in edges])
    fig.add_trace(go.Scatter(
        x=mids[:, 0], y=mids[:, 1], mode="markers+text", text=[f"E{i}" for i in range(len(edges))],
        textposition="top center", textfont=dict(color=FG, size=12, family=MONO),
        marker=dict(size=14, symbol="square", color=[C_PICK if i == picked else BG for i in range(len(edges))],
                    line=dict(color=[C_PICK if i == picked else "#8c8c8c" for i in range(len(edges))], width=1.5)),
        customdata=list(range(len(edges))),
        hovertemplate="E%{customdata}: click to select<extra></extra>", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                      font=dict(family=MONO, color="#8c8c8c", size=11),
                      hoverlabel=dict(bgcolor=SURFACE, bordercolor=FG, font=dict(family=MONO, color=FG)),
                      clickmode="event+select", dragmode="pan")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, title="X (MM)")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, scaleanchor="x", scaleratio=1,
                     title="Y (MM)")
    return fig


def build_part(geom, edges, supports, loads, thickness, E, nu, rho):
    fixed, trac = [], []
    for idx in supports:
        fixed += [(tuple(p), tuple(q)) for p, q in edges[idx]["segs"]]
    for idx, (fx, fy) in loads.items():
        L = edges[idx]["length"]
        tx, ty = fx / (L * thickness), fy / (L * thickness)          # N / mm^2 = MPa
        trac += [(tuple(p), tuple(q), (tx, ty)) for p, q in edges[idx]["segs"]]
    return {"poly": [np.asarray(p, float) for p in geom["outline"]], "circles": geom["circles"],
            "poly_holes": geom["poly_holes"], "fixed": fixed, "traction": trac,
            "E": E, "nu": nu, "corner_radius": rho}


def result_figure(part, sols, flags, regular):
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
    for _, r in flags[flags.singular].iterrows():
        geo = r.type == "sharp re-entrant corner"
        ax.plot(r.x, r.y, "o", ms=14, mfc="none", mec=C_SING if geo else "#8c8c8c", mew=2)
        lab = f"V{int(r.vertex)}"
        if geo and np.isfinite(r.get("predicted_real_stress", np.nan)):
            lab += f" → {r.predicted_real_stress:.0f} MPA"
        ax.annotate(lab, (r.x, r.y), xytext=(10, 10), textcoords="offset points", fontsize=9, color=FG,
                    family=mono, bbox=box)
    v, p = regular["fine"]
    ax.plot(*p, "s", ms=9, mfc="none", mec=FG, mew=1.6)
    ax.annotate(f"MAX ELSEWHERE {v:.0f} MPA", p, xytext=(10, -16), textcoords="offset points", fontsize=9,
                color=FG, family=mono, bbox=box)
    ax.plot([], [], "o", ms=10, mfc="none", mec=C_SING, mew=2, label="SHARP CORNER · REAL STRESS PREDICTED")
    ax.plot([], [], "o", ms=10, mfc="none", mec="#8c8c8c", mew=2, label="SUPPORT / LOAD ARTEFACT")
    ax.plot([], [], "s", ms=9, mfc="none", mec=FG, mew=1.6, label="MAX CONVERGED STRESS ELSEWHERE")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
                    prop={"family": mono, "size": 8})
    for t in leg.get_texts():
        t.set_color("#8c8c8c")
    fig.tight_layout()
    return fig


def report_text(name, settings, supports, loads, edges, flags, regular):
    lines = [f"# Singularity report: {name}", "",
             f"Thickness {settings['t']} mm, E = {settings['E']:.0f} MPa, nu = {settings['nu']}, "
             f"plane stress, linear elastic.", "",
             "## Boundary conditions",
             *[f"- E{i}: fixed support (length {edges[i]['length']:.1f} mm)" for i in supports],
             *[f"- E{i}: force Fx = {fx:.1f} N, Fy = {fy:.1f} N" for i, (fx, fy) in loads.items()], "",
             "## Singular points", "",
             "| Point | Angle (deg) | Type | FE coarse (MPa) | FE fine (MPa) | Rise on refining | "
             "Corner radius (mm) | Predicted real stress (MPa) | Note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in flags[flags.singular].iterrows():
        pr = r.get("predicted_real_stress", np.nan)
        lines.append(f"| V{int(r.vertex)} ({r.x:.1f}, {r.y:.1f}) | {r.angle_deg:.0f} | {r.type} | "
                     f"{r.fe_stress_coarse:.1f} | {r.fe_stress_fine:.1f} | x{r.refine_ratio:.2f} | "
                     f"{r.get('corner_radius', np.nan) if r.type.startswith('sharp') else '-'} | "
                     f"{'-' if not np.isfinite(pr) else f'{pr:.1f}'} | {r.get('note', '') if isinstance(r.get('note', ''), str) else ''} |")
    v_c, v_f = regular["coarse"][0], regular["fine"][0]
    lines += ["", f"Highest converged stress away from singular points: {v_f:.1f} MPa at "
                  f"({regular['fine'][1][0]:.1f}, {regular['fine'][1][1]:.1f}) "
                  f"(coarse mesh {v_c:.1f} MPa, change {100 * (v_f - v_c) / v_c:+.1f}%).", "",
              "Method: FE solved at two meshes; points whose stress rises >5% when the local mesh is halved are "
              "singular. Sharp corners: the singular field 0.1-0.3 l from the corner (mesh-independent) is fed to "
              "a model trained on ~49,000 sharp-vs-filleted FE pairs to predict the finite peak stress for the "
              "given corner radius. Typical accuracy within the trained range: ~2% mean, ~5.5% at the 95th "
              "percentile. Support/load artefacts are not physical stresses and are not predicted."]
    return "\n".join(lines)


# ------------------------------------------------------------------ state
ss = st.session_state
for k, v in {"supports": set(), "loads": {}, "picked": None, "geom_key": None, "results": None,
             "pending_pick": None}.items():
    ss.setdefault(k, v)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("01 · Part")
    if not SAMPLES.exists() or not any(SAMPLES.glob("*.dxf")):
        SAMPLES.mkdir(exist_ok=True)
        write_sample_dxfs(str(SAMPLES))
    up = st.file_uploader("Upload a 2D drawing (DXF)", type=["dxf"])
    sample_names = sorted(p.stem for p in SAMPLES.glob("*.dxf"))
    sample = st.selectbox("…or try a sample", ["—"] + sample_names, index=0 if up else 1)
    scale = st.number_input("Drawing units → mm (multiply by)", value=1.0, min_value=1e-6, format="%g",
                            help="1 if the DXF is in mm, 25.4 if in inches, 1000 if in metres")
    st.header("02 · Material")
    thickness = st.number_input("Thickness t (mm)", value=10.0, min_value=0.01)
    E = st.number_input("Young's modulus E (MPa)", value=210000.0, min_value=1.0, format="%g")
    nu = st.number_input("Poisson's ratio (nu)", value=0.30, min_value=0.0, max_value=0.49)
    rho = st.number_input("Real corner radius at sharp corners (mm)", value=0.5, min_value=0.001,
                          help="Machined inside corners are never perfectly sharp - typically 0.2-1 mm. "
                               "The real stress depends on this radius.")

# ------------------------------------------------------------------ geometry
try:
    if up is not None:
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
    ss.supports, ss.loads, ss.picked, ss.results, ss.geom_key = set(), {}, None, None, gkey
    ss.pending_pick = 0
edges = edge_groups(geom["outline"])

st.markdown('<p class="eyebrow">FEM singularity finder · real-stress predictor</p>', unsafe_allow_html=True)
st.title("EigenStress")
st.markdown('<p style="font-size:18px;max-width:640px;margin-top:-4px">Finds the points where FE stress never converges, '
            'and predicts the real peak stress at every sharp corner.</p>', unsafe_allow_html=True)

col_plot, col_ctrl = st.columns([3, 2], gap="large")
with col_ctrl:
    st.subheader("03 · Supports + forces")
    st.write("Click an edge label (E0, E1, …) on the drawing, or pick it here:")
    opts = list(range(len(edges)))
    if ss.pending_pick is not None:                 # a click on the drawing selects that edge
        ss.edge_select = ss.pending_pick
        ss.pending_pick = None
    if ss.get("edge_select") not in opts:
        ss.edge_select = 0
    sel = st.selectbox("Selected edge", opts,
                       format_func=lambda i: f"E{i}  ·  {edges[i]['length']:.1f} mm"
                                             f"{'  (curved)' if edges[i]['curved'] else ''}",
                       key="edge_select")
    ss.picked = sel
    c1, c2 = st.columns(2)
    if c1.button("Fix this edge (support)", width="stretch"):
        ss.supports.add(sel)
        ss.loads.pop(sel, None)
        ss.results = None
        st.rerun()
    if c2.button("Clear this edge", width="stretch"):
        ss.supports.discard(sel)
        ss.loads.pop(sel, None)
        ss.results = None
        st.rerun()
    with st.form("force_form", border=True):
        st.write(f"Force on **E{sel}** (total, spread evenly along the edge)")
        f1, f2 = st.columns(2)
        fx = f1.number_input("Fx (N)", value=0.0, format="%g")
        fy = f2.number_input("Fy (N)", value=-1000.0, format="%g")
        if st.form_submit_button("Apply force to this edge", width="stretch"):
            ss.loads[sel] = (fx, fy)
            ss.supports.discard(sel)
            ss.results = None
            st.rerun()
    bc_rows = ([{"edge": f"E{i}", "condition": "fixed support", "Fx (N)": "", "Fy (N)": ""} for i in sorted(ss.supports)]
               + [{"edge": f"E{i}", "condition": "force", "Fx (N)": f"{fx:g}", "Fy (N)": f"{fy:g}"}
                  for i, (fx, fy) in sorted(ss.loads.items())])
    if bc_rows:
        st.dataframe(pd.DataFrame(bc_rows), hide_index=True, width="stretch")
    else:
        st.caption("No supports or forces yet.")

with col_plot:
    st.markdown(f'<p class="eyebrow">Part → {name}</p>', unsafe_allow_html=True)
    fig = geometry_figure(geom, edges, ss.supports, ss.loads, ss.picked)
    fig_key = f"geom_{len(ss.supports)}_{len(ss.loads)}_{ss.picked}"
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
            mids = np.array([e["mid"] for e in edges])
            cd = int(np.argmin(np.hypot(mids[:, 0] - p["x"], mids[:, 1] - p["y"])))
        if int(cd) != ss.picked and ss.get("last_click") != (int(cd), p.get("x"), p.get("y"), fig_key):
            ss.last_click = (int(cd), p.get("x"), p.get("y"), fig_key)
            ss.pending_pick = int(cd)
            st.rerun()
    holes = len(geom["circles"]) + len(geom["poly_holes"])
    st.caption(f"{len(edges)} edges · {holes} hole(s) · solid white = support · dotted = force · yellow = selected")

# ------------------------------------------------------------------ run
st.divider()
st.subheader("04 · Run")
ready = bool(ss.supports) and bool(ss.loads)
if not ready:
    st.markdown('<p class="eyebrow">→ Add at least one fixed edge and one force to run</p>', unsafe_allow_html=True)
run = st.button("Run analysis", type="primary", disabled=not ready)

if run:
    part = build_part(geom, edges, ss.supports, ss.loads, thickness, E, nu, rho)
    with st.spinner("Meshing and solving at two mesh sizes…"):
        try:
            sols = solve_two_meshes(part)
        except Exception as e:
            st.error(f"Solve failed: {e}")
            st.stop()
    ss.results = {"part": part, "sols": sols}

if ss.results:
    part, sols = ss.results["part"], ss.results["sols"]
    part["corner_radius"] = rho
    model = load_model()
    flags = analyse(part, sols, model)
    regular = max_regular_stress(part, sols, flags)
    sharp = flags[(flags.type == "sharp re-entrant corner")]
    artef = flags[(flags.type == "support / load artefact")]

    m1, m2, m3 = st.columns(3)
    m1.metric("Sharp corners found", len(sharp))
    m2.metric("Support / load artefacts", len(artef))
    v_c, v_f = regular["coarse"][0], regular["fine"][0]
    m3.metric("Max converged stress elsewhere", f"{v_f:.0f} MPa",
              help=f"Away from all singular points (holes, real fillets, straight edges). Coarse mesh gave "
                   f"{v_c:.0f} MPa ({100 * (v_f - v_c) / v_c:+.1f}%), so this value is mesh-converged.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.pyplot(result_figure(part, sols, flags, regular), width="stretch")
    with right:
        st.markdown("### Sharp corners → real stress")
        if len(sharp):
            t = pd.DataFrame({
                "point": [f"V{int(v)}" for v in sharp.vertex],
                "real stress (MPa)": sharp.get("predicted_real_stress", pd.Series(np.nan, index=sharp.index)).round(0),
                "radius": sharp.corner_radius,
                "angle": sharp.angle_deg.round(0),
                "FE coarse→fine": [f"{a:.0f} → {b:.0f}" for a, b in zip(sharp.fe_stress_coarse, sharp.fe_stress_fine)],
            })
            st.dataframe(t, hide_index=True, width="stretch")
            st.caption("FE coarse/fine = what the FE contour shows at the corner. It rises every time the mesh "
                       "is refined, so it is not a real stress. 'Real stress' is the model's prediction for the "
                       "corner radius set in the sidebar (change it and the table updates without re-solving).")
            for _, r in sharp.iterrows():
                if isinstance(r.get("note"), str) and r.note:
                    st.markdown(f'<div class="note">NOTE → V{int(r.vertex)}: {r.note}</div>', unsafe_allow_html=True)
        else:
            st.write("No sharp re-entrant corners with singular stress.")
        if len(artef):
            st.markdown("### Support / load artefacts")
            st.dataframe(pd.DataFrame({"point": [f"V{int(v)}" for v in artef.vertex],
                                       "FE fine (MPa)": artef.fe_stress_fine.round(0),
                                       "rise on refining": [f"×{x:.2f}" for x in artef.refine_ratio]}),
                         hide_index=True, width="stretch")
            st.caption("These come from idealised boundary conditions (perfectly rigid clamp, load on an edge "
                       "end). They are not physical. Read stresses away from them, or model the real support.")

    rep = report_text(name, {"t": thickness, "E": E, "nu": nu}, sorted(ss.supports), ss.loads, edges, flags, regular)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (Markdown)", rep, file_name=f"{Path(name).stem}_singularity_report.md")
    d2.download_button("Download results (CSV)", flags.to_csv(index=False),
                       file_name=f"{Path(name).stem}_singular_points.csv")

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
        "A model trained on ~49,000 sharp-vs-filleted FE pairs maps them, plus the real corner radius, to the "
        "finite peak stress.\n\n"
        "**Limits:** 2D plane stress, linear elastic, fixed (clamped) supports, forces on edges. Corner angles "
        "190–330°, corner radius 1–30% of the local feature size.")
