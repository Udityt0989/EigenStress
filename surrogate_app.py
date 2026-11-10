"""
surrogate_app.py - the "Instant" page: stress, deformation and photoelastic fringes for parametric parts,
straight from the trained surrogate (no FE at run time). A button runs one FE solve to check it.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from surrogate import (FAMILIES, Surrogate, predict, physical_fields, to_image, fe_solve, arc_profiles,
                       compliance, LAM_270)
from design_tools import fatigue, life_text
from views import deformed_figure, auto_exaggeration, photo_controls, photo_panel

HERE = Path(__file__).parent
BG, FG, MUTED, ACCENT, BORDER = "#0e0e0e", "#f2f2f2", "#5a5a5a", "#e8ff5a", "rgba(255,255,255,0.07)"
MONO = "Courier New, Courier, monospace"
JET = [[0, "#0000ff"], [0.25, "#00b4ff"], [0.5, "#00ff00"], [0.75, "#ffff00"], [1, "#ff0000"]]


@st.cache_resource(show_spinner=False)
def load_surrogate(key):
    return Surrogate.load(HERE / f"surrogate_{key}.npz")


@st.cache_data(show_spinner=False)
def validation_table():
    f = HERE / "surrogate_validation.csv"
    return pd.read_csv(f) if f.exists() else None


def _inputs(fam):
    """Physical inputs -> (dimensionless params, length scale L, a describe-the-part string)."""
    c1, c2, c3 = st.columns(3)
    if fam.key == "lbracket":
        L = c1.number_input("Outer size L (mm)", min_value=1.0, value=100.0, format="%g", key="sg_L")
        h1 = c2.number_input("Horizontal arm height (mm)", min_value=0.01, value=20.0, format="%g", key="sg_h1")
        h2 = c3.number_input("Vertical arm width (mm)", min_value=0.01, value=20.0, format="%g", key="sg_h2")
        c4, c5, c6 = st.columns(3)
        rho = c4.number_input("Inner fillet radius ρ (mm)", min_value=0.001, value=2.0, format="%g", key="sg_rho")
        Fx = c5.number_input("Fx at the free end (N)", value=0.0, format="%g", key="sg_Fx")
        Fy = c6.number_input("Fy at the free end (N)", value=-1000.0, format="%g", key="sg_Fy")
        a, b = h1 / L, h2 / L
        p = np.array([a, b, (rho / L) / min(a, b)])
        desc = (f"{L:g} × {L:g} mm L-bracket, arms {h1:g} / {h2:g} mm, inner fillet {rho:g} mm; top edge clamped, "
                f"load on the right end")
        to_p = lambda rr: np.array([a, b, (rr / L) / min(a, b)])
        rho_of = lambda q: q[2] * min(q[0], q[1]) * L
    else:
        d = c1.number_input("Narrow width d (mm)", min_value=0.01, value=20.0, format="%g", key="sg_d")
        D = c2.number_input("Wide width D (mm)", min_value=0.01, value=40.0, format="%g", key="sg_D")
        rho = c3.number_input("Shoulder fillet radius r (mm)", min_value=0.001, value=2.0, format="%g", key="sg_r")
        c4, c5 = st.columns(2)
        Fx = c4.number_input("Axial force Fx (N)", value=10000.0, format="%g", key="sg_Fx2")
        Fy = c5.number_input("Transverse force Fy at the end (N)", value=0.0, format="%g", key="sg_Fy2")
        L = d
        p = np.array([D / d, rho / d])
        desc = (f"Flat bar {D:g} → {d:g} mm with shoulder fillets r = {rho:g} mm; wide part {2 * d:g} mm long "
                f"(clamped end), narrow part {3 * d:g} mm long (loaded end)")
        to_p = lambda rr: np.array([D / d, rr / d])
        rho_of = lambda q: q[1] * d
    return p, L, rho, Fx, Fy, desc, to_p, rho_of


def _range_problems(fam, p):
    msgs = []
    for k, (nm, lo, hi) in enumerate(zip(fam.pnames, fam.lo, fam.hi)):
        if not (lo * 0.999 <= p[k] <= hi * 1.001):
            label = {"a": "horizontal arm / L", "b": "vertical arm / L", "rho_rel": "fillet / thinner arm",
                     "D": "D / d", "r": "r / d"}[nm]
            msgs.append(f"{label} = {p[k]:.3g} is outside the trained range {lo:g}–{hi:g}")
    if not fam.valid(p):
        msgs.append("the fillet does not fit on the shoulder (r must be below 0.95 × (D − d)/2)")
    return msgs


def stress_figure(grid, ext, peak_xy, peak, unit="MPa", height=520):
    sx, sy, txy = grid["sx"], grid["sy"], grid["txy"]
    Z = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
    ny, nx = Z.shape
    gx, gy = np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny)
    hi = float(np.nanpercentile(Z, 99.5))
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=Z, zmin=0, zmax=hi, colorscale=JET, hoverongaps=False,
                               colorbar=dict(title=dict(text=unit, font=dict(family=MONO, color="#8c8c8c", size=11)),
                                             tickfont=dict(family=MONO, color="#8c8c8c", size=10), outlinewidth=0,
                                             thickness=14),
                               hovertemplate="x %{x:.2f} · y %{y:.2f}<br>%{z:.4g} MPa<extra></extra>"))
    if peak_xy is not None:
        fig.add_trace(go.Scatter(x=[peak_xy[0]], y=[peak_xy[1]], mode="markers+text", text=[f"PEAK {peak:.4g}"],
                                 textposition="top right", textfont=dict(family=MONO, color=FG, size=11),
                                 marker=dict(symbol="triangle-up", size=13, color=BG, line=dict(color=FG, width=2)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                      font=dict(family=MONO, color="#8c8c8c", size=11), dragmode="pan",
                      hoverlabel=dict(bgcolor="#171717", bordercolor=FG, font=dict(family=MONO, color=FG)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, title="X (MM)")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=BORDER, scaleanchor="x", scaleratio=1, title="Y (MM)")
    return fig


def _peak_location(fam, p, L, res):
    from fe2d import fillet_points
    poly = fam.outline(p)
    i, r = sorted(fam.fillets(p).items())[res["peak_fillet"]]
    n = len(poly)
    t1, c, t2 = fillet_points(poly[i - 1], poly[i], poly[(i + 1) % n], r)
    a1, a2 = np.arctan2(*(t1 - c)[::-1]), np.arctan2(*(t2 - c)[::-1])
    span = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    ang = a1 + res["peak_s"] * span
    return (c + r * np.array([np.cos(ang), np.sin(ang)])) * L, np.degrees(span), (c * L, r * L, a1, span)


def surrogate_page(t, E, sy, sf, su, se, cycle):
    st.markdown('<p class="eyebrow">Parametric surrogate · no FE at run time</p>', unsafe_allow_html=True)
    st.markdown("### Pick a part family, type the dimensions, get the answer instantly")
    fam_key = st.radio("Part family", list(FAMILIES), horizontal=True, key="sg_fam",
                       format_func=lambda k: FAMILIES[k].name)
    fam = FAMILIES[fam_key]
    try:
        sur = load_surrogate(fam_key)
    except FileNotFoundError:
        st.error(f"surrogate_{fam_key}.npz is missing: run python train_surrogate.py")
        return
    p, L, rho, Fx, Fy, desc, to_p, rho_of = _inputs(fam)
    st.caption(desc + f" · t = {t:g} mm · E = {E:g} MPa · ν = 0.3 (built into the surrogate)")
    bad = _range_problems(fam, p)
    if bad:
        st.error("Outside what the surrogate was trained on: " + "; ".join(bad) +
                 ". Use the 'Any drawing (FE)' mode for this part.")
        return
    if Fx == 0 and Fy == 0:
        st.info("Enter a force.")
        return

    t0 = time.perf_counter()
    res = predict(sur, p, L, t, E, Fx, Fy)
    X, Y, inside, fields, tri = physical_fields(sur, p, L, t, E, Fx, Fy)
    ms = 1000 * (time.perf_counter() - t0)
    allow = sy / sf
    peak = res["peak"]

    st.markdown(f'<div class="note">SURROGATE ANSWER IN {ms:.0f} MS · POD-RBF reduced-order model, '
                f'{sur.n_modes} modes, trained on {len(sur._P)} offline FE runs</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak stress", f"{peak:.0f} MPa",
              help="At the fillet surface (the stress there is uniaxial, so this is also von Mises). Built from "
                   "the unit-load fillet profiles: exact superposition of Fx and Fy.")
    m2.metric("Kt", f"{res['Kt']:.2f}", help="Peak / nominal. Nominal: " +
              ("beam bending + axial stress in the horizontal arm at the joint face." if fam_key == "lbracket"
               else "Peterson's net-section stress in the narrow part, P/(d t) + 6M/(t d²)."))
    u = res["u_load"]
    m3.metric("Load-point deflection", f"{np.hypot(*u):.3g} mm",
              help=f"Mean displacement of the loaded edge: ux = {u[0]:.3g} mm, uy = {u[1]:.3g} mm.")
    m4.metric("Safety factor", f"{sy / peak:.2f}", help=f"σy / peak, σy = {sy:g} MPa. Target {sf:g}.")

    # required radius: scan the fillet instantly
    from surrogate import required_radius
    k = 2 if fam_key == "lbracket" else 1
    rr_rel, rr_p = required_radius(sur, lambda v: np.r_[p[:k], v], L, t, Fx, Fy, allow)
    if rr_rel is None:
        rr_txt = f"no fillet in the trained range reaches {allow:.0f} MPa: add material or reduce the load"
    else:
        rr_txt = f"{rho_of(rr_p):.3g} mm"
    fat = fatigue(peak, res["nominal"], rho, su, se, cycle)
    st.markdown(f'<div class="card"><div class="ml">REQUIRED FILLET for ≤ {allow:.0f} MPa (σy / {sf:g}) → '
                f'<b>{rr_txt}</b></div>'
                f'<div class="ml">STRAIN ENERGY {res["energy"]:.4g} N·mm · nominal stress {res["nominal"]:.3g} MPa'
                f'</div><div class="ml">FATIGUE ({"±F" if cycle == "R=-1" else "0→F"}) · q = {fat["q"]:.2f}, '
                f'Kf = {fat["Kf"]:.2f} · life {life_text(fat["life_cycles"])} cycles</div></div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["Stress map", "Deformation", "Photoelastic", "Design sweep", "Check against FE", "Accuracy"])
    peak_xy, span_deg, arcinfo = _peak_location(fam, p, L, res)
    grid, ext = to_image(X, Y, tri, fields, n_px=600)
    with tabs[0]:
        st.plotly_chart(stress_figure(grid, ext, peak_xy, peak), width="stretch",
                        config={"displaylogo": False, "scrollZoom": True})
        st.caption("Von Mises field from the POD-RBF model (Fx and Fy fields superposed). The colour scale stops "
                   "at the 99.5th percentile; the field grid is coarser than a small fillet, so the exact peak "
                   "comes from the separate fillet-profile model (marked ▲).")
        prof = res["arc_profile"][res["peak_fillet"]]
        ang = np.linspace(0, span_deg, len(prof))
        fig = go.Figure(go.Scatter(x=np.abs(ang), y=prof, mode="lines", line=dict(color=ACCENT, width=2)))
        fig.add_hline(y=allow, line=dict(color=FG, dash="dot", width=1))
        fig.add_hline(y=-allow, line=dict(color=FG, dash="dot", width=1))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                          font=dict(family=MONO, color="#8c8c8c", size=11),
                          title=dict(text="STRESS ALONG THE FILLET (tangent point → tangent point)", font=dict(size=11)))
        fig.update_xaxes(title="angle along the arc (deg)", showgrid=False, linecolor=BORDER)
        fig.update_yaxes(title="σ tangential (MPa)", gridcolor="#1f1f1f", linecolor=BORDER)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with tabs[1]:
        sc_auto = auto_exaggeration(np.c_[X.ravel(), Y.ravel()], np.hypot(fields["ux"], fields["uy"]))
        mult = st.number_input("Exaggeration (× auto)", min_value=0.0, value=1.0, step=0.5, key="sg_mult")
        xy = np.c_[X.ravel(), Y.ravel()]
        poly = np.vstack(fam.outline(p)) * L
        st.pyplot(deformed_figure(xy, tri, fields["ux"].ravel(), fields["uy"].ravel(), sc_auto * mult,
                                  [np.vstack([poly, poly[:1]])], False, (),
                                  title=f"DEFORMED SHAPE · ×{sc_auto * mult:.3g}"), width="stretch")
        st.caption(f"Load-point displacement ux = {u[0]:.4g} mm, uy = {u[1]:.4g} mm; strain energy "
                   f"U = ½ F·C·F = {res['energy']:.4g} N·mm. Deformation is the best-predicted quantity: "
                   "it is finite and smooth even where the stress concentrates.")
    with tabs[2]:
        opts = photo_controls("sgph", t)
        ny, nx = grid["sx"].shape
        GX, GY = np.meshgrid(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny))
        c, r, _, _ = arcinfo
        ok = np.hypot(GX - c[0], GY - c[1]) > 3 * r          # set the load away from the fillet
        lf = photo_panel(grid, ext, ok, opts, float(np.hypot(Fx, Fy)), [(peak_xy[0], peak_xy[1], "FILLET")])
        st.markdown(f"Fringe order at the fillet root with the real radius: **N = {t * peak * lf / opts['f']:.1f}** "
                    f"(t σ_peak / f_σ). A photoelastic model of this part, loaded to {lf * np.hypot(Fx, Fy):.3g} N, "
                    "would show this many fringes packed into the fillet.")
    with tabs[3]:
        st.markdown("**Peak stress against fillet radius** · each point is a full answer from the surrogate")
        lo = fam.lo[k] if fam_key == "lbracket" else fam.lo[1]
        hi = fam.hi[k] if fam_key == "lbracket" else fam.hi[1]
        vals = np.geomspace(lo, hi, 60)
        t1 = time.perf_counter()
        rows = []
        for v in vals:
            q = np.r_[p[:k], v]
            if fam.valid(q):
                rr = predict(sur, q, L, t, E, Fx, Fy)
                rows.append((rho_of(q), rr["peak"], np.hypot(*rr["u_load"])))
        sweep_ms = 1000 * (time.perf_counter() - t1)
        rows = np.array(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rows[:, 0], y=rows[:, 1], mode="lines", line=dict(color=ACCENT, width=2),
                                 name="peak stress"))
        # reference lines as traces: plotly shapes on log axes would need log10 coordinates
        fig.add_trace(go.Scatter(x=[rows[0, 0], rows[-1, 0]], y=[allow, allow], mode="lines+text",
                                 line=dict(color=FG, dash="dot", width=1), text=["", f"allowable {allow:.0f} MPa"],
                                 textposition="top left", textfont=dict(family=MONO, color=FG, size=11)))
        fig.add_trace(go.Scatter(x=[rho, rho], y=[rows[:, 1].min(), rows[:, 1].max()], mode="lines",
                                 line=dict(color=MUTED, dash="dot", width=1)))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=BG, paper_bgcolor=BG,
                          font=dict(family=MONO, color="#8c8c8c", size=11), showlegend=False)
        fig.update_xaxes(type="log", title="fillet radius (mm)", showgrid=False, linecolor=BORDER)
        fig.update_yaxes(type="log", title="peak stress (MPa)", gridcolor="#1f1f1f", linecolor=BORDER)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        slope = np.polyfit(np.log(rows[:8, 0]), np.log(rows[:8, 1]), 1)[0] if len(rows) > 8 else np.nan
        st.caption(f"{len(rows)} designs in {sweep_ms:.0f} ms. For small fillets the curve has slope "
                   f"{slope:.2f} on log axes; Williams' theory for a 270° corner gives λ − 1 = {LAM_270 - 1:.3f}. "
                   "The stiffness hardly changes across the sweep: the fillet sets the strength, not the "
                   "deflection.")
    with tabs[4]:
        st.markdown("Run one real FE solve of the same part and compare. This is the check, not the method.")
        if st.button("Run FE check", key="sg_fe"):
            t2 = time.perf_counter()
            m, sols = fe_solve(fam, p)
            fe_ms = 1000 * (time.perf_counter() - t2)
            arcs = np.array([[pr["Fx"], pr["Fy"]] for pr in arc_profiles(fam, p, m, sols)])
            comb = Fx * arcs[:, 0] + Fy * arcs[:, 1]
            fe_peak = float(np.abs(comb).max()) / (t * L)
            C = compliance(fam, p, m, sols)
            fe_u = C @ np.array([Fx, Fy]) / (t * E)
            fe_U = 0.5 * np.array([Fx, Fy]) @ C @ np.array([Fx, Fy]) / (t * E)
            tab = pd.DataFrame({
                "": ["peak stress (MPa)", "load-point deflection (mm)", "strain energy (N·mm)", "time (ms)"],
                "surrogate": [f"{peak:.4g}", f"{np.hypot(*u):.4g}", f"{res['energy']:.4g}", f"{ms:.0f}"],
                "FE": [f"{fe_peak:.4g}", f"{np.hypot(*fe_u):.4g}", f"{fe_U:.4g}", f"{fe_ms:.0f}"],
                "difference": [f"{100 * (peak / fe_peak - 1):+.2f} %",
                               f"{100 * (np.hypot(*u) / np.hypot(*fe_u) - 1):+.2f} %",
                               f"{100 * (res['energy'] / fe_U - 1):+.2f} %", f"×{fe_ms / max(ms, 1e-3):.0f}"]})
            st.dataframe(tab, hide_index=True, width="stretch")
            st.caption(f"FE: {m.p.shape[1]:,} nodes, 6-node triangles, fillet refined to r/8, two unit load cases. "
                       "The surrogate never saw this exact part.")
    with tabs[5]:
        val = validation_table()
        if val is None:
            st.info("No validation file.")
        else:
            v = val[val.family == fam_key]
            cols = {"peak_Fx_err_%": "peak stress, Fx", "peak_Fy_err_%": "peak stress, Fy",
                    "defl_err_%": "deflection", "field_stress_relL2_%": "stress field (rel. L2)",
                    "field_disp_relL2_%": "displacement field (rel. L2)"}
            rows = [{"quantity": nm, "mean |error|": f"{v[c].abs().mean():.2f} %",
                     "median": f"{v[c].abs().median():.2f} %", "worst": f"{v[c].abs().max():.2f} %"}
                    for c, nm in cols.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"{len(v)} held-out FE runs with random dimensions the surrogate was not trained on. "
                       "The largest peak errors are at the smallest fillets with the thinnest arms.")
