"""
views.py - the Deformation and Photoelastic panels, shared by the FE mode and the surrogate mode.
Both panels take plain arrays, so they do not care where the fields came from.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import streamlit as st

from deformation import fringe_rgb, PHOTO_MATERIALS

BG, FG, MUTED, ACCENT = "#0e0e0e", "#f2f2f2", "#5a5a5a", "#e8ff5a"
MONO = ["Courier New", "Courier", "DejaVu Sans Mono", "monospace"]


def _ax(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def auto_exaggeration(xy, umag):
    size = np.ptp(xy, axis=0).max()
    return 0.08 * size / max(float(np.nanmax(umag)), 1e-30)


def deformed_figure(xy, tri, ux, uy, scale, outline_segs=(), show_mesh=False, marks=(), title=""):
    """Deformed shape, coloured by |u| (mm). outline_segs: list of (N,2) arrays drawn undeformed, dashed."""
    umag = np.hypot(ux, uy)
    fig, ax = _ax()
    for seg in outline_segs:
        ax.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=1, ls=(0, (4, 3)))
    X, Y = xy[:, 0] + scale * ux, xy[:, 1] + scale * uy
    tpc = ax.tripcolor(X, Y, tri, umag, shading="gouraud", cmap="viridis")
    if show_mesh:
        ax.triplot(X, Y, tri, color="#000000", lw=0.15, alpha=0.5)
    for (x, y, lab) in marks:
        ax.plot(x, y, "o", ms=11, mfc="none", mec=ACCENT, mew=1.6)
        ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points", color=FG, fontsize=8, family=MONO)
    j = int(np.argmax(umag))
    ax.plot(X[j], Y[j], "^", ms=10, mfc=BG, mec=FG, mew=1.6)
    ax.annotate(f"MAX {umag[j]:.4g} mm", (X[j], Y[j]), xytext=(8, -14), textcoords="offset points",
                color=FG, fontsize=9, family=MONO)
    cb = fig.colorbar(tpc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|u| (mm)", color="#8c8c8c", family=MONO, fontsize=9)
    cb.ax.tick_params(colors="#8c8c8c", labelsize=8)
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title, color="#8c8c8c", fontsize=9, family=MONO, loc="left")
    fig.tight_layout()
    return fig


def fringe_figure(rgb, extent, marks=(), title=""):
    fig, ax = _ax()
    img = np.where(np.isfinite(rgb), rgb, np.array([0.055, 0.055, 0.055]))
    ax.imshow(img, origin="lower", extent=extent, interpolation="bilinear")
    for (x, y, lab) in marks:
        ax.annotate(lab, (x, y), xytext=(10, 10), textcoords="offset points", color=FG, fontsize=8, family=MONO,
                    bbox=dict(boxstyle="square,pad=0.25", fc=BG, ec="#2a2a2a"))
    if title:
        ax.set_title(title, color="#8c8c8c", fontsize=9, family=MONO, loc="left")
    fig.tight_layout()
    return fig


def photo_controls(key, t_default):
    """Sidebar-free controls for the photoelastic panel. Returns a settings dict."""
    c1, c2 = st.columns(2)
    mat = c1.selectbox("Model material", list(PHOTO_MATERIALS), key=f"{key}_mat",
                       help="Fringe constant f_σ (N/mm per fringe) at ~560 nm. Polycarbonate is the usual "
                            "teaching material.")
    light = c2.selectbox("Light", ["White light (colour)", "Monochromatic (sodium)"], key=f"{key}_light")
    c3, c4, c5 = st.columns(3)
    pol = c3.selectbox("Polariscope", ["circular", "plane"], key=f"{key}_pol",
                       help="Circular: isochromatics only (σ1 − σ2). Plane: isoclinics added — dark bands "
                            "where a principal direction lines up with the polariser.")
    field = c4.selectbox("Field", ["dark", "light"], key=f"{key}_field",
                         help="Dark field: zero stress is black, fringes at N = 1, 2, 3 … "
                              "Light field: zero stress is bright, half-order fringes.")
    nmax = c5.number_input("Fringe order N (98th percentile)", min_value=0.5, max_value=40.0,
                           value=6.0, step=0.5, key=f"{key}_nmax",
                           help="Sets the load on the photoelastic model. A real test loads a plastic copy "
                                "of the part until a readable number of fringes appears.")
    ang = 0.0
    if pol == "plane":
        ang = st.slider("Polariser angle (deg)", 0, 90, 0, 5, key=f"{key}_ang")
    t_model = st.number_input("Model thickness (mm)", min_value=0.5, value=float(t_default), key=f"{key}_t",
                              help="Fringe order N = t (σ1 − σ2) / f_σ. For the same load per unit thickness, "
                                   "the stresses scale 1/t, so N depends on the load, not the thickness.")
    return {"f": PHOTO_MATERIALS[mat], "white": light.startswith("White"), "pol": pol,
            "dark": field == "dark", "nmax": nmax, "ang": ang, "t": t_model, "mat": mat}


def photo_panel(grid, extent, ok_mask, opts, force_total, marks=(), corner_rows=(), title=""):
    """grid: dict sx, sy, txy on the image grid (real stresses, MPa). ok_mask: grid cells away from singular
    points (used to set the load). force_total: |F| of the real load (N), only to report the model load."""
    from deformation import principal
    s1, s2, _ = principal(np.nan_to_num(grid["sx"]), np.nan_to_num(grid["sy"]), np.nan_to_num(grid["txy"]))
    d = np.where(np.isfinite(grid["sx"]) & ok_mask, s1 - s2, np.nan)
    # 98th percentile, not the max: a clamp or load patch should not black out the rest of the part
    dmax = np.nanpercentile(d, 98) if np.isfinite(d).any() else np.nanmax(s1 - s2)
    lf = opts["nmax"] * opts["f"] / (opts["t"] * max(dmax, 1e-12))
    rgb, N = fringe_rgb(grid["sx"], grid["sy"], grid["txy"], opts["t"], opts["f"], lf, white=opts["white"],
                        polariscope=opts["pol"], dark_field=opts["dark"], analyser_deg=opts["ang"])
    st.pyplot(fringe_figure(rgb, extent, marks, title), width="stretch")
    model_load = lf * force_total
    st.caption(f"Stress-optic law N = t (σ1 − σ2) / f_σ · {opts['mat']}, f_σ = {opts['f']:g} N/mm/fringe, "
               f"t = {opts['t']:g} mm · the plastic model carries {lf:.3g} × the real load"
               + (f" ≈ {model_load:.3g} N" if force_total else "") +
               ". Fringes crowd into a sharp corner without limit; at a real fillet they stop at a finite order.")
    if len(corner_rows):
        st.dataframe(pd.DataFrame(corner_rows), hide_index=True, width="stretch")
    return lf


def outline_segments(rings):
    segs = []
    for r in rings:
        r = np.asarray(r, float)
        segs.append(np.vstack([r, r[:1]]))
    return segs
