"""make_figures.py - report figures + unseen-part validation table."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fe2d import mesh_polygon, solve_plane_stress, interior_angles
from corner_tools import corner_features, sampler_from_mesh, corner_length, MAX_H_OVER_ELL
from sing_tool import load_part, solve_two_meshes, analyse, predict
from test_parts import PARTS, ground_truth

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_ML, C_FE, C_PHYS = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
model = joblib.load("singularity_model.joblib")


# ---------------------------------------------------------------- 1. money plot
def money_plot():
    a, L, F, rho = 20.0, 100.0, 1000.0, 1.0
    poly = [np.array(p, float) for p in [(0, 0), (L, 0), (L, a), (a, a), (a, L), (0, L)]]
    fixed, trac = [((0, L), (a, L))], [((L, 0), (L, a), (0.0, -F / a))]
    ell = corner_length(poly, 3)
    m = mesh_polygon(poly, fillets={3: rho}, refine=[(3, rho / 16)], h_global=a / 4)
    s = solve_plane_stress(m, fixed, trac)
    truth = s["vm"][np.linalg.norm(m.p.T - poly[3], axis=1) < 3 * rho * 1.5].max()
    divs = np.array([5, 10, 20, 40, 80, 160, 320])
    fe, ml, n_nodes = [], [], []
    for k in divs:
        h = ell / k
        m = mesh_polygon(poly, refine=[(3, h)], h_global=a / 4)
        s = solve_plane_stress(m, fixed, trac)
        fe.append(s["vm"][np.argmin(np.linalg.norm(m.p.T - poly[3], axis=1))])
        f = corner_features(sampler_from_mesh(m, s), poly, 3, h)
        ml.append(predict(model, f, rho) if f is not None else np.nan)
        n_nodes.append(m.p.shape[1])
    h_mm = ell / divs
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(truth, color=INK, ls="--", lw=1.2)
    ax.text(h_mm[0], truth * 0.94, f"  true peak, {rho:g} mm fillet: {truth:.0f} MPa",
            ha="left", va="top", color=INK, fontsize=9)
    ax.plot(h_mm, fe, "-o", color=C_FE, lw=2, ms=7, mec="white", mew=1.5, label="FE stress at sharp corner node")
    ax.plot(h_mm, ml, "-o", color=C_ML, lw=2, ms=7, mec="white", mew=1.5, label="Model prediction (same meshes)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Element size at the corner, h (mm)  →  finer mesh")
    ax.set_ylabel("von Mises stress (MPa)")
    ax.set_title("FE corner stress diverges; the prediction does not")
    ax.annotate(f"{fe[-1]:.0f} MPa and still rising", (h_mm[-1], fe[-1]), xytext=(-12, 0),
                textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0, 0.62))
    ax.set_ylim(0, max(fe) * 1.12)
    fig.tight_layout()
    fig.savefig("fig1_mesh_divergence.png")
    pd.DataFrame({"h_mm": h_mm, "nodes": n_nodes, "fe_corner_node": fe, "model": ml, "truth": truth}) \
        .to_csv("fig1_data.csv", index=False)
    print("fig1", np.round(fe, 0), np.round(ml, 0), round(truth, 0))


# ---------------------------------------------------------------- 2. parity + 3. error by mesh
def test_set_plots():
    t = pd.read_csv("test_predictions.csv")
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lo, hi = t.target_peak.min() * 0.7, t.target_peak.max() * 1.4
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.scatter(t.target_peak, t.vm_corner_node, s=9, color=C_FE, alpha=0.35, lw=0, label="FE corner-node stress")
    ax.scatter(t.target_peak, t.pred_ml, s=9, color=C_ML, alpha=0.5, lw=0, label="Model prediction")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True peak stress, filleted corner (MPa)")
    ax.set_ylabel("Reported / predicted stress (MPa)")
    ax.set_title(f"Unseen test geometries (n = {len(t)})")
    ax.legend(frameon=False, loc="upper left", markerscale=2)
    fig.tight_layout(); fig.savefig("fig2_parity.png")

    t["err_fe"] = 100 * (t.vm_corner_node - t.target_peak).abs() / t.target_peak
    t["err_ml"] = 100 * (t.pred_ml - t.target_peak).abs() / t.target_peak
    g = t.groupby(t.h_over_ell.round(4))[["err_fe", "err_ml"]].mean().sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(g)); w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, g.err_fe, w, color=C_FE, label="FE corner-node stress")
    b2 = ax.bar(x + w / 2 + 0.01, g.err_ml, w, color=C_ML, label="Model prediction")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"l/{round(1 / v)}" for v in g.index])
    ax.set_xlabel("Element size at the corner (l = local feature size)  →  finer mesh")
    ax.set_ylabel("Mean |error| vs true peak (%)")
    ax.set_title("Error does not depend on the mesh")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig("fig3_error_by_mesh.png")


# ---------------------------------------------------------------- 4. unseen parts
def unseen_parts():
    rows, figs = [], []
    for key, spec in PARTS.items():
        part = load_part(f"part_{key}.json")
        sols = solve_two_meshes(part)
        out = analyse(part, sols, model)
        corners = out[out.type == "sharp re-entrant corner"]
        truth = ground_truth(spec, list(corners.vertex), spec["corner_radius"])
        for _, r in corners.iterrows():
            tv = truth[r.vertex]
            rows.append({"part": spec["name"], "corner": int(r.vertex), "angle_deg": r.angle_deg,
                         "radius_mm": r.corner_radius, "true_peak_MPa": tv,
                         "FE_coarse_MPa": r.fe_stress_coarse, "FE_fine_MPa": r.fe_stress_fine,
                         "model_MPa": r.predicted_real_stress,
                         "FE_fine_err_%": 100 * (r.fe_stress_fine - tv) / tv,
                         "model_err_%": 100 * (r.predicted_real_stress - tv) / tv})
        figs.append((spec, sols["fine"], out))
    res = pd.DataFrame(rows)
    res.to_csv("unseen_parts_results.csv", index=False)
    print(res.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (spec, f, out) in zip(axes, figs):
        m, vm = f["m"], f["vm"]
        cap = np.percentile(vm, 97)
        tc = ax.tripcolor(m.p[0], m.p[1], m.t.T, np.minimum(vm, cap), shading="gouraud", cmap="Blues")
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(spec["name"], fontsize=10)
        for _, r in out[out.singular].iterrows():
            geo = r.type == "sharp re-entrant corner"
            ax.plot(r.x, r.y, "o", ms=11, mfc="none", mec=C_FE if geo else INK2, mew=2.2)
            if geo and np.isfinite(r.get("predicted_real_stress", np.nan)):
                ax.annotate(f"{r.predicted_real_stress:.0f} MPa\n(r = {r.corner_radius:g} mm)", (r.x, r.y),
                            xytext=(8, 8), textcoords="offset points", fontsize=8.5, color=INK,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
    axes[0].plot([], [], "o", ms=10, mfc="none", mec=C_FE, mew=2, label="Sharp corner → real stress predicted")
    axes[0].plot([], [], "o", ms=10, mfc="none", mec=INK2, mew=2, label="Support / load artefact (ignored)")
    fig.legend(loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Tool output on parts never seen in training (von Mises, capped at 97th percentile)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig("fig4_tool_output.png")


if __name__ == "__main__":
    import sys
    if "--fig1" in sys.argv:
        money_plot()
    else:
        money_plot()
        test_set_plots()
        unseen_parts()
