"""
validate_reference.py - check EigenStress against independent textbook solutions (nothing here is used for training).

  A. Plate with a circular hole in tension  vs  Kirsch exact elasticity solution (full contour + profiles)
  B. End-loaded cantilever                  vs  beam theory  sigma_x = M y / I,  tau_xy parabolic
  C. Stepped flat bar, opposite shoulder fillets, tension  vs  Peterson's Kt (Pilkey curve fits)
       - filleted FE peaks (solver accuracy)
       - EigenStress prediction from the SHARP model (ML accuracy)

Outputs: fig5_kirsch.png, fig6_beam.png, fig7_peterson.png, validation_reference.csv
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation, LinearTriInterpolator

from fe2d import mesh_polygon, solve_plane_stress
from sing_tool import solve_two_meshes, analyse

warnings.filterwarnings("ignore")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_ML, C_FE, C_REF = "#2a78d6", "#eb6834", "#0b0b0b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
rows = []


def interp(m, s, key):
    return LinearTriInterpolator(Triangulation(m.p[0], m.p[1], m.t.T), s[key])


# ------------------------------------------------------------------ A. Kirsch
def kirsch(x, y, a, S):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    q = (a / r) ** 2
    srr = S / 2 * (1 - q) + S / 2 * (1 - 4 * q + 3 * q ** 2) * np.cos(2 * th)
    stt = S / 2 * (1 + q) - S / 2 * (1 + 3 * q ** 2) * np.cos(2 * th)
    srt = -S / 2 * (1 + 2 * q - 3 * q ** 2) * np.sin(2 * th)
    c, s_ = np.cos(th), np.sin(th)
    sx = srr * c ** 2 + stt * s_ ** 2 - 2 * srt * s_ * c
    sy = srr * s_ ** 2 + stt * c ** 2 + 2 * srt * s_ * c
    txy = (srr - stt) * s_ * c + srt * (c ** 2 - s_ ** 2)
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
    return sx, sy, txy, vm


def case_kirsch():
    W, a, S = 400.0, 5.0, 100.0                       # d/W = 0.025 -> finite-width effect < 0.2 %
    poly = [(-W / 2, -W / 2), (W / 2, -W / 2), (W / 2, W / 2), (-W / 2, W / 2)]
    m = mesh_polygon(poly, circles=[((0.0, 0.0), a, None, 2 * np.pi * a / 128)], h_global=W / 25)
    s = solve_plane_stress(m, [((-W / 2, W / 2), (-W / 2, -W / 2))], [((W / 2, -W / 2), (W / 2, W / 2), (S, 0.0))])
    fx, fvm = interp(m, s, "sx"), interp(m, s, "vm")

    # 1) stress around the hole edge (sampled just inside the material)
    th = np.radians(np.linspace(0, 180, 181))
    rr = a * 1.002
    # hoop stress on the edge from FE: sigma_tt = sx sin^2 + sy cos^2 - 2 txy s c
    fy, ft = interp(m, s, "sy"), interp(m, s, "txy")
    X, Y = rr * np.cos(th), rr * np.sin(th)
    stt_fe = fx(X, Y) * np.sin(th) ** 2 + fy(X, Y) * np.cos(th) ** 2 - 2 * ft(X, Y) * np.sin(th) * np.cos(th)
    stt_ex = S * (1 - 2 * np.cos(2 * th))
    # 2) sigma_x along the section through the hole (x = 0, y from a to 6a)
    yy = np.linspace(a * 1.002, 6 * a, 120)
    sx_fe = np.asarray(fx(np.zeros_like(yy), yy))
    sx_ex = S * (1 + 0.5 * (a / yy) ** 2 + 1.5 * (a / yy) ** 4)
    # 3) full field error in the window |x|,|y| <= 6a
    g = np.linspace(-6 * a, 6 * a, 241)
    GX, GY = np.meshgrid(g, g)
    inside = np.hypot(GX, GY) > a * 1.01
    vm_fe = np.where(inside, np.ma.filled(fvm(GX, GY), np.nan), np.nan)
    vm_ex = np.where(inside, kirsch(GX, GY, a, S)[3], np.nan)
    err = 100 * (vm_fe - vm_ex) / S                    # % of applied stress

    peak_fe, peak_ex = float(np.nanmax(stt_fe)), 3 * S
    rows.append({"case": "A Kirsch plate with hole", "quantity": "peak hoop stress at hole (MPa)",
                 "reference": peak_ex, "EigenStress": peak_fe, "error_%": 100 * (peak_fe - peak_ex) / peak_ex})
    rows.append({"case": "A Kirsch plate with hole", "quantity": "sigma_x section x=0, RMS error (% of S)",
                 "reference": 0.0, "EigenStress": float(np.sqrt(np.mean(((sx_fe - sx_ex) / S * 100) ** 2))),
                 "error_%": float(np.sqrt(np.mean(((sx_fe - sx_ex) / S * 100) ** 2)))})
    rows.append({"case": "A Kirsch plate with hole", "quantity": "von Mises field |x|,|y|<=6a, max abs error (% of S)",
                 "reference": 0.0, "EigenStress": float(np.nanmax(np.abs(err))), "error_%": float(np.nanmax(np.abs(err)))})

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    lv = np.linspace(0, 3.1 * S, 16)
    for k, (Z, ttl) in enumerate([(vm_fe, "EigenStress FE: von Mises"), (vm_ex, "Kirsch exact: von Mises")]):
        cs = ax[k].contourf(GX / a, GY / a, Z, levels=lv, cmap="viridis")
        ax[k].set_aspect("equal"); ax[k].set_title(ttl); ax[k].set_xlabel("x / a"); ax[k].set_ylabel("y / a")
        ax[k].grid(False)
    fig.colorbar(cs, ax=ax[1], shrink=0.85, label="MPa")
    ce = ax[2].contourf(GX / a, GY / a, err, levels=np.linspace(-3, 3, 13), cmap="RdBu_r")
    ax[2].set_aspect("equal"); ax[2].set_title("Difference (% of applied stress)"); ax[2].grid(False)
    ax[2].set_xlabel("x / a")
    fig.colorbar(ce, ax=ax[2], shrink=0.85, label="%")
    ax[3].plot(np.degrees(th), stt_ex, color=C_REF, lw=1.5, ls="--", label="Kirsch exact")
    ax[3].plot(np.degrees(th)[::6], stt_fe[::6], "o", ms=5, color=C_ML, mec="white", mew=1, label="EigenStress FE")
    ax[3].set_xlabel("Angle around hole from load axis (deg)"); ax[3].set_ylabel("Hoop stress at hole edge (MPa)")
    ax[3].set_title(f"Peak: FE {peak_fe:.1f} vs exact {peak_ex:.0f} MPa"); ax[3].legend(frameon=False)
    fig.suptitle("A · Plate with a hole in tension (100 MPa): FE contours vs Kirsch exact solution", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig5_kirsch.png"); plt.close(fig)
    print(f"A  peak hoop {peak_fe:.2f} vs 300 | section RMS {rows[-2]['error_%']:.2f}% | field max {rows[-1]['error_%']:.2f}%")


# ------------------------------------------------------------------ B. Cantilever
def case_beam():
    L, h, P = 200.0, 20.0, 50.0                       # P per unit thickness (N/mm); tip shear load
    poly = [(0, -h / 2), (L, -h / 2), (L, h / 2), (0, h / 2)]
    m = mesh_polygon(poly, h_global=h / 10)
    s = solve_plane_stress(m, [((0, h / 2), (0, -h / 2))], [((L, -h / 2), (L, h / 2), (0.0, -P / h))])
    fx, ft = interp(m, s, "sx"), interp(m, s, "txy")
    I = h ** 3 / 12
    out = {}
    for xs in (50.0, 100.0, 150.0):
        y = np.linspace(-h / 2 * 0.999, h / 2 * 0.999, 41)
        sx_fe = np.asarray(fx(np.full_like(y, xs), y))
        sx_ex = P * (L - xs) * y / I
        tx_fe = np.asarray(ft(np.full_like(y, xs), y))
        tx_ex = -1.5 * P / h * (1 - 4 * y ** 2 / h ** 2)
        out[xs] = (y, sx_fe, sx_ex, tx_fe, tx_ex)
    y, sx_fe, sx_ex, tx_fe, tx_ex = out[100.0]
    e_s = 100 * np.max(np.abs(sx_fe - sx_ex)) / np.max(np.abs(sx_ex))
    e_t = 100 * np.max(np.abs(tx_fe - tx_ex)) / np.max(np.abs(tx_ex))
    rows.append({"case": "B Cantilever, tip load", "quantity": "max bending stress at mid-span (MPa)",
                 "reference": float(np.max(sx_ex)), "EigenStress": float(np.max(sx_fe)),
                 "error_%": 100 * (np.max(sx_fe) - np.max(sx_ex)) / np.max(sx_ex)})
    rows.append({"case": "B Cantilever, tip load", "quantity": "sigma_x profile, max error (% of peak)",
                 "reference": 0.0, "EigenStress": e_s, "error_%": e_s})
    rows.append({"case": "B Cantilever, tip load", "quantity": "tau_xy parabola, max error (% of peak)",
                 "reference": 0.0, "EigenStress": e_t, "error_%": e_t})

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    gx, gy = np.linspace(0, L, 401), np.linspace(-h / 2, h / 2, 41)
    GX, GY = np.meshgrid(gx, gy)
    cs = ax[0].contourf(GX, GY, np.ma.filled(fx(GX, GY), np.nan), levels=21, cmap="RdBu_r")
    ax[0].set_aspect(2.5); ax[0].set_title("EigenStress FE: σx contour (aspect stretched ×2.5)")
    ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)"); ax[0].grid(False)
    fig.colorbar(cs, ax=ax[0], shrink=0.8, label="MPa")
    for xs, col in zip((50.0, 100.0, 150.0), ("#1baf7a", C_ML, "#6250d6")):
        y, a1, b1, a2, b2 = out[xs]
        ax[1].plot(b1, y, color=C_REF, lw=1.2, ls="--")
        ax[1].plot(a1[::4], y[::4], "o", ms=5, color=col, mec="white", mew=1, label=f"x = {xs:.0f} mm")
        ax[2].plot(b2, y, color=C_REF, lw=1.2, ls="--")
        ax[2].plot(a2[::4], y[::4], "o", ms=5, color=col, mec="white", mew=1)
    ax[1].set_title("σx across the section"); ax[1].set_xlabel("σx (MPa)"); ax[1].set_ylabel("y (mm)")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    ax[2].set_title("τxy across the section"); ax[2].set_xlabel("τxy (MPa)")
    ax[2].text(0.03, 0.03, "dashed = beam theory\ndots = EigenStress FE", transform=ax[2].transAxes, fontsize=8,
               color=INK2)
    fig.suptitle("B · Cantilever with tip load: FE vs beam theory (σ = My/I, parabolic shear)", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig6_beam.png"); plt.close(fig)
    print(f"B  bending peak err {rows[-3]['error_%']:.2f}% | sx profile {e_s:.2f}% | tau {e_t:.2f}%")


# ------------------------------------------------------------------ C. Peterson stepped bar
def peterson_kt(D, d, r):
    """Opposite shoulder fillets, flat bar, axial tension (Pilkey/Peterson curve fits). sigma_nom = P/(t d)."""
    h = (D - d) / 2
    q = h / r
    if 0.1 <= q <= 2.0:
        C = [1.007 + 1.000 * q ** .5 - 0.031 * q, -0.114 - 0.585 * q ** .5 + 0.314 * q,
             0.241 - 0.992 * q ** .5 - 0.271 * q, -0.134 + 0.577 * q ** .5 - 0.012 * q]
    elif 2.0 < q <= 20.0:
        C = [1.042 + 0.982 * q ** .5 - 0.036 * q, -0.074 - 0.156 * q ** .5 - 0.010 * q,
             -3.418 + 1.220 * q ** .5 - 0.005 * q, 3.450 - 2.046 * q ** .5 + 0.051 * q]
    else:
        return np.nan
    z = 2 * h / D
    return C[0] + C[1] * z + C[2] * z ** 2 + C[3] * z ** 3


MODEL_PATH = "singularity_model.joblib"
TAG = ""


def case_peterson():
    model = joblib.load(MODEL_PATH)
    res = []
    for D, d in ((40.0, 20.0), (40.0, 30.0), (30.0, 20.0)):
        h = (D - d) / 2
        Lb = 3 * D                                      # long enough that the clamp / load are far from the step
        y0 = (D - d) / 2
        poly = [np.array(p, float) for p in [(0, 0), (Lb, 0), (Lb, y0), (2 * Lb, y0), (2 * Lb, y0 + d),
                                             (Lb, y0 + d), (Lb, D), (0, D)]]
        S = 100.0                                      # nominal stress in the narrow section (MPa)
        fixed = [((0.0, D), (0.0, 0.0))]
        trac = [((2 * Lb, y0), (2 * Lb, y0 + d), (S, 0.0))]
        part = {"poly": poly, "fixed": fixed, "traction": trac, "corner_radius": 1.0}
        sols = solve_two_meshes(part)                   # the SHARP model, as a user would run it
        for r in (0.5, 1.0, 2.0, 4.0):
            if not (0.1 <= h / r <= 20):
                continue
            kt_ref = peterson_kt(D, d, r)
            # filleted FE (ground truth from our solver)
            m = mesh_polygon(poly, fillets={2: r, 5: r}, refine=[(2, r / 12), (5, r / 12)], h_global=d / 6)
            s = solve_plane_stress(m, fixed, trac)
            near = np.linalg.norm(m.p.T - poly[2], axis=1) < 3 * r
            kt_fe = float(s["vm"][near].max()) / S
            part["corner_radius"] = r
            fl = analyse(part, sols, model)
            pr = fl[fl.vertex == 2].iloc[0]
            kt_ml = float(pr.predicted_real_stress) / S
            kt_naive = float(pr.fe_stress_fine) / S
            res.append({"D": D, "d": d, "r": r, "h/r": h / r, "Kt Peterson": kt_ref, "Kt FE fillet": kt_fe,
                        "Kt EigenStress (from sharp model)": kt_ml, "Kt naive sharp node": kt_naive})
    df = pd.DataFrame(res)
    for col in ("Kt FE fillet", "Kt EigenStress (from sharp model)", "Kt naive sharp node"):
        df[col.replace("Kt", "err %")] = 100 * (df[col] - df["Kt Peterson"]) / df["Kt Peterson"]
    for _, r_ in df.iterrows():
        rows.append({"case": f"C Peterson stepped bar D/d={r_.D / r_.d:.2f}", "quantity": f"Kt, r = {r_.r:g} mm",
                     "reference": r_["Kt Peterson"], "EigenStress": r_["Kt EigenStress (from sharp model)"],
                     "error_%": r_["err % EigenStress (from sharp model)"],
                     "FE fillet": r_["Kt FE fillet"], "FE fillet error_%": r_["err % FE fillet"],
                     "naive sharp node": r_["Kt naive sharp node"], "naive error_%": r_["err % naive sharp node"]})

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for (D, d), col in zip(((40.0, 20.0), (40.0, 30.0), (30.0, 20.0)), (C_ML, "#1baf7a", "#6250d6")):
        g = df[(df.D == D) & (df.d == d)]
        rr = np.linspace(g.r.min(), g.r.max(), 60)
        ax[0].plot(rr / d, [peterson_kt(D, d, x) for x in rr], color=col, lw=1.6, label=f"Peterson, D/d = {D / d:.2f}")
        ax[0].plot(g.r / d, g["Kt EigenStress (from sharp model)"], "o", ms=7, color=col, mec="white", mew=1.2)
        ax[0].plot(g.r / d, g["Kt FE fillet"], "s", ms=6, mfc="none", mec=col, mew=1.4)
    ax[0].plot([], [], "o", color=INK2, label="EigenStress (predicted from SHARP model)")
    ax[0].plot([], [], "s", mfc="none", mec=INK2, label="FE with fillet modelled")
    ax[0].set_xscale("log"); ax[0].set_xlabel("r / d"); ax[0].set_ylabel("Stress concentration factor Kt")
    ax[0].set_title("Kt vs fillet radius"); ax[0].legend(frameon=False, fontsize=8)
    w = 0.27
    x = np.arange(len(df))
    ax[1].bar(x - w, df["err % naive sharp node"].abs(), w, color=C_FE, label="Naive sharp-corner node")
    ax[1].bar(x, df["err % EigenStress (from sharp model)"].abs(), w, color=C_ML, label="EigenStress")
    ax[1].bar(x + w, df["err % FE fillet"].abs(), w, color="#8a8984", label="FE with fillet modelled")
    ax[1].set_xticks(x, [f"D/d {a / b:.1f}\nr={c:g}" for a, b, c in zip(df.D, df.d, df.r)], fontsize=7)
    ax[1].set_ylabel("|error| vs Peterson (%)"); ax[1].set_title("Error against Peterson's charts")
    ax[1].legend(frameon=False, fontsize=8); ax[1].grid(axis="x", visible=False)
    fig.suptitle("C · Stepped flat bar in tension: EigenStress vs Peterson's stress concentration factors",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"fig7_peterson{TAG}.png"); plt.close(fig)
    df.to_csv(f"peterson_table{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(f"C  mean |err| vs Peterson: EigenStress {df['err % EigenStress (from sharp model)'].abs().mean():.2f}% | "
          f"FE fillet {df['err % FE fillet'].abs().mean():.2f}% | naive {df['err % naive sharp node'].abs().mean():.2f}%")


if __name__ == "__main__":
    import sys
    if "--peterson-only" in sys.argv:
        if "--model" in sys.argv:
            MODEL_PATH = sys.argv[sys.argv.index("--model") + 1]
            TAG = "_" + MODEL_PATH.replace("singularity_model", "").replace(".joblib", "").strip("_")
        case_peterson()
        raise SystemExit
    case_kirsch()
    case_beam()
    case_peterson()
    pd.DataFrame(rows).to_csv("validation_reference.csv", index=False)
    print("saved validation_reference.csv, fig5_kirsch.png, fig6_beam.png, fig7_peterson.png")
