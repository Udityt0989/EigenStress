"""
deformation_study.py - "deformation tells the truth where stress lies".

L-bracket (100 x 100 mm, 20 mm arms, t = 10 mm, steel), top edge clamped, 1000 N down on the right end.
The sharp inner corner (270 deg, lam = 0.5445) is refined from l/5 to l/160 (32x). At each level we record
  * the corner-node von Mises stress          (singular: should keep rising ~ 2^(1-lam) per halving)
  * the tip deflection and the strain energy  (finite: should converge)
  * K1 from the stresses (bisector fit) and K1 from the DISPLACEMENTS (flank opening), and the exponent
    measured from the displacements alone
and compare the deflection with a hand calculation (Castigliano, beam theory on the centre lines).
Writes deformation_study.csv and fig8_deformation.png.
"""
import numpy as np
import pandas as pd

from fe2d import mesh_polygon, solve_plane_stress
from corner_tools import corner_features, sampler_from_mesh, corner_length, williams_lambda
from deformation import DispSampler, corner_opening

E, NU, T, P = 210000.0, 0.3, 10.0, 1000.0
POLY = [np.array(p, float) for p in [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]]]
CORNER = 3
FIXED = [((20.0, 100.0), (0.0, 100.0))]
LOAD = [((100.0, 0.0), (100.0, 20.0), (0.0, -P / (20 * T)))]       # traction (MPa) over the 20 mm end


def castigliano():
    """Centre-line frame: horizontal arm a = 90 mm (tip to joint), vertical arm L = 90 mm (joint to clamp).
    U = int M^2/2EI + N^2/2EA + k V^2/2GA ;  delta = dU/dP."""
    h, a, L = 20.0, 90.0, 90.0
    I, A, G, k = T * h ** 3 / 12, T * h, E / (2 * (1 + NU)), 1.2
    parts = {"bending, horizontal arm  P a^3 / 3EI": P * a ** 3 / (3 * E * I),
             "bending, vertical arm  P a^2 L / EI": P * a ** 2 * L / (E * I),
             "axial, vertical arm  P L / EA": P * L / (E * A),
             "shear, horizontal arm  k P a / GA": k * P * a / (G * A)}
    return sum(parts.values()), parts


def castigliano_rigid_joint():
    """Lower bound: the 20 x 20 mm joint block treated as rigid. Flexible lengths a' = L' = 80 mm,
    joint moment still P x 90 mm."""
    h, a, L, arm = 20.0, 80.0, 80.0, 90.0
    I, A, G, k = T * h ** 3 / 12, T * h, E / (2 * (1 + NU)), 1.2
    return (P * a ** 3 / (3 * E * I) + P * arm ** 2 * L / (E * I) + P * L / (E * A) + k * P * a / (G * A))


def run(divs=(5, 10, 20, 40, 80, 160)):
    ell = corner_length(POLY, CORNER)
    lam = williams_lambda(270.0)
    rows = []
    for d in divs:
        h = ell / d
        m = mesh_polygon(POLY, refine=[(CORNER, h)], h_global=4.0)
        s = solve_plane_stress(m, FIXED, LOAD, E=E, nu=NU)
        xy = m.p.T
        jc = np.argmin(np.linalg.norm(xy - POLY[CORNER], axis=1))
        tip = np.abs(xy[:, 0] - 100) < 1e-6
        f = corner_features(sampler_from_mesh(m, s), POLY, CORNER, h, ell=ell)
        op = corner_opening(DispSampler(m, s), POLY, CORNER, ell, E, NU)
        from corner_tools import bisector_profile
        r0 = np.array([0.02 * ell])
        k1_tip = bisector_profile(sampler_from_mesh(m, s), POLY, CORNER, r0)[0][0] * r0[0] ** (1 - lam)
        rows.append({"h_over_ell": 1 / d, "K1_stress_tip": k1_tip, "nodes": m.p.shape[1], "corner_vm": s["vm"][jc],
                     "tip_deflection": -s["uy"][tip].mean(), "strain_energy_Nmm": s["energy"] * T,
                     "K1_stress": f["K1"] if f else np.nan, "K1_disp": op["K1_u"], "lam_disp": op["lam_u"],
                     "lam_theory": lam})
        print(rows[-1])
    return pd.DataFrame(rows)


def figure(df, hand):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    x = 1 / df.h_over_ell
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ref = lambda c: 100 * df[c] / df[c].iloc[-1]
    ax[0].plot(x, ref("corner_vm"), "o-", c="#c2410c", label="corner stress (singular)")
    ax[0].plot(x, ref("tip_deflection"), "s-", c="#1d4ed8", label="tip deflection")
    ax[0].plot(x, ref("strain_energy_Nmm"), "^-", c="#0f766e", label="strain energy")
    ax[0].plot(x, ref("K1_disp"), "d--", c="#7c3aed", label="K1 from displacement")
    ax[0].set_xlabel("corner refinement  l / h")
    ax[0].set_ylabel("value as % of finest mesh")
    ax[0].set_title(f"Stress diverges, deformation converges\n(hand calc δ = {df.hand_rigid.iloc[0]:.3f}–{hand:.3f} mm, "
                    f"FE δ = {df.tip_deflection.iloc[-1]:.3f} mm)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9)
    ref = df.K1_stress_tip.iloc[-1]
    ax[1].plot(x, df.K1_disp, "d-", c="#7c3aed", label="K1 from flank opening (displacement)")
    ax[1].plot(x, df.K1_stress_tip, "o-", c="#c2410c", label="K1 from stress at r = 0.02 l")
    ax[1].axhspan(0.97 * ref, 1.03 * ref, color="#7c3aed", alpha=0.08, lw=0)
    ax[1].set_ylim(0, 1.5 * ref)
    ax[1].set_xlabel("corner refinement  l / h")
    ax[1].set_ylabel("K1  (MPa · mm^(1−λ))")
    ax[1].set_title("Corner intensity: displacement settles on a coarse mesh", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    for a_ in ax:
        a_.set_xscale("log", base=2)
        a_.set_xticks(list(x))
        a_.set_xticklabels([f"{v:g}" for v in x])
    fig.tight_layout()
    fig.savefig("fig8_deformation.png", dpi=170)


if __name__ == "__main__":
    hand, parts = castigliano()
    for k, v in parts.items():
        print(f"{k:45s} {v:.4f} mm")
    print(f"Castigliano total {hand:.4f} mm")
    df = run()
    df["hand_deflection"] = hand
    df["hand_rigid"] = castigliano_rigid_joint()
    print("rigid-joint bound", df.hand_rigid.iloc[0])
    df.to_csv("deformation_study.csv", index=False)
    figure(df, hand)
    print(df.round(4).to_string())
