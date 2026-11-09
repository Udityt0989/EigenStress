"""
hard_test_lever.py - out-of-distribution stress test: every sharp corner of the topology-optimised lever.
85 re-entrant corners (outline + 11 cutouts), angles 190-300 deg, corners close to each other and to bolt holes,
bearing loads. Truth = the same lever with EVERY corner filleted at rho (drawn arcs), solved with the fillets
meshed at rho/8 and rho/16 (convergence check). Writes hard_test_lever.csv.
"""
import pickle, time, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
from fe2d import mesh_polygon, solve_plane_stress, interior_angles
from sing_tool import solve_two_meshes, analyse, part_holes, hole_fixture_arcs
from design_tools import apply_fillets, max_fillet, interval

T = 10.0
from geometry_io import read_dxf
geo = read_dxf("samples/lever_optimised_sharp.dxf")              # the optimised lever, sharp corners
circles = [(tuple(c), r) for c, r in geo["circles"]]
# bolt holes by position: the two left holes are bolted (fixed), the two right holes take the bearing load
hole_bc = {j: ({"type": "fixed"} if c[0] < 100 else {"type": "bearing", "F": (100 / T, -310 / T)})
           for j, (c, r) in enumerate(circles)}
base = {"poly": [np.asarray(p, float) for p in geo["outline"]], "circles": circles, "poly_holes": geo["poly_holes"],
        "fixed": [], "traction": [], "hole_bc": hole_bc, "E": 210000.0, "nu": 0.3}
model = joblib.load("singularity_model.joblib")
t0 = time.time()
sols = solve_two_meshes(base)
print(f"sharp solves {time.time() - t0:.0f}s, fine nodes {sols['fine']['m'].p.shape[1]}")

out = [tuple(map(float, p)) for p in geo["outline"]]
_, cw = part_holes(base)
vo = [i for i, a in enumerate(interior_angles([np.asarray(p) for p in out])) if a > 190]
vh = [[i for i, a in enumerate(interior_angles([np.asarray(p) for p in r])) if a > 190] for r in cw]
rows = []
for rho in (0.5, 1.0):
    base["corner_radius"] = rho
    fl = analyse(base, sols, model)
    sharp = fl[fl.type == "sharp re-entrant corner"]
    capped_o = {i for i in vo if max_fillet(out, i) < rho}
    capped_h = {(k, i) for k, idx in enumerate(vh) for i in idx if max_fillet(cw[k], i) < rho}
    o2 = apply_fillets(out, {i: min(rho, max_fillet(out, i)) for i in vo}, 3.0)
    h2 = [apply_fillets([tuple(p) for p in r], {i: min(rho, max_fillet(r, i)) for i in idx}, 3.0) for r, idx in zip(cw, vh)]
    # fillet arc points -> refine there
    arc_o = [j for j, p in enumerate(o2) if tuple(p) not in set(out)]
    arc_h = [(k, j) for k, r in enumerate(h2) for j, p in enumerate(r) if tuple(p) not in set(map(tuple, cw[k]))]
    _, _, _, splits = hole_fixture_arcs({**base})
    fx, rad, brg, _ = hole_fixture_arcs({**base})
    peaks = {}
    for div in (8, 16):
        hf = rho / div
        circ = [(c, r, splits.get(j, []), 2 * np.pi * r / 96) for j, (c, r) in enumerate(circles)]
        m = mesh_polygon(o2, refine=[(j, hf) for j in arc_o], h_global=4.0, circles=circ,
                         poly_holes=h2, hole_refine=[(k, j, hf) for k, j in arc_h])
        s = solve_plane_stress(m, [], [], E=210000.0, nu=0.3, fixed_arcs=fx, radial_arcs=rad, bearing_arcs=brg)
        xy = m.p.T
        for _, r in sharp.iterrows():
            P = np.array([r.x, r.y])
            psi = np.radians(360 - r.angle_deg)
            win = 3 * rho / np.sin(psi / 2) + 0.05 * r.ell
            d = np.linalg.norm(xy - P, axis=1)
            peaks[(r.label, div)] = float(s["vm"][d < win].max()) if (d < win).any() else np.nan
        print(f"  rho={rho} div={div}: {m.p.shape[1]} nodes")
    for _, r in sharp.iterrows():
        lab = r.label
        capped = (lab.startswith("V") and int(lab[1:]) in capped_o) or \
                 (lab.startswith("C") and tuple(map(int, lab[1:].split("."))) in capped_h)
        t8, t16 = peaks[(lab, 8)], peaks[(lab, 16)]
        pr = r.get("predicted_real_stress", np.nan)
        kind = "cutout" if lab.startswith("C") else "outline"
        lo, hi, _ = interval(pr, rho / r.ell, kind)
        rows.append({"rho": rho, "corner": lab, "kind": kind, "angle": r.angle_deg, "ell": r.ell, "rho_over_ell": rho / r.ell,
                     "capped": capped, "fe_fine_sharp": r.fe_stress_fine, "pred": pr, "lo90": lo, "hi90": hi,
                     "truth": t16, "truth_change_%": 100 * (t16 / t8 - 1), "fingerprint": r.get("fingerprint"),
                     "note": r.get("note") if isinstance(r.get("note"), str) else ""})
df = pd.DataFrame(rows)
df["err_%"] = 100 * (df.pred / df.truth - 1)
df["sharp_err_%"] = 100 * (df.fe_fine_sharp / df.truth - 1)
df["inside90"] = (df.truth >= df.lo90) & (df.truth <= df.hi90)
df.to_csv("hard_test_lever.csv", index=False)
print(f"done {time.time() - t0:.0f}s")
