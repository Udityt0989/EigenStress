"""
Paired training-data generator for the FEM stress-singularity ML model.

For each randomly sampled L-bracket geometry it solves, in ANSYS MAPDL (static structural, 2D plane stress):
  1. the SHARP re-entrant corner at several mesh sizes -> singular stress that grows with refinement (model INPUT)
  2. the same part with a real FILLET radius, finely meshed  -> finite peak stress (model TARGET)

Each output row = one sharp-mesh solve + the matching fillet target, so the model learns
"singular FE field near the corner + geometry + mesh size  ->  real peak stress".

Requirements:  pip install ansys-mapdl-core numpy pandas   (plus a local ANSYS licence)
Usage:         python generate_lbracket_data.py --n 200 --out lbracket_dataset.csv
Units:         mm, N, MPa
"""
import argparse
import time

import numpy as np
import pandas as pd
from ansys.mapdl.core import launch_mapdl

E, NU, THK = 210000.0, 0.3, 1.0          # steel, plane stress with unit thickness
SAMPLE_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2]  # sampling distances from corner, as fraction of leg width a
SHARP_REFINE = [10, 20, 40]               # sharp-corner local element size = a / value


def build_geometry(mapdl, L, a, r):
    """L-bracket: outer corner at origin, legs of width a, overall size L. Re-entrant corner at (a, a)."""
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "PLANE183")
    mapdl.keyopt(1, 3, 3)                 # plane stress with thickness
    mapdl.r(1, THK)
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, NU)

    pts = [(0, 0), (L, 0), (L, a), (a, a), (a, L), (0, L)]
    for i, (x, y) in enumerate(pts, start=1):
        mapdl.k(i, x, y)
    for i in range(1, 7):
        mapdl.l(i, i % 6 + 1)             # lines 3 and 4 meet at the re-entrant corner (kp 4)

    fillet_line = None
    if r > 0:
        mapdl.lfillt(3, 4, r)
        fillet_line = int(mapdl.get("_NL", "LINE", 0, "NUM", "MAX"))  # fillet gets the highest line number
    mapdl.al("ALL")
    return fillet_line


def mesh(mapdl, a, h_local, fillet_line):
    mapdl.esize(a / 4)                    # coarse global size
    if fillet_line is None:
        mapdl.kesize(4, h_local)          # refine at the sharp corner
    else:
        mapdl.lesize(fillet_line, h_local)  # refine along the fillet arc
    mapdl.mshape(0, "2D")
    mapdl.mshkey(0)
    mapdl.amesh("ALL")


def apply_bcs_and_solve(mapdl, L, F):
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Y", L)        # top edge of vertical leg: fully fixed
    mapdl.d("ALL", "ALL", 0)
    mapdl.nsel("S", "LOC", "X", L)        # free end of horizontal leg: downward load
    n = int(mapdl.get("_NN", "NODE", 0, "COUNT"))
    mapdl.f("ALL", "FY", -F / n)          # spread equally over edge nodes (far from corner, St-Venant)
    mapdl.allsel()
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()
    mapdl.post1()
    mapdl.set(1, 1)
    mapdl.allsel()


def nodal_vm(mapdl):
    """Averaged nodal von Mises stress with node coordinates; drops midside nodes (reported as 0)."""
    s = np.asarray(mapdl.post_processing.nodal_eqv_stress())
    xy = np.asarray(mapdl.mesh.nodes)[:, :2]
    keep = s > 0
    return xy[keep], s[keep]


def sharp_features(xy, s, a):
    """Stress at the corner node and at points along the bisector into the material."""
    corner = np.array([a, a])
    d_corner = np.linalg.norm(xy - corner, axis=1)
    feats = {"s_corner": float(s[np.argmin(d_corner)])}
    direction = np.array([-1.0, -1.0]) / np.sqrt(2)     # bisector pointing into the solid
    for frac in SAMPLE_FRACS:
        p = corner + frac * a * direction
        feats[f"s_d{frac}"] = float(s[np.argmin(np.linalg.norm(xy - p, axis=1))])  # nearest node (v1)
    feats["s_max_local"] = float(s[d_corner < 0.25 * a].max())
    return feats


def fillet_peak(xy, s, a, r):
    """Peak stress in a local window around the fillet (avoids the singular support corners)."""
    window = np.linalg.norm(xy - np.array([a, a]), axis=1) < 2 * r + 0.05 * a
    return float(s[window].max())


def run_case(mapdl, L, a, r, F):
    rows = []
    # target: filleted geometry, fine mesh on the arc
    fl = build_geometry(mapdl, L, a, r)
    mesh(mapdl, a, r / 8, fl)
    apply_bcs_and_solve(mapdl, L, F)
    xy, s = nodal_vm(mapdl)
    target = fillet_peak(xy, s, a, r)

    # inputs: sharp geometry at several mesh densities
    for k in SHARP_REFINE:
        h = a / k
        build_geometry(mapdl, L, a, 0.0)
        mesh(mapdl, a, h, None)
        apply_bcs_and_solve(mapdl, L, F)
        xy, s = nodal_vm(mapdl)
        row = {"L": L, "a": a, "r": r, "F": F, "h_local": h, "h_over_a": 1 / k,
               "s_nominal": 6 * F * (L - a) / (THK * a ** 2)}  # beam bending stress at the corner section
        row.update(sharp_features(xy, s, a))
        row["target_fillet_peak"] = target
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of geometries")
    ap.add_argument("--out", default="lbracket_dataset.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mapdl = launch_mapdl(loglevel="ERROR")
    rows, t0 = [], time.time()
    try:
        for i in range(args.n):
            a = rng.uniform(5, 30)
            L = rng.uniform(3 * a, 6 * a)
            r = rng.uniform(0.02, 0.3) * a
            F = rng.uniform(100, 2000)
            try:
                rows += run_case(mapdl, L, a, r, F)
            except Exception as e:  # keep going if one geometry fails to mesh/solve
                print(f"case {i} failed: {e}")
                continue
            pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint every case
            print(f"{i + 1}/{args.n} done  ({time.time() - t0:.0f}s)")
    finally:
        mapdl.exit()


if __name__ == "__main__":
    main()
