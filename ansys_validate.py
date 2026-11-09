"""
ansys_validate.py - fully automatic EigenStress vs ANSYS validation.

For each validation part it
  1. solves the SHARP model at corner element sizes 2 / 1 / 0.5 mm and saves the nodal stresses
     (X, Y, SX, SY, SXY, SEQV) to ansys_out/<part>_sharp_h<h>.csv  - the format EigenStress reads;
  2. solves the FILLETED model (r = 0.5 / 1 / 2 mm) at two fillet mesh sizes (r/8 and r/16)
     -> converged ANSYS peak stress (ground truth) + convergence check;
  3. feeds the ANSYS sharp results (h = 1 and 0.5 mm) into EigenStress and compares its predicted real
     stress with the ANSYS filleted peak.
Results: ansys_out/ansys_validation_results.csv (+ printed table).

Usage (in the EigenStress folder):
  .venv\\Scripts\\python.exe -m pip install ansys-mapdl-core          (once)
  .venv\\Scripts\\python.exe ansys_validate.py --parts lbracket          (quick first test, ~1 min)
  .venv\\Scripts\\python.exe ansys_validate.py                           (all parts)
  python ansys_validate.py --solver python                               (same pipeline, built-in solver)

All parts: 2D plane stress, steel E = 210 GPa, nu = 0.3, thickness 1 mm, PLANE183 (quadratic triangles).
Loads are tractions in MPa on an edge (= N per mm of edge per mm of thickness).
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from fe2d import interior_angles, mesh_polygon, solve_plane_stress
from corner_tools import corner_length
from sing_tool import ansys_sampler, analyse, refine_vertices

warnings.filterwarnings("ignore")
E, NU, THK = 210000.0, 0.3, 1.0
SHARP_H = (2.0, 1.0, 0.5)
RADII = (0.5, 1.0, 2.0)
OUT = "ansys_out"

PARTS = {
    "lbracket": {
        "name": "L-bracket, tip load",
        "outline": [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]],
        "fixed": [[[0, 100], [20, 100]]],
        "traction": [[[100, 0], [100, 20], [0, -50]]],
    },
    "stepped_bar": json.load(open("part_stepped_bar.json")) if os.path.exists("part_stepped_bar.json") else None,
    "v_notch_beam": json.load(open("part_v_notch_beam.json")) if os.path.exists("part_v_notch_beam.json") else None,
    "chamfer_bracket": json.load(open("part_chamfer_bracket.json")) if os.path.exists("part_chamfer_bracket.json") else None,
}


def prep_part(spec):
    part = dict(spec)
    part["poly"] = [np.array(p, float) for p in spec["outline"]]
    part["fixed"] = [tuple(map(tuple, e)) for e in spec.get("fixed", [])]
    part["traction"] = [(tuple(e[0]), tuple(e[1]), tuple(e[2])) for e in spec.get("traction", [])]
    return part


def sharp_corners(part):
    beta = interior_angles(part["poly"])
    return [i for i, b in enumerate(beta) if b > 190.0]


def global_size(part):
    poly = part["poly"]
    size = np.ptp(np.vstack(poly), axis=0).max()
    ells = [corner_length(poly, i) for i in refine_vertices(part)] or [size]
    return min(size / 15, min(ells) / 2)


def segment_index(poly, p, q):
    """Index j of the polygon segment j -> j+1 matching edge (p, q) in either direction."""
    n = len(poly)
    p, q = np.asarray(p, float), np.asarray(q, float)
    for j in range(n):
        a, b = poly[j], poly[(j + 1) % n]
        if (np.allclose(a, p) and np.allclose(b, q)) or (np.allclose(a, q) and np.allclose(b, p)):
            return j
    raise ValueError(f"edge {p}-{q} is not a polygon edge")


# ------------------------------------------------------------------ built-in solver (for testing)
class PythonSolver:
    name = "python"

    def solve(self, part, h_corner=None, fillet_r=None, fillet_div=None):
        poly = part["poly"]
        hg = global_size(part)
        if fillet_r is None:
            m = mesh_polygon(poly, refine=[(i, h_corner) for i in refine_vertices(part)], h_global=hg)
        else:
            cs = sharp_corners(part)
            m = mesh_polygon(poly, fillets={i: fillet_r for i in cs},
                             refine=[(i, fillet_r / fillet_div) for i in cs], h_global=hg)
        s = solve_plane_stress(m, part["fixed"], part["traction"], E=E, nu=NU)
        return pd.DataFrame({"X": m.p[0], "Y": m.p[1], "SX": s["sx"], "SY": s["sy"], "SXY": s["txy"],
                             "SEQV": s["vm"]})

    def close(self):
        pass


# ------------------------------------------------------------------ ANSYS MAPDL solver
class MapdlSolver:
    name = "ansys"

    def __init__(self):
        from ansys.mapdl.core import launch_mapdl
        self.mapdl = launch_mapdl(loglevel="ERROR")

    def _vget(self, item, comp, n):
        m = self.mapdl
        m.run("*DEL,_V_,,NOPR")
        m.run(f"*DIM,_V_,ARRAY,{n}")
        m.run(f"*VGET,_V_(1),NODE,1,{item}" + (f",{comp}" if comp else ""))
        return np.asarray(m.parameters["_V_"], float).ravel()

    def _selected_nodes(self):
        m = self.mapdl
        m.run("*GET,_NMAX_,NODE,0,NUM,MAXD")
        n = int(m.parameters["_NMAX_"])
        sel = self._vget("NSEL", "", n) > 0
        ids = np.arange(1, n + 1)[sel]
        x = self._vget("LOC", "X", n)[sel]
        y = self._vget("LOC", "Y", n)[sel]
        return n, sel, ids, x, y

    def solve(self, part, h_corner=None, fillet_r=None, fillet_div=None):
        m = self.mapdl
        poly = part["poly"]
        n = len(poly)
        m.clear()
        m.prep7()
        m.et(1, "PLANE183")
        m.keyopt(1, 3, 3)                         # plane stress with thickness
        m.r(1, THK)
        m.mp("EX", 1, E)
        m.mp("PRXY", 1, NU)
        for i, (x, y) in enumerate(poly):
            m.k(i + 1, float(x), float(y))
        for i in range(n):                        # line j+1 runs from keypoint j+1 to j+2
            m.l(i + 1, (i + 1) % n + 1)
        fillet_lines = []
        if fillet_r is not None:
            for v in sharp_corners(part):         # vertex v sits between line v (or n) and line v+1
                la, lb = (v if v > 0 else n), v + 1
                m.lfillt(la, lb, fillet_r)
                fillet_lines.append(int(m.get("_LMX_", "LINE", 0, "NUM", "MAX")))
        m.al("ALL")
        m.esize(global_size(part))
        if fillet_r is None:
            for v in refine_vertices(part):
                m.kesize(v + 1, h_corner)
        else:
            for ln in fillet_lines:
                m.lesize(ln, fillet_r / fillet_div)
        m.mshape(0, "2D")
        m.mshkey(0)
        m.amesh("ALL")

        for p, q in part["fixed"]:
            j = segment_index(poly, p, q)
            m.lsel("S", "LINE", "", j + 1)
            m.nsll("S", 1)
            m.d("ALL", "ALL", 0)
            m.allsel()
        for p, q, (tx, ty) in part["traction"]:
            j = segment_index(poly, p, q)
            m.lsel("S", "LINE", "", j + 1)
            m.nsll("S", 1)
            _, _, ids, x, y = self._selected_nodes()
            a = np.asarray(p, float)
            d = np.asarray(q, float) - a
            L = np.linalg.norm(d)
            sp = ((x - a[0]) * d[0] + (y - a[1]) * d[1]) / L
            o = np.argsort(sp)
            ids, sp = ids[o], sp[o]
            w = np.zeros(len(ids))
            if len(ids) % 2 == 1:                 # quadratic edge: corner, mid, corner, ... -> 1/6, 2/3, 1/6
                for k in range(0, len(ids) - 2, 2):
                    Le = sp[k + 2] - sp[k]
                    w[k] += Le / 6
                    w[k + 1] += 2 * Le / 3
                    w[k + 2] += Le / 6
            else:
                w[:] = L / len(ids)
            for nid, wi in zip(ids, w):
                if tx:
                    m.f(int(nid), "FX", float(tx * wi * THK))
                if ty:
                    m.f(int(nid), "FY", float(ty * wi * THK))
            m.allsel()

        m.allsel()
        m.finish()
        m.slashsolu()
        m.antype("STATIC")
        m.solve()
        m.finish()
        m.post1()
        m.set(1, 1)
        m.allsel()
        m.nsle("S", "CORNER")                     # averaged nodal results at element corner nodes
        nmax, sel, ids, x, y = self._selected_nodes()
        out = pd.DataFrame({"X": x, "Y": y,
                            "SX": self._vget("S", "X", nmax)[sel], "SY": self._vget("S", "Y", nmax)[sel],
                            "SXY": self._vget("S", "XY", nmax)[sel], "SEQV": self._vget("S", "EQV", nmax)[sel]})
        m.allsel()
        m.finish()
        return out

    def close(self):
        self.mapdl.exit()


# ------------------------------------------------------------------ pipeline
def corner_value(df, P):
    j = np.argmin(np.hypot(df.X - P[0], df.Y - P[1]))
    return float(df.SEQV.iloc[j])


def fillet_peak(df, part, v, r):
    P = part["poly"][v]
    psi = np.radians(360 - interior_angles(part["poly"])[v])
    win = 3 * r / np.sin(psi / 2) + 0.05 * corner_length(part["poly"], v)
    d = np.hypot(df.X - P[0], df.Y - P[1])
    return float(df.SEQV[d < win].max())


def run(parts, solver, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = joblib.load("singularity_model.joblib")
    rows = []
    for key in parts:
        spec = PARTS[key]
        if spec is None:
            print(f"skip {key}: part file missing")
            continue
        part = prep_part(spec)
        cs = sharp_corners(part)
        print(f"\n=== {spec.get('name', key)}  ({solver.name}) · sharp corners {cs}")
        sharp = {}
        for h in SHARP_H:
            df = solver.solve(part, h_corner=h)
            path = os.path.join(out_dir, f"{key}_sharp_h{h:g}.csv")
            df.to_csv(path, index=False)
            sharp[h] = path
            print(f"  sharp h={h:g} mm: " + ", ".join(f"V{v} {corner_value(df, part['poly'][v]):.1f} MPa" for v in cs))
        truth = {}
        for r in RADII:
            d8 = solver.solve(part, fillet_r=r, fillet_div=8)
            d16 = solver.solve(part, fillet_r=r, fillet_div=16)
            for v in cs:
                p8, p16 = fillet_peak(d8, part, v, r), fillet_peak(d16, part, v, r)
                truth[(v, r)] = (p16, 100 * (p16 - p8) / p8)
            print(f"  fillet r={r:g}: " + ", ".join(f"V{v} {truth[(v, r)][0]:.1f} MPa (mesh change "
                                                 f"{truth[(v, r)][1]:+.1f}%)" for v in cs))
        # EigenStress on the solver's sharp results (coarse h=1, fine h=0.5)
        sols = {}
        for tag, h in (("coarse", 1.0), ("fine", 0.5)):
            sm, xy, vm = ansys_sampler(sharp[h], part["poly"])
            sols[tag] = {"sampler": sm, "xy": xy, "vm": vm, "h": [h] * len(part["poly"])}
        for r in RADII:
            part["corner_radius"] = r
            fl = analyse(part, sols, model)
            for v in cs:
                rr = fl[fl.vertex == v].iloc[0]
                pred = rr.get("predicted_real_stress", np.nan)
                tv, conv = truth[(v, r)]
                rows.append({"part": spec.get("name", key), "solver": solver.name, "corner": f"V{v}",
                             "angle_deg": round(float(interior_angles(part["poly"])[v]), 1), "radius_mm": r,
                             "sharp_h2": corner_value(pd.read_csv(sharp[2.0]), part["poly"][v]),
                             "sharp_h1": corner_value(pd.read_csv(sharp[1.0]), part["poly"][v]),
                             "sharp_h0.5": corner_value(pd.read_csv(sharp[0.5]), part["poly"][v]),
                             "fillet_peak_truth": tv, "fillet_mesh_change_%": conv,
                             "EigenStress_pred": pred,
                             "EigenStress_err_%": 100 * (pred - tv) / tv,
                             "sharp_h0.5_err_%": 100 * (corner_value(pd.read_csv(sharp[0.5]), part["poly"][v]) - tv) / tv,
                             "note": rr.get("note", "") if isinstance(rr.get("note", ""), str) else ""})
    res = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"{solver.name}_validation_results.csv")
    res.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print("\n" + res[["part", "corner", "angle_deg", "radius_mm", "sharp_h2", "sharp_h1", "sharp_h0.5",
                      "fillet_peak_truth", "EigenStress_pred", "EigenStress_err_%", "sharp_h0.5_err_%"]]
          .round(1).to_string(index=False))
    print(f"\nmean |EigenStress error| {res['EigenStress_err_%'].abs().mean():.2f}%  "
          f"(max {res['EigenStress_err_%'].abs().max():.2f}%)  |  mean |sharp 0.5 mm reading error| "
          f"{res['sharp_h0.5_err_%'].abs().mean():.2f}%")
    print(f"saved {path} and the nodal CSVs in {out_dir}/")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["ansys", "python"], default="ansys")
    ap.add_argument("--parts", nargs="*", default=list(PARTS))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    solver = MapdlSolver() if args.solver == "ansys" else PythonSolver()
    try:
        run(args.parts, solver, args.out)
    finally:
        solver.close()


if __name__ == "__main__":
    main()
