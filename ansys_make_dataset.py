"""
ansys_make_dataset.py - generate EigenStress training data with ANSYS MAPDL as the solver.

Same geometry families, random parameters, corner features and CSV columns as make_dataset.py
(seed N gives the same part in both), but every FE solve is done by ANSYS MAPDL (PLANE183, plane
stress, averaged nodal stresses), driven through PyMAPDL. The output CSV can be fed straight into
train_model.py (--extra) or checked against the current model with ansys_check_model.py.

Per geometry: 3 filleted solves (target = real peak stress) + 4 sharp solves at l/10 ... l/80
(features) = 7 MAPDL runs, roughly 10-30 s per geometry depending on the PC.

Tested with ansys-mapdl-core 0.7x; written for ANSYS 2025 R1 (v251).

Setup (once, in the EigenStress folder, Windows cmd):
    .venv\\Scripts\\python.exe -m pip install ansys-mapdl-core
Quick test (2 geometries, ~1 min):
    .venv\\Scripts\\python.exe ansys_make_dataset.py --family bracket --n 2 --out ansys_test.csv
Real runs (examples; seeds match the Python datasets so the two solvers can be compared 1:1):
    .venv\\Scripts\\python.exe ansys_make_dataset.py --family bracket       --n 200 --seed0 0      --out ansys_bracket.csv
    .venv\\Scripts\\python.exe ansys_make_dataset.py --family notched_plate --n 150 --seed0 100000 --out ansys_notch.csv
    .venv\\Scripts\\python.exe ansys_make_dataset.py --family stepped       --n 150 --seed0 300000 --rho-max 1.0 --out ansys_stepped.csv
    .venv\\Scripts\\python.exe ansys_make_dataset.py --family window        --n 150 --seed0 600000 --rho-max 1.0 --out ansys_window.csv
Options:
    --workers 2        two MAPDL instances in parallel (needs 2 solver licences; each runs on 1 core)
    --exec "C:\\Program Files\\ANSYS Inc\\v251\\ansys\\bin\\winx64\\ANSYS251.exe"   if MAPDL is not found
    --solver emulate   no ANSYS: built-in solver, but results go through exactly the same
                       node-cloud -> features path as ANSYS results (to test this script)
The run can be stopped at any time (Ctrl+C) and restarted with the same command: finished seeds are
skipped and new rows are appended.
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from matplotlib.path import Path

import make_dataset as md
from fe2d import mesh_polygon as py_mesh_polygon, solve_plane_stress as py_solve

E, NU, THK = 210000.0, 0.3, 1.0


# ------------------------------------------------------------------ geometry spec (what make_dataset asks to mesh)
class Spec:
    """Stand-in for a mesh. make_dataset only uses m.p (2 x N node coordinates) and, through
    sampler_from_mesh, m.t (3 x M triangles); both are filled in after the solve."""

    def __init__(self, poly, fillets=None, refine=None, h_global=5.0, poly_holes=None, hole_refine=None, **_):
        self.poly = [np.asarray(p, float) for p in poly]
        self.fillets = dict(fillets or {})
        self.refine = list(refine or [])
        self.h_global = float(h_global)
        self.poly_holes = [[np.asarray(p, float) for p in h] for h in (poly_holes or [])]
        self.hole_refine = list(hole_refine or [])
        self.p = None
        self.t = None

    def size(self):
        P = np.vstack(self.poly)
        return float(np.ptp(P, axis=0).max())


def node_cloud_mesh(spec, x, y):
    """Delaunay triangulation of the corner nodes, minus triangles outside the part (across notches / holes)."""
    from scipy.spatial import Delaunay
    xy = np.column_stack([x, y])
    tri = Delaunay(xy).simplices
    cen = xy[tri].mean(1)
    keep = Path(np.vstack(spec.poly)).contains_points(cen)
    if spec.fillets:          # a filleted re-entrant corner adds material outside the sharp polygon: keep it
        for i, r in spec.fillets.items():
            keep |= np.linalg.norm(cen - spec.poly[i], axis=1) < 3 * r
    for h in spec.poly_holes:
        keep &= ~Path(np.vstack(h)).contains_points(cen)
    spec.p = xy.T
    spec.t = tri[keep].T


# ------------------------------------------------------------------ ANSYS backend
class MapdlBackend:
    def __init__(self, exec_file=None, port=None, nproc=1):
        from ansys.mapdl.core import launch_mapdl
        kw = {"loglevel": "ERROR", "nproc": nproc, "override": True, "cleanup_on_exit": True}
        if exec_file:
            kw["exec_file"] = exec_file
        if port:
            kw["port"] = port
        self.m = launch_mapdl(**kw)
        print(f"MAPDL started: {self.m.version if hasattr(self.m, 'version') else ''}", flush=True)

    # --- helpers
    def _vget(self, item, comp, n):
        m = self.m
        m.run("*DEL,_V_,,NOPR")
        m.run(f"*DIM,_V_,ARRAY,{n}")
        m.run(f"*VGET,_V_(1),NODE,1,{item}" + (f",{comp}" if comp else ""))
        return np.asarray(m.parameters["_V_"], float).ravel()

    def _selected_nodes(self):
        m = self.m
        m.run("*GET,_NMAX_,NODE,0,NUM,MAXD")
        n = int(m.parameters["_NMAX_"])
        sel = self._vget("NSEL", "", n) > 0
        ids = np.arange(1, n + 1)[sel]
        return n, sel, ids, self._vget("LOC", "X", n)[sel], self._vget("LOC", "Y", n)[sel]

    def _kp_at(self, P, tol):
        m = self.m
        m.ksel("S", "LOC", "X", P[0] - tol, P[0] + tol)
        m.ksel("R", "LOC", "Y", P[1] - tol, P[1] + tol)
        cnt = int(m.get("_KC_", "KP", 0, "COUNT"))
        k = int(m.get("_KN_", "KP", 0, "NUM", "MAX")) if cnt else 0
        m.ksel("ALL")
        return k

    def _select_edge_nodes(self, p, q, tol):
        """Select the line whose centroid is the midpoint of segment p-q, then its nodes."""
        m = self.m
        c = 0.5 * (np.asarray(p, float) + np.asarray(q, float))
        m.lsel("S", "LOC", "X", c[0] - tol, c[0] + tol)
        m.lsel("R", "LOC", "Y", c[1] - tol, c[1] + tol)
        if int(m.get("_LC_", "LINE", 0, "COUNT")) == 0:
            m.allsel()
            raise RuntimeError(f"no line found at {c}")
        m.nsll("S", 1)

    # --- build, mesh, solve
    def solve(self, spec, fixed, traction):
        m = self.m
        tol = 1e-5 * spec.size()
        m.clear()
        m.prep7()
        m.et(1, "PLANE183")
        m.keyopt(1, 1, 1)                    # triangle shape option (6-node)
        m.keyopt(1, 3, 3)                    # plane stress with thickness
        m.r(1, THK)
        m.mp("EX", 1, E)
        m.mp("PRXY", 1, NU)
        poly = spec.poly
        n = len(poly)
        for i, (x, y) in enumerate(poly):
            m.k(i + 1, float(x), float(y))
        for i in range(n):                   # line i+1: vertex i -> vertex i+1
            m.l(i + 1, (i + 1) % n + 1)
        fillet_lines = {}
        for v, r in spec.fillets.items():    # vertex v sits between line v (or n) and line v+1
            la, lb = (v if v > 0 else n), v + 1
            m.lfillt(la, lb, float(r))
            fillet_lines[v] = int(m.get("_LMX_", "LINE", 0, "NUM", "MAX"))
        m.al("ALL")
        a_out = int(m.get("_AMX_", "AREA", 0, "NUM", "MAX"))
        for k, ring in enumerate(spec.poly_holes):
            base = 100000 * (k + 1)
            nh = len(ring)
            for i, (x, y) in enumerate(ring):
                m.k(base + i + 1, float(x), float(y))
            l0 = int(m.get("_LMX_", "LINE", 0, "NUM", "MAX")) + 1
            for i in range(nh):
                m.l(base + i + 1, base + (i + 1) % nh + 1)
            l1 = int(m.get("_LMX_", "LINE", 0, "NUM", "MAX"))
            m.lsel("S", "LINE", "", l0, l1)
            m.al("ALL")
            a_hole = int(m.get("_AMX_", "AREA", 0, "NUM", "MAX"))
            m.allsel()
            m.asba(a_out, a_hole)
            a_out = int(m.get("_AMX_", "AREA", 0, "NUM", "MAX"))
        m.allsel()
        m.esize(spec.h_global)
        m.kesize("ALL", spec.h_global)        # every keypoint gets the global size, so lines grade smoothly
        for i, h in spec.refine:
            if i in fillet_lines:
                m.lesize(fillet_lines[i], float(h))
            else:
                kp = self._kp_at(poly[i], tol)
                if kp:
                    m.kesize(kp, float(h))
        for k, i, h in spec.hole_refine:
            kp = self._kp_at(spec.poly_holes[k][i], tol)
            if kp:
                m.kesize(kp, float(h))
        m.mshape(1, "2D")                    # 1 = triangles
        m.mshkey(0)                          # free mesh
        m.amesh("ALL")

        for p, q in fixed:
            self._select_edge_nodes(p, q, tol)
            m.d("ALL", "ALL", 0)
            m.allsel()
        for p, q, (tx, ty) in traction:
            self._select_edge_nodes(p, q, tol)
            _, _, ids, x, y = self._selected_nodes()
            a = np.asarray(p, float)
            d = np.asarray(q, float) - a
            L = np.linalg.norm(d)
            sp = ((x - a[0]) * d[0] + (y - a[1]) * d[1]) / L
            o = np.argsort(sp)
            ids, sp = ids[o], sp[o]
            w = np.zeros(len(ids))
            if len(ids) % 2 == 1:            # quadratic edge: corner, mid, corner... -> 1/6, 2/3, 1/6
                for j in range(0, len(ids) - 2, 2):
                    Le = sp[j + 2] - sp[j]
                    w[j] += Le / 6
                    w[j + 1] += 2 * Le / 3
                    w[j + 2] += Le / 6
            else:
                w[:] = L / len(ids)
            with m.non_interactive:
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
        m.nsle("S", "CORNER")                # averaged nodal stresses at element corner nodes
        nmax, sel, _, x, y = self._selected_nodes()
        s = {"sx": self._vget("S", "X", nmax)[sel], "sy": self._vget("S", "Y", nmax)[sel],
             "txy": self._vget("S", "XY", nmax)[sel], "vm": self._vget("S", "EQV", nmax)[sel]}
        m.allsel()
        m.finish()
        node_cloud_mesh(spec, x, y)
        return s

    def close(self):
        try:
            self.m.exit()
        except Exception:
            pass


class EmulateBackend:
    """Built-in solver, but only corner-node values are passed on (re-triangulated), like ANSYS output."""

    def solve(self, spec, fixed, traction):
        fil = spec.fillets or None
        mm = py_mesh_polygon(spec.poly, fillets=fil, refine=spec.refine, h_global=spec.h_global,
                             poly_holes=[[tuple(p) for p in h] for h in spec.poly_holes] or None,
                             hole_refine=spec.hole_refine or None)
        s = py_solve(mm, fixed, traction, E=E, nu=NU)
        nc = np.unique(mm.t[:3])                       # corner nodes only
        node_cloud_mesh(spec, mm.p[0, nc], mm.p[1, nc])
        return {k: np.asarray(s[k])[nc] for k in ("sx", "sy", "txy", "vm")}

    def close(self):
        pass


BACKEND = None


def _mesh(poly, **kw):
    return Spec(poly, **kw)


def _solve(spec, fixed, traction, **_):
    return BACKEND.solve(spec, fixed, traction)


def _init(family, rho, solver, exec_file, port):
    global BACKEND
    md._set_family(family, rho)
    md.mesh_polygon = _mesh               # make_dataset now builds Specs and solves them with BACKEND
    md.solve_plane_stress = _solve
    BACKEND = MapdlBackend(exec_file, port) if solver == "ansys" else EmulateBackend()


def _one(seed):
    try:
        return seed, md.run_geometry(seed), None
    except Exception as e:
        return seed, None, f"{e}\n{traceback.format_exc(limit=2)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="bracket", choices=["bracket", "notched_plate", "stepped", "window"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="ansys_dataset.csv")
    ap.add_argument("--rho-min", type=float, default=0.01)
    ap.add_argument("--rho-max", type=float, default=0.3)
    ap.add_argument("--solver", default="ansys", choices=["ansys", "emulate"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--exec", dest="exec_file", default=None, help="path to ANSYS251.exe if not auto-detected")
    a = ap.parse_args()
    rho = (a.rho_min, a.rho_max)

    done = set()
    if os.path.exists(a.out):
        try:
            done = set(pd.read_csv(a.out, usecols=["seed"])["seed"].astype(int))
        except Exception:
            pass
    donef = a.out + ".done"                          # seeds that ran (including ones that gave no rows)
    if os.path.exists(donef):
        done |= {int(s) for s in open(donef).read().split()}
    seeds = [s for s in range(a.seed0, a.seed0 + a.n) if s not in done]
    print(f"{a.family}: {len(seeds)} geometries to run ({len(done)} already done) -> {a.out}", flush=True)
    if not seeds:
        return

    t0, nrows, nfail = time.time(), 0, 0

    def record(seed, rows, err, i):
        nonlocal nrows, nfail
        if err:
            nfail += 1
            print(f"  seed {seed} failed: {err.splitlines()[0]}", flush=True)
        elif rows:
            df = pd.DataFrame(rows)
            df["solver"] = a.solver
            df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
            nrows += len(df)
        with open(donef, "a") as fh:
            fh.write(f"{seed}\n")
        el = time.time() - t0
        print(f"  {i}/{len(seeds)}  seed {seed}: {len(rows or [])} rows  |  total {nrows} rows, {nfail} failed, "
              f"{el / 60:.1f} min, ~{el / i * (len(seeds) - i) / 60:.0f} min left", flush=True)

    if a.workers <= 1:
        _init(a.family, rho, a.solver, a.exec_file, None)
        try:
            for i, s in enumerate(seeds, 1):
                record(*_one(s), i)
        finally:
            BACKEND.close()
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        ports = iter(range(50060, 50060 + a.workers))
        pools = []
        # one MAPDL per worker, each on its own port
        chunks = [seeds[w::a.workers] for w in range(a.workers)]
        for w in range(a.workers):
            pools.append(ctx.Pool(1, initializer=_init, initargs=(a.family, rho, a.solver, a.exec_file, next(ports))))
        its = [p.imap(_one, c) for p, c in zip(pools, chunks)]
        i = 0
        try:
            active = list(its)
            while active:
                for it in list(active):
                    try:
                        res = next(it)
                    except StopIteration:
                        active.remove(it)
                        continue
                    i += 1
                    record(*res, i)
        finally:
            for p in pools:
                p.terminate()
    print(f"done: {nrows} new rows, {nfail} failed, {(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
