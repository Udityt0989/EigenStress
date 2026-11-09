"""
sing_tool.py - FEM singularity finder + real-stress predictor.

    part (2D outline + supports + loads)
        -> FE solve at a coarse and a refined mesh   (built-in solver, or ANSYS CSV exports)
        -> flag singular points (stress keeps rising under refinement)
        -> classify: sharp re-entrant corner (geometry)  vs  support / load point (boundary condition)
        -> for geometric corners: read the mesh-independent singular field and predict the
           finite peak stress for a real corner radius rho

Part file (JSON):
{
  "name": "L-bracket",
  "outline": [[0,0],[100,0],[100,20],[20,20],[20,100],[0,100]],   # CCW, mm
  "fixed":   [[[0,100],[20,100]]],                                  # clamped edges
  "traction":[[[100,0],[100,20],[0,-50]]],                          # edge + (tx, ty) MPa
  "corner_radius": 0.5,                                             # default real radius, mm
  "corner_radius_by_vertex": {"3": 1.0},                            # optional overrides
  "E": 210000, "nu": 0.3
}

Usage:
  python sing_tool.py part.json                        # built-in solver
  python sing_tool.py part.json --ansys coarse.csv fine.csv --h 2.0 0.5
        (ANSYS nodal exports with columns x, y, SX, SY, SXY, SEQV; --h = element size at corners)
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from matplotlib.path import Path

from fe2d import mesh_polygon, solve_plane_stress, interior_angles, signed_area
from corner_tools import (FieldSampler, sampler_from_mesh, corner_features, corner_length,
                          williams_lambda, MAX_H_OVER_ELL)

MODEL_PATH = "singularity_model.joblib"
RATIO_THRESHOLD = 1.05   # stress at a point must rise >5% when the local mesh is halved to be "singular"


# ------------------------------------------------------------------ inputs
def load_part(path):
    part = json.load(open(path))
    poly = [np.array(p, float) for p in part["outline"]]
    if signed_area(poly) < 0:
        raise ValueError("outline must be counter-clockwise")
    part["poly"] = poly
    return part


def bc_vertices(part):
    """Vertices that touch a support or a loaded edge (BC singularities live here)."""
    poly = part["poly"]
    ends = [np.array(p, float) for e in part.get("fixed", []) for p in e]
    ends += [np.array(p, float) for e in part.get("traction", []) for p in e[:2]]
    return {i for i, P in enumerate(poly) if any(np.linalg.norm(P - q) < 1e-6 for q in ends)}


def fixed_vertices(part):
    """Vertices touching a clamped edge (there the corner field is changed by the clamp itself)."""
    poly = part["poly"]
    ends = [np.array(p, float) for e in part.get("fixed", []) for p in e]
    return {i for i, P in enumerate(poly) if any(np.linalg.norm(P - q) < 1e-6 for q in ends)}


def ansys_sampler(csv_path, poly):
    """Build a field sampler from an ANSYS nodal export (x, y, SX, SY, SXY, SEQV)."""
    from scipy.spatial import Delaunay
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    xy = df[["X", "Y"]].to_numpy(float)
    tri = Delaunay(xy).simplices
    inside = Path(np.vstack(poly)).contains_points(xy[tri].mean(1))   # drop triangles across notches
    st = {"sx": df["SX"].to_numpy(), "sy": df["SY"].to_numpy(), "txy": df["SXY"].to_numpy(),
          "vm": df["SEQV"].to_numpy(), "tri": tri[inside]}
    return FieldSampler(xy, st), xy, st["vm"]


# ------------------------------------------------------------------ core
CANDIDATE_ANGLE = 190.0   # turns sharper than this are treated as potential singular corners
MODEL_BETA_RANGE = (190.0, 330.0)


def part_holes(part):
    return ([((c[0][0], c[0][1]), c[1]) for c in part.get("circles", [])],
            [[tuple(p) for p in h] for h in part.get("poly_holes", [])])


HOLE_DIVS = {"coarse": 48, "fine": 96}   # elements around each hole circumference


def hole_fixture_arcs(part):
    """
    part["hole_bc"] = {hole_index: {"type": "fixed" | "pin" | "bearing",
                                    "angle": contact-centre angle in deg (pin),
                                    "F": (Fx, Fy) bearing force per unit thickness, N/mm}}
    Returns solver arc lists + the angles where each fixture starts/ends (mesh nodes are placed there).
    """
    circles = part.get("circles", [])
    fixed, radial, bearing, splits = [], [], [], {j: [] for j in range(len(circles))}
    for j, bc in part.get("hole_bc", {}).items():
        j = int(j)
        c, r = tuple(circles[j][0]), circles[j][1]
        if bc["type"] == "fixed":
            fixed.append((c, r, 0.0, 360.0))
        elif bc["type"] == "pin":
            a = bc.get("angle", 90.0)
            radial.append((c, r, a - 90.0, a + 90.0))
            splits[j] += [a - 90.0, a + 90.0]
        elif bc["type"] == "bearing":
            Fx, Fy = bc["F"]
            if Fx or Fy:
                th = np.degrees(np.arctan2(Fy, Fx))
                bearing.append((c, r, (Fx, Fy)))
                splits[j] += [th - 90.0, th + 90.0]
    return fixed, radial, bearing, splits


def refine_vertices(part):
    beta = interior_angles(part["poly"])
    return sorted({i for i, b in enumerate(beta) if b > CANDIDATE_ANGLE} | bc_vertices(part))


def min_wall(part, n_pts=1500):
    """Thinnest wall/strut in the part: small percentile of the distance from each boundary point to the
    facing boundary (another loop, or the same loop far away along its length)."""
    from scipy.spatial import cKDTree
    loops = [np.asarray(part["poly"], float)] + [np.asarray(h, float) for h in part.get("poly_holes", [])]
    for c, r in part.get("circles", []):
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        loops.append(np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)])
    perim = sum(np.linalg.norm(np.diff(np.vstack([L, L[:1]]), axis=0), axis=1).sum() for L in loops)
    step = perim / n_pts
    P, lid, s_arc = [], [], []
    for k, L in enumerate(loops):
        Lc = np.vstack([L, L[:1]])
        seg = np.linalg.norm(np.diff(Lc, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        ss = np.arange(0, cum[-1], step)
        P.append(np.c_[np.interp(ss, cum, Lc[:, 0]), np.interp(ss, cum, Lc[:, 1])])
        lid.append(np.full(len(ss), k)); s_arc.append(np.c_[ss, np.full(len(ss), cum[-1])])
    P, lid, s_arc = np.vstack(P), np.concatenate(lid), np.vstack(s_arc)
    T = cKDTree(P)
    d, j = T.query(P, k=min(40, len(P)))
    best = np.full(len(P), np.inf)
    for col in range(1, d.shape[1]):
        jj = j[:, col]
        other = lid[jj] != lid
        ds = np.abs(s_arc[jj, 0] - s_arc[:, 0])
        ds = np.minimum(ds, s_arc[:, 1] - ds)
        far = (~other) & (ds > 4 * d[:, col] + 2 * step)
        ok = (other | far) & (d[:, col] < best)
        best[ok] = d[ok, col]
    best = best[np.isfinite(best)]
    return float(np.percentile(best, 5)) if len(best) else np.inf


def solve_two_meshes(part, h_fine_div=40, h_coarse_div=20):
    """Built-in solver: refine corners and support/load ends to l/div, coarse then fine (factor 2)."""
    poly = part["poly"]
    circles, poly_holes = part_holes(part)
    size = np.ptp(np.vstack(poly), axis=0).max()
    verts = refine_vertices(part)
    ells = {i: corner_length(poly, i, circles, poly_holes) for i in range(len(poly))}
    hg = size / 15
    if verts:
        hg = min(hg, min(ells[i] for i in verts) / 2)
    hg = min(hg, max(min_wall(part) / 3, size / 400))      # at least ~3 elements across the thinnest wall
    res = {}
    fixed_arcs, radial_arcs, bearing_arcs, splits = hole_fixture_arcs(part)
    for tag, div in (("coarse", h_coarse_div), ("fine", h_fine_div)):
        hs = {i: ells[i] / div for i in verts}
        circ = [(c, r, splits.get(j, []), 2 * np.pi * r / HOLE_DIVS[tag]) for j, (c, r) in enumerate(circles)]
        m = mesh_polygon(poly, refine=list(hs.items()), h_global=hg, circles=circ, poly_holes=poly_holes)
        s = solve_plane_stress(m, [tuple(map(tuple, e)) for e in part.get("fixed", [])],
                               [(e[0], e[1], tuple(e[2])) for e in part.get("traction", [])],
                               E=part.get("E", 210000.0), nu=part.get("nu", 0.3),
                               fixed_arcs=fixed_arcs, radial_arcs=radial_arcs, bearing_arcs=bearing_arcs)
        res[tag] = {"m": m, "s": s, "h": hs, "ell": ells, "sampler": sampler_from_mesh(m, s),
                    "xy": m.p.T, "vm": s["vm"]}
    return res


def node_value(xy, vm, P):
    return float(vm[np.argmin(np.linalg.norm(xy - P, axis=1))])


def analyse(part, sols, model=None):
    poly = part["poly"]
    beta = interior_angles(poly)
    bcv = bc_vertices(part)
    fixv = fixed_vertices(part)
    circles, poly_holes = part_holes(part)
    rad_default = part.get("corner_radius", 0.5)
    rad_by = {int(k): v for k, v in part.get("corner_radius_by_vertex", {}).items()}
    c, f = sols["coarse"], sols["fine"]
    scale = np.abs(f["vm"]).max()
    rows = []
    for i in sorted(set(refine_vertices(part))):
        P = poly[i]
        v_c, v_f = node_value(c["xy"], c["vm"], P), node_value(f["xy"], f["vm"], P)
        ratio = v_f / max(v_c, 1e-9)
        rising = ratio > RATIO_THRESHOLD and v_f > 1e-3 * scale
        ell = f.get("ell", {}).get(i) if isinstance(f.get("ell"), dict) else None
        if ell is None:
            ell = corner_length(poly, i, circles, poly_holes)
        row = {"vertex": i, "label": f"V{i}", "x": P[0], "y": P[1], "angle_deg": round(beta[i], 1),
               "fe_stress_coarse": v_c, "fe_stress_fine": v_f, "refine_ratio": ratio,
               "singular": bool(rising), "ell": ell}
        if not rising:
            row["type"] = "regular"
        elif beta[i] > CANDIDATE_ANGLE and i not in fixv:
            row["type"] = "sharp re-entrant corner"
            row["williams_lambda"] = williams_lambda(beta[i])
            rho = rad_by.get(i, rad_default)
            row["corner_radius"] = rho
            h = f["h"][i] if isinstance(f["h"], dict) else f["h"][i]
            feats = corner_features(f["sampler"], poly, i, h, ell=ell)
            notes = []
            if not (MODEL_BETA_RANGE[0] <= beta[i] <= MODEL_BETA_RANGE[1]):
                notes.append("corner angle outside trained range (190-330 deg)")
            if not (0.01 <= rho / ell <= 1.0):
                notes.append(f"radius/l = {rho / ell:.3f} outside trained range 0.01-1.0")
            if i in bcv:
                notes.append("a load is applied on an edge touching this corner; the prediction assumes the "
                             "corner field is dominated by the geometry, so treat it as approximate")
            if feats is None:
                notes.append(f"mesh too coarse at corner (need h <= {MAX_H_OVER_ELL:.2f}*l)")
            elif model is not None:
                row["predicted_real_stress"] = predict(model, feats, rho)
            if notes:
                row["note"] = "; ".join(notes)
        else:
            row["type"] = "support / load artefact"
            row["note"] = ("idealised clamp or load edge - not a physical stress; "
                           "read stress away from it or model the real support")
        rows.append(row)
    cols = ["vertex", "label", "x", "y", "angle_deg", "fe_stress_coarse", "fe_stress_fine", "refine_ratio",
            "singular", "ell", "type", "note"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = pd.Series(dtype=bool if c == "singular" else float if c not in ("label", "type", "note") else object)
    return df


def hole_profile(sol, c, r):
    """Nodal von Mises stress around a hole: (angles in deg, stress), sorted by angle."""
    xy, vm = sol["xy"], sol["vm"]
    d = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
    on = np.abs(d - r) < 1e-6 * r + 1e-9
    ang = np.degrees(np.arctan2(xy[on, 1] - c[1], xy[on, 0] - c[0])) % 360
    o = np.argsort(ang)
    return ang[o], vm[on][o]


def scan_holes(part, sols):
    """
    Singularity check around every circular hole.
    * At each place a fixture starts/ends (e.g. edges of a pin contact patch) compare the stress at that
      exact point on the coarse and fine mesh: still rising => singular (boundary-condition artefact).
    * Everywhere else, report the converged peak stress on the hole edge.
    Returns (rows for singular points, per-hole summary list).
    """
    circles = part.get("circles", [])
    _, _, _, splits = hole_fixture_arcs(part)
    kinds = {int(j): bc["type"] for j, bc in part.get("hole_bc", {}).items()}
    rows, summary = [], []
    for j, (c, r) in enumerate(circles):
        c = np.asarray(c, float)
        a_c, v_c = hole_profile(sols["coarse"], c, r)
        a_f, v_f = hole_profile(sols["fine"], c, r)
        if len(a_f) < 8 or len(a_c) < 8:
            continue
        coarse_at = lambda a: float(np.interp(a % 360, a_c, v_c, period=360))
        flagged = []
        for a in sorted({round(x % 360, 6) for x in splits.get(j, [])}):
            k = int(np.argmin(np.abs((a_f - a + 180) % 360 - 180)))
            vf, vc = float(v_f[k]), coarse_at(a)
            ratio = vf / max(vc, 1e-9)
            if ratio > RATIO_THRESHOLD and vf > 1e-3 * np.abs(sols["fine"]["vm"]).max():
                flagged.append(a)
                P = c + r * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
                rows.append({"vertex": np.nan, "label": f"H{j} @ {a:.0f}°", "x": P[0], "y": P[1],
                             "angle_deg": np.nan, "fe_stress_coarse": vc, "fe_stress_fine": vf,
                             "refine_ratio": ratio, "singular": True, "ell": r,
                             "type": "support / load artefact",
                             "note": f"edge of the {kinds.get(j, 'fixture')} contact on hole H{j}: an idealised "
                                     "contact patch with a sharp end. Real contact pressure fades out smoothly; "
                                     "model contact (nonlinear) to resolve it"})
        # converged peak on the hole edge, away from flagged points
        far = np.ones(len(a_f), bool)
        for a in flagged:
            far &= np.abs((a_f - a + 180) % 360 - 180) > 12
        if far.any():
            k = int(np.argmax(np.where(far, v_f, -np.inf)))
            pk, ang = float(v_f[k]), float(a_f[k])
            pc = coarse_at(ang)
            summary.append({"hole": f"H{j}", "fixture": kinds.get(j, "free"), "radius_mm": r,
                            "peak_stress": pk, "at_angle_deg": ang, "coarse_mesh": pc,
                            "change_pct": 100 * (pk - pc) / max(pc, 1e-9),
                            "x": c[0] + r * np.cos(np.radians(ang)), "y": c[1] + r * np.sin(np.radians(ang)),
                            "singular_points": len(flagged)})
    return rows, summary


def max_regular_stress(part, sols, flags):
    """Highest von Mises stress away from all flagged singular points (holes, drawn fillets, etc.)."""
    out = {}
    for tag in ("coarse", "fine"):
        xy, vm = sols[tag]["xy"], sols[tag]["vm"]
        keep = np.ones(len(vm), bool)
        for _, r in flags[flags.singular].iterrows():
            keep &= np.linalg.norm(xy - np.array([r.x, r.y]), axis=1) > 0.3 * r.ell
        j = np.argmax(np.where(keep, vm, -np.inf))
        out[tag] = (float(vm[j]), xy[j])
    return out


# ------------------------------------------------------------------ model
def feature_vector(feats, rho, cols):
    d = dict(feats)
    d["log_rho_over_ell"] = np.log(rho / feats["ell"])
    d["rho_over_ell"] = rho / feats["ell"]
    return np.array([[d[c] for c in cols]])


def predict(model, feats, rho):
    """Model predicts log(peak / |s_ref|); s_ref = hoop stress at 0.2 l, read from the FE field."""
    x = pd.DataFrame(feature_vector(feats, rho, model["cols"]), columns=model["cols"])
    return float(np.exp(model["est"].predict(x)[0]) * abs(feats["s_ref"]))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--ansys", nargs=2, metavar=("COARSE_CSV", "FINE_CSV"))
    ap.add_argument("--h", nargs=2, type=float, metavar=("H_COARSE", "H_FINE"),
                    help="element size at the corners in the two ANSYS meshes (mm)")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    part = load_part(args.part)
    model = joblib.load(args.model)
    if args.ansys:
        sols = {}
        for tag, path, h in zip(("coarse", "fine"), args.ansys, args.h):
            sm, xy, vm = ansys_sampler(path, part["poly"])
            sols[tag] = {"sampler": sm, "xy": xy, "vm": vm, "h": [h] * len(part["poly"])}
    else:
        sols = solve_two_meshes(part)
    out = analyse(part, sols, model)
    pd.set_option("display.width", 200)
    print(f"\n=== {part.get('name', args.part)} ===")
    print(out.drop(columns=["x", "y"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
