"""
ansys_io.py - use stress results from ANSYS instead of the built-in solver.

Accepted files
  * ANSYS result file (.rst) from Mechanical (Workbench: <project>_files/dp0/SYS/MECH/file.rst) or MAPDL.
    Read with ansys-dpf-core (needs ANSYS installed, e.g. 2025 R1) or, without ANSYS, with ansys-mapdl-reader.
  * CSV with node coordinates and stresses: two of X / Y / Z, the in-plane normal and shear stresses
    (e.g. SX, SY, SXY; SEQV optional). Units mm + MPa, or m + Pa (detected and converted).
    Header names are matched loosely ("X Location (mm)", "Normal Stress X", "sxy", ...).

2D (plane stress) and 3D models both work. For a 3D plate model the face that matches the DXF is found
automatically (XY, XZ or YZ) and the nodes on the outermost face in the thickness direction are used.
Coordinates are shifted onto the DXF if the ANSYS model was built at a different origin.

Upload one result (fine mesh) - every sharp re-entrant corner is then treated as singular from its geometry -
or two (coarse + fine) so EigenStress can also confirm singularity by mesh refinement, as with the built-in solver.
"""
import io
import os
import re
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.path import Path


# ------------------------------------------------------------------ reading
def _vm3(s):
    sx, sy, sz, sxy, syz, sxz = (s[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + sxz ** 2))


def _read_rst_dpf(path):
    from ansys.dpf import core as dpf
    model = dpf.Model(path)
    nsets = model.metadata.time_freq_support.n_sets
    op = model.results.stress()
    op.inputs.requested_location(dpf.locations.nodal)          # averaged nodal stresses
    op.inputs.time_scoping([nsets])                             # last load step
    f = op.outputs.fields_container()[0]
    ids = np.asarray(f.scoping.ids)
    data = np.asarray(f.data, float)                            # SX SY SZ SXY SYZ SXZ
    nodes = model.metadata.meshed_region.nodes
    cf = nodes.coordinates_field
    cid = np.asarray(cf.scoping.ids)
    cxyz = np.asarray(cf.data, float)
    pos = {n: i for i, n in enumerate(cid)}
    xyz = cxyz[[pos[n] for n in ids]]
    return xyz, data


def _read_rst_reader(path):
    from ansys.mapdl import reader as pymapdl_reader
    rst = pymapdl_reader.read_binary(path)
    nnum, st = rst.nodal_stress(rst.nsets - 1)                  # SX SY SZ SXY SYZ SXZ, averaged; NaN at mid-side
    pos = {n: i for i, n in enumerate(rst.mesh.nnum)}
    xyz = np.asarray(rst.mesh.nodes, float)[[pos[n] for n in nnum]]
    return xyz, np.asarray(st, float)


def read_rst(data, name="file.rst"):
    """-> DataFrame X, Y, Z, SX, SY, SZ, SXY, SYZ, SXZ (all nodes with results)."""
    td = tempfile.mkdtemp()
    path = os.path.join(td, os.path.basename(name) or "file.rst")
    with open(path, "wb") as fh:
        fh.write(data)
    errors = []
    for reader in (_read_rst_dpf, _read_rst_reader):
        try:
            xyz, s = reader(path)
            break
        except Exception as e:                                  # library missing / no DPF server / format
            errors.append(f"{reader.__name__[10:]}: {e}")
    else:
        raise RuntimeError("could not read the .rst file. Install ansys-dpf-core (with ANSYS installed) or "
                           "ansys-mapdl-reader, or export the results as CSV instead.\n" + "\n".join(errors))
    ok = np.isfinite(s).all(1)
    df = pd.DataFrame(np.column_stack([xyz[ok], s[ok]]),
                      columns=["X", "Y", "Z", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ"])
    return df


_ALIASES = {
    "X": [r"^x$", r"x\s*loc", r"^loc\s*x", r"^x\s*\("],
    "Y": [r"^y$", r"y\s*loc", r"^loc\s*y", r"^y\s*\("],
    "Z": [r"^z$", r"z\s*loc", r"^loc\s*z", r"^z\s*\("],
    "SX": [r"^sx$", r"^s_?x$", r"normal\s*stress.*\bx\b", r"^sigma_?x"],
    "SY": [r"^sy$", r"^s_?y$", r"normal\s*stress.*\by\b", r"^sigma_?y"],
    "SZ": [r"^sz$", r"^s_?z$", r"normal\s*stress.*\bz\b", r"^sigma_?z"],
    "SXY": [r"^sxy$", r"^s_?xy$", r"shear\s*stress.*\bxy\b", r"^tau_?xy"],
    "SYZ": [r"^syz$", r"shear\s*stress.*\byz\b"],
    "SXZ": [r"^sxz$", r"^szx$", r"shear\s*stress.*\b(xz|zx)\b"],
    "SEQV": [r"^seqv$", r"equivalent", r"von\s*mises", r"^vm$"],
}


def read_csv(data):
    txt = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    sep = "\t" if txt.count("\t") > txt.count(",") else ","
    raw = pd.read_csv(io.StringIO(txt), sep=sep)
    out = {}
    for key, pats in _ALIASES.items():
        for c in raw.columns:
            cl = str(c).strip().lower()
            if any(re.search(p, cl) for p in pats) and key not in out:
                out[key] = pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
    coords = [k for k in "XYZ" if k in out]
    normals = [k for k in ("SX", "SY", "SZ") if k in out]
    if len(coords) < 2 or len(normals) < 2:
        raise ValueError("CSV needs node coordinates (two of X / Y / Z) and stress components (at least two "
                         f"normal stresses, plus the in-plane shear). Found: {', '.join(map(str, raw.columns))}")
    n = len(raw)
    for k in ("X", "Y", "Z", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ"):
        out.setdefault(k, np.zeros(n))                          # missing coordinate / component -> 0
    df = pd.DataFrame(out)
    df = df[np.isfinite(df[coords + normals]).all(1)].reset_index(drop=True)
    return df


def read_result(data, name):
    return read_rst(data, name) if name.lower().endswith(".rst") else read_csv(data)


# ------------------------------------------------------------------ to the DXF plane
def to_plane(df, poly, tol=0.03):
    """Pick the coordinate plane whose node bounding box matches the DXF outline, keep the outer face of a 3D
    model, shift onto the DXF origin. Returns (DataFrame X, Y, SX, SY, SXY, SEQV, info text)."""
    P = np.vstack([np.asarray(p, float) for p in poly])
    w, h = np.ptp(P, axis=0)
    xyz0 = df[["X", "Y", "Z"]].to_numpy(float)
    names = "XYZ"
    best = None
    for unit, k in (("mm", 1.0), ("m", 1000.0)):               # a model built in metres -> coordinates x1000
        xyz_k = xyz0 * k
        for a, b in ((0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)):
            wa, hb = np.ptp(xyz_k[:, a]), np.ptp(xyz_k[:, b])
            err = max(abs(wa - w) / max(w, 1e-9), abs(hb - h) / max(h, 1e-9))
            if best is None or err < best[0]:
                best = (err, a, b, unit, k)
    err, a, b, unit, k = best
    xyz = xyz0 * k
    if unit == "m":                                             # SI model: metres and Pa -> mm and MPa
        df = df.copy()
        for c_ in ("SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "SEQV"):
            if c_ in df:
                df[c_] = df[c_] / 1e6
        df[["X", "Y", "Z"]] = xyz
    if err > tol:
        raise ValueError(f"the ANSYS model ({np.ptp(xyz[:, 0]):.1f} x {np.ptp(xyz[:, 1]):.1f} x "
                         f"{np.ptp(xyz[:, 2]):.1f} mm) does not match the DXF outline ({w:.1f} x {h:.1f} mm) in any "
                         "plane. Check units (mm) and that the DXF is the same part.")
    c = 3 - a - b
    info = [f"plane {names[a]}{names[b]}"] + (["model in m / Pa: converted to mm / MPa"] if unit == "m" else [])
    keep = np.ones(len(df), bool)
    span = np.ptp(xyz[:, c])
    if span > 1e-6 * max(w, h):                                 # 3D: use the outer face (max of thickness axis)
        keep = xyz[:, c] >= xyz[:, c].max() - 1e-4 * max(w, h)
        info.append(f"3D model {span:.2f} mm thick: using the face at {names[c]} = {xyz[:, c].max():.2f}")
    comp = {(0, 1): "SXY", (1, 0): "SXY", (0, 2): "SXZ", (2, 0): "SXZ", (1, 2): "SYZ", (2, 1): "SYZ"}
    sa, sb = ("SX", "SY", "SZ")[a], ("SX", "SY", "SZ")[b]
    d = df[keep]
    u = d[names[a]].to_numpy(float)
    v = d[names[b]].to_numpy(float)
    du, dv = P[:, 0].min() - u.min(), P[:, 1].min() - v.min()
    if max(abs(du), abs(dv)) > 1e-3 * max(w, h):
        info.append(f"shifted by ({du:+.2f}, {dv:+.2f}) mm onto the DXF")
    s6 = d[["SX", "SY", "SZ", "SXY", "SYZ", "SXZ"]].to_numpy(float)
    if "SEQV" in d and np.isfinite(d["SEQV"].to_numpy(float)).all():
        vm = d["SEQV"].to_numpy(float)                          # ANSYS's own equivalent stress
    else:
        vm = _vm3(s6)                                           # 3D von Mises (= plane-stress value in 2D)
    out = pd.DataFrame({"X": u + du, "Y": v + dv, "SX": d[sa].to_numpy(float), "SY": d[sb].to_numpy(float),
                        "SXY": d[comp[(a, b)]].to_numpy(float), "SEQV": vm})
    # merge duplicate coordinates (shared nodes listed twice in some exports)
    out = out.groupby([out.X.round(9), out.Y.round(9)], as_index=False).mean(numeric_only=True)
    return out, "; ".join(info)


# ------------------------------------------------------------------ to an EigenStress solution
def _cloud_mesh(xy, poly, holes):
    from scipy.spatial import Delaunay
    tri = Delaunay(xy).simplices
    cen = xy[tri].mean(1)
    keep = Path(np.vstack(poly)).contains_points(cen, radius=1e-9)
    for h in holes:
        keep &= ~Path(np.vstack(h)).contains_points(cen)
    u, v = xy[tri[:, 1]] - xy[tri[:, 0]], xy[tri[:, 2]] - xy[tri[:, 0]]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    keep &= area > 1e-8 * np.median(area)
    t = tri[keep]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    return SimpleNamespace(p=xy.T.copy(), t=t.T.copy(), facets=np.unique(e, axis=0).T)


def solution_from(df, part):
    """One EigenStress solution dict (same keys as the built-in solver) from a node table on the DXF plane."""
    from corner_tools import FieldSampler, corner_length
    from sing_tool import part_holes, refine_vertices, cutout_corners
    from scipy.spatial import cKDTree
    poly = part["poly"]
    circles, holes = part_holes(part)
    circ_rings = [[(c[0] + r * np.cos(t), c[1] + r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
                  for c, r in circles]
    xy = df[["X", "Y"]].to_numpy(float)
    m = _cloud_mesh(xy, poly, list(holes) + circ_rings)
    s = {"sx": df.SX.to_numpy(float), "sy": df.SY.to_numpy(float), "txy": df.SXY.to_numpy(float),
         "vm": df.SEQV.to_numpy(float)}
    st = dict(s)
    st["tri"] = m.t.T
    sampler = FieldSampler(xy, st)
    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]

    def local_h(P, ell):
        idx = tree.query_ball_point(P, 0.3 * ell)
        return float(np.median(nn[idx])) if len(idx) >= 3 else float(np.median(nn))

    ells = {i: corner_length(poly, i, circles, holes) for i in range(len(poly))}
    hs = {i: local_h(np.asarray(poly[i], float), ells[i]) for i in refine_vertices(part)}
    hh = {(k, i): local_h(np.asarray(holes[k][i], float), ell) for k, i, _, ell in cutout_corners(part)}
    return {"m": m, "s": s, "h": hs, "ell": ells, "hole_h": hh, "sampler": sampler, "xy": xy, "vm": s["vm"],
            "external": True}


def sols_from_results(part, fine, coarse=None):
    """fine / coarse: (bytes, filename). Returns (sols, info). With one result, sols['geometric_only'] = True."""
    info = []
    out = {}
    for tag, f in (("fine", fine), ("coarse", coarse)):
        if f is None:
            continue
        df, txt = to_plane(read_result(*f), part["poly"])
        out[tag] = solution_from(df, part)
        info.append(f"{tag}: {f[1]} · {len(df):,} nodes · {txt}")
    if "coarse" not in out:
        out["coarse"] = out["fine"]
        out["geometric_only"] = True
    return out, info
