"""
corner_tools.py - singularity physics shared by the dataset generator and the tool.

* williams_lambda(beta)     : Williams mode-I eigenvalue for a sharp corner with material angle beta
* candidate_vertices(poly)  : polygon vertices that can be singular (re-entrant corners, support ends)
* corner_length(poly, i)    : local length scale l = distance from the corner to the nearest non-adjacent edge
* bisector_profile(...)     : hoop / shear / von Mises stress sampled along the corner bisector
* corner_features(...)      : mesh-independent descriptors (K1, T, mode mix) used by the ML model
"""
import numpy as np
from scipy.optimize import brentq
from matplotlib.tri import Triangulation, LinearTriInterpolator

from fe2d import interior_angles


def williams_lambda(beta_deg):
    """Mode-I eigenvalue: sin(2*alpha*lam) + lam*sin(2*alpha) = 0 with 2*alpha = material angle."""
    two_alpha = np.radians(beta_deg)
    if beta_deg <= 180.0 + 1e-9:
        return 1.0                                   # convex or flat corner: no singularity
    f = lambda lam: np.sin(two_alpha * lam) + lam * np.sin(two_alpha)
    return brentq(f, 0.5 + 1e-9, 1.0 - 1e-9)


def _point_segment_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def corner_length(poly, i, circles=None, poly_holes=None):
    """Local length scale l: distance from vertex i to the nearest boundary that does not touch it."""
    poly = np.asarray(poly, float)
    n = len(poly)
    P = poly[i]
    d = [_point_segment_dist(P, poly[j], poly[(j + 1) % n])
         for j in range(n) if j not in (i, (i - 1) % n)]
    for c, r in circles or []:
        d.append(np.linalg.norm(P - np.asarray(c, float)) - r)
    for hole in poly_holes or []:
        h = np.asarray(hole, float)
        d += [_point_segment_dist(P, h[k], h[(k + 1) % len(h)]) for k in range(len(h))]
    return min(d)


def material_bisector(poly, i):
    poly = np.asarray(poly, float)
    n = len(poly)
    P = poly[i]
    ua = poly[i - 1] - P
    ub = poly[(i + 1) % n] - P
    ua /= np.linalg.norm(ua)
    ub /= np.linalg.norm(ub)
    b = ua + ub
    b /= np.linalg.norm(b)
    beta = interior_angles(poly)[i]
    return -b if beta > 180 else b                  # unit vector pointing into the material


def candidate_vertices(poly):
    """All vertices with interior angle > 180 deg (geometric singularities)."""
    beta = interior_angles(poly)
    return [i for i, b in enumerate(beta) if b > 180.5]


class FieldSampler:
    """Linear interpolation of averaged nodal stresses (same as probing an ANSYS contour)."""

    def __init__(self, xy, stresses):
        self.tri = Triangulation(xy[:, 0], xy[:, 1], stresses["tri"]) if "tri" in stresses else None
        self.interp = {k: LinearTriInterpolator(self.tri, stresses[k]) for k in ("sx", "sy", "txy", "vm")}

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        return {k: np.asarray(f(pts[:, 0], pts[:, 1]).filled(np.nan)) for k, f in self.interp.items()}


def sampler_from_mesh(m, s):
    st = dict(s)
    st["tri"] = m.t.T
    return FieldSampler(m.p.T, st)


def bisector_profile(sampler, poly, i, r):
    P = np.asarray(poly[i], float)
    er = material_bisector(poly, i)
    et = np.array([-er[1], er[0]])
    pts = P + np.outer(r, er)
    v = sampler(pts)
    S = np.stack([[v["sx"], v["txy"]], [v["txy"], v["sy"]]])           # (2,2,n)
    s_tt = np.einsum("i,ijn,j->n", et, S, et)                            # hoop stress
    t_rt = np.einsum("i,ijn,j->n", er, S, et)                            # shear stress
    return s_tt, t_rt, v["vm"]


PROFILE_R = (0.1, 0.15, 0.2, 0.3, 0.4)    # sampling distances along the bisector, as fractions of l
FIT_WINDOW = (0.1, 0.3)                    # K1 fitting window (fractions of l) - mesh-stable for h <= l/10
MAX_H_OVER_ELL = 0.12                      # coarser than this near the corner -> refine first


def corner_features(sampler, poly, i, h_local, ell=None):
    """
    Mesh-independent description of the singular field at corner i.
    Only uses stresses at r >= 0.1*l, which converge on any mesh with h <= l/10,
    and never the corner node itself (that value is meaningless - it grows forever).

    K1, T : two-term Williams fit  s_tt(r) = K1 * r^(lam-1) + T  over r in [0.1 l, 0.3 l]
    p_*   : hoop / shear / von Mises stress at r = PROFILE_R * l, normalised by s_ref = von Mises(0.2 l)
            (von Mises never passes through zero, unlike the hoop stress under shear-dominated loads)
    """
    beta = interior_angles(poly)[i]
    lam = williams_lambda(beta)
    if ell is None:
        ell = corner_length(poly, i)
    if h_local / ell > MAX_H_OVER_ELL:
        return None
    r = np.geomspace(FIT_WINDOW[0] * ell, FIT_WINDOW[1] * ell, 15)
    s_tt, _, _ = bisector_profile(sampler, poly, i, r)
    if not np.all(np.isfinite(s_tt)):
        return None
    A = np.stack([r ** (lam - 1), np.ones_like(r)], 1)
    (K1, T), *_ = np.linalg.lstsq(A, s_tt, rcond=None)
    fit_res = np.sqrt(np.mean((A @ [K1, T] - s_tt) ** 2)) / (np.abs(s_tt).mean() + 1e-12)

    rp = np.array(PROFILE_R) * ell
    p_tt, p_rt, p_vm = bisector_profile(sampler, poly, i, rp)
    s_ref = bisector_profile(sampler, poly, i, np.array([0.2 * ell]))[2][0]
    feats = {
        "beta": beta, "lam": lam, "ell": ell, "h": h_local, "h_over_ell": h_local / ell,
        "K1": K1, "T": T, "s_ref": s_ref,
        "K1_hat": K1 * ell ** (lam - 1) / s_ref,           # dimensionless singular amplitude
        "T_hat": T / s_ref,
        "fit_residual": fit_res,
    }
    for rr, a_, b_, c_ in zip(PROFILE_R, p_tt, p_rt, p_vm):
        feats[f"ptt_{rr}"] = a_ / s_ref
        feats[f"prt_{rr}"] = abs(b_) / abs(s_ref)
        feats[f"pvm_{rr}"] = c_ / abs(s_ref)
    return feats


def detect_singularities(poly, s_coarse, m_coarse, s_fine, m_fine, threshold=1.05):
    """
    Numerical singularity check at every polygon vertex: stress at the vertex node on the
    coarse and the refined mesh. Ratio > threshold => stress is not converging => singular.
    """
    out = []
    for i, P in enumerate(np.asarray(poly, float)):
        jc = np.argmin(np.linalg.norm(m_coarse.p.T - P, axis=1))
        jf = np.argmin(np.linalg.norm(m_fine.p.T - P, axis=1))
        ratio = s_fine["vm"][jf] / max(s_coarse["vm"][jc], 1e-9)
        out.append({"vertex": i, "vm_coarse": s_coarse["vm"][jc], "vm_fine": s_fine["vm"][jf],
                    "ratio": ratio, "singular": ratio > threshold})
    return out
