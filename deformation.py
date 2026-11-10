"""
deformation.py - what stays finite at a sharp corner.

Near a sharp re-entrant corner (Williams, mode I, material angle beta = 2*alpha):

    stress        ~ K1 r^(lam-1)      -> infinite at the tip, never converges in FE
    displacement  ~ K1 r^lam / 2mu    -> finite, goes smoothly to zero, converges
    strain energy ~ int r^(2lam-2) r dr  finite because lam > 0, converges

So deformation can be trusted where stress cannot. This module:

* williams_opening_coeff : closed-form flank-opening displacement per unit K1 (hand derivation, Airy function)
* corner_opening         : measure the flank opening delta(r) in the FE displacements and fit
                           delta = a r^lam + b r  ->  K1 from DISPLACEMENT, plus a free-exponent fit -> lam_u
* deformation_summary    : max displacement and strain energy on the two meshes (convergence check)
* fringe_image           : synthetic photoelasticity (stress-optic law, circular / plane polariscope,
                           monochromatic or white light)
"""
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator

from corner_tools import material_bisector, williams_lambda
from fe2d import interior_angles


# --------------------------------------------------------------------------- Williams displacement field
def williams_mode1(lam, alpha):
    """Airy stress function phi = r^(lam+1) [A cos((lam+1)th) + B cos((lam-1)th)], traction-free at
    th = +-alpha, normalised so that the hoop stress on the bisector is s_tt(r, 0) = 1 * r^(lam-1)."""
    A = 1.0
    B = -A * np.cos((lam + 1) * alpha) / np.cos((lam - 1) * alpha)
    k = lam * (lam + 1) * (A + B)                  # s_tt(r,0) = lam(lam+1)(A+B) r^(lam-1)
    return A / k, B / k


def williams_displacement(lam, alpha, th, nu):
    """2*mu*u_r / r^lam and 2*mu*u_th / r^lam for unit K1 (plane stress, kappa = (3-nu)/(1+nu))."""
    A, B = williams_mode1(lam, alpha)
    kap = (3 - nu) / (1 + nu)
    ur = -(lam + 1) * A * np.cos((lam + 1) * th) + (kap - lam) * B * np.cos((lam - 1) * th)
    ut = (lam + 1) * A * np.sin((lam + 1) * th) + (kap + lam) * B * np.sin((lam - 1) * th)
    return ur, ut


def williams_opening_coeff(beta_deg, E, nu):
    """Flank opening delta(r) = c * K1 * r^lam, measured perpendicular to the bisector between the two flank
    points at distance r. Rigid translation and rotation cancel in this difference, and so does the
    antisymmetric (mode II) field, so delta isolates mode I."""
    lam = williams_lambda(beta_deg)
    alpha = np.radians(beta_deg) / 2
    mu = E / (2 * (1 + nu))
    ur, ut = williams_displacement(lam, alpha, alpha, nu)
    return 2 * (ur * np.sin(alpha) + ut * np.cos(alpha)) / (2 * mu), lam


# --------------------------------------------------------------------------- FE measurement
class DispSampler:
    def __init__(self, m, s):
        T = Triangulation(m.p[0], m.p[1], m.t.T)
        self.fx = LinearTriInterpolator(T, s["ux"])
        self.fy = LinearTriInterpolator(T, s["uy"])

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        return np.stack([np.asarray(self.fx(pts[:, 0], pts[:, 1]).filled(np.nan)),
                         np.asarray(self.fy(pts[:, 0], pts[:, 1]).filled(np.nan))], 1)


OPEN_WINDOW = (0.1, 0.4)       # fractions of l: resolved once h <= l/10 near the corner


def corner_opening(dsamp, ring, i, ell, E, nu, window=OPEN_WINDOW, n=20):
    """K1 from displacement at corner i of ring. Returns dict with K1_u, lam_u (free fit), the opening
    profile and the fit residual, or None if the flanks cannot be sampled."""
    ring = np.asarray(ring, float)
    beta = interior_angles(ring)[i]
    if beta <= 180.5:
        return None
    c, lam = williams_opening_coeff(beta, E, nu)
    P = ring[i]
    er = material_bisector(ring, i)
    et = np.array([-er[1], er[0]])
    flanks = []
    for nb in (ring[i - 1], ring[(i + 1) % len(ring)]):
        t = nb - P
        L = np.linalg.norm(t)
        flanks.append((t / L, L))
    flanks.sort(key=lambda f: -(f[0] @ et))          # P+ (towards +et) first
    (tp, Lp), (tm, Lm) = flanks
    r = np.geomspace(window[0] * ell, min(window[1] * ell, 0.9 * Lp, 0.9 * Lm), n)
    # sample just inside the material: step a hair towards the bisector
    nudge = 0.01 * r[:, None]
    up = dsamp(P + r[:, None] * tp + nudge * (er - (er @ tp) * tp) / max(np.linalg.norm(er - (er @ tp) * tp), 1e-12))
    um = dsamp(P + r[:, None] * tm + nudge * (er - (er @ tm) * tm) / max(np.linalg.norm(er - (er @ tm) * tm), 1e-12))
    delta = (up - um) @ et
    ok = np.isfinite(delta)
    if ok.sum() < 6:
        return None
    r, delta = r[ok], delta[ok]
    # delta = a r^lam + b r + c r^2 : singular term + the smooth (bending / T-stress) part of the opening
    A = np.stack([r ** lam, r, r ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, delta, rcond=None)
    K1_u = coef[0] / c
    res = np.sqrt(np.mean((A @ coef - delta) ** 2)) / (np.abs(delta).mean() + 1e-30)
    # the exponent measured from deformation alone: the lam that best fits the same three-term form
    grid = np.linspace(0.3, 0.98, 137)
    err = []
    for lg in grid:
        Ag = np.stack([r ** lg, r, r ** 2], 1)
        cg, *_ = np.linalg.lstsq(Ag, delta, rcond=None)
        err.append(np.sum((Ag @ cg - delta) ** 2))
    k = int(np.argmin(err))
    lam_u = grid[k]
    if 0 < k < len(grid) - 1:                         # parabolic refinement of the minimum
        y0, y1, y2 = err[k - 1], err[k], err[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            lam_u = grid[k] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
    return {"K1_u": K1_u, "lam": lam, "lam_u": lam_u, "coeff": c, "fit_residual": res,
            "r": r, "delta": delta, "opening_at_01l": float(np.interp(0.1 * ell, r, delta))}


def deformation_summary(sols):
    """Max |u| and strain energy on both meshes: the quantities that converge even at a singular corner."""
    out = {}
    for tag in ("coarse", "fine"):
        s = sols[tag]["s"]
        if "ux" not in s:
            return None
        umag = np.hypot(s["ux"], s["uy"])
        j = int(np.argmax(umag))
        out[tag] = {"umax": float(umag[j]), "at": sols[tag]["m"].p[:, j].copy(), "energy": s.get("energy", np.nan),
                    "work": s.get("work", np.nan)}
    return out


# --------------------------------------------------------------------------- photoelasticity
# typical fringe constants f_sigma (N/mm per fringe, ~ 550-575 nm light)
PHOTO_MATERIALS = {"Polycarbonate (PSM-1 type)": 7.0, "Epoxy (Araldite type)": 10.5,
                   "Homalite 100 (polyester)": 23.6, "Urethane rubber": 0.18}
LAMBDA0 = 560.0          # nm, wavelength the fringe constant is quoted at


def _cie_xyz(wl):
    """Wyman, Sloan & Shirley (2013) multi-lobe fit of the CIE 1931 colour matching functions."""
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    x = 1.056 * g(wl, 599.8, 37.9, 31.0) + 0.362 * g(wl, 442.0, 16.0, 26.7) - 0.065 * g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * g(wl, 568.8, 46.9, 40.5) + 0.286 * g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * g(wl, 437.0, 11.8, 36.0) + 0.681 * g(wl, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


_WL = np.linspace(400, 700, 46)
_XYZ = _cie_xyz(_WL)
_M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def isochromatic_rgb(N, white=True, dark_field=True):
    """Light intensity behind a circular polariscope for fringe order N (at LAMBDA0).
    dark field: I = sin^2(pi N), light field: cos^2(pi N). White light sums over the visible spectrum,
    with N scaling as LAMBDA0 / wavelength (dispersion of birefringence neglected)."""
    N = np.asarray(N, float)
    trig = np.sin if dark_field else np.cos
    if not white:
        I = trig(np.pi * N) ** 2
        return np.stack([I, I, I], -1)
    phase = np.pi * N[..., None] * (LAMBDA0 / _WL)
    I = trig(phase) ** 2                                   # (..., n_wl)
    xyz = I @ _XYZ / _XYZ[:, 1].sum()
    rgb = xyz @ _M.T
    rgb = np.clip(rgb, 0, None)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return np.clip(rgb * 1.15, 0, 1)


def principal(sx, sy, txy):
    c = 0.5 * (sx + sy)
    R = np.sqrt((0.5 * (sx - sy)) ** 2 + txy ** 2)
    phi = 0.5 * np.arctan2(2 * txy, sx - sy)              # angle of sigma_1 from x
    return c + R, c - R, phi


def fringe_rgb(sx, sy, txy, t_mm, f_sigma, load_factor, white=True, polariscope="circular", dark_field=True,
               analyser_deg=0.0):
    """Stress-optic law on gridded stresses (NaN outside the part). N = t (s1 - s2) / f_sigma, with stresses
    scaled by load_factor (the photoelastic model carries load_factor x the real load)."""
    mask = ~np.isfinite(sx)
    sx, sy, txy = (np.nan_to_num(v) * load_factor for v in (sx, sy, txy))
    s1, s2, phi = principal(sx, sy, txy)
    N = t_mm * (s1 - s2) / f_sigma
    rgb = isochromatic_rgb(N, white=white, dark_field=dark_field)
    if polariscope == "plane":                             # isoclinics: extinction where sigma_1 // polariser
        iso = np.sin(2 * (phi - np.radians(analyser_deg))) ** 2
        rgb = rgb * iso[..., None] if dark_field else 1 - (1 - rgb) * iso[..., None]
    rgb[mask] = np.nan
    return rgb, np.where(mask, np.nan, N)


def grid_fields(m, s, n_px=700, bounds=None, keys=("sx", "sy", "txy")):
    """Interpolate nodal fields of an FE solution onto a regular image grid (NaN outside the mesh)."""
    T = Triangulation(m.p[0], m.p[1], m.t.T)
    x0, x1 = m.p[0].min(), m.p[0].max()
    y0, y1 = m.p[1].min(), m.p[1].max()
    if bounds is not None:
        x0, x1, y0, y1 = bounds
    ny = max(int(n_px * (y1 - y0) / max(x1 - x0, 1e-12)), 2)
    gx, gy = np.linspace(x0, x1, n_px), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(gx, gy)
    out = {k: np.ma.filled(LinearTriInterpolator(T, s[k])(X, Y), np.nan) for k in keys}
    return out, (x0, x1, y0, y1)


def fringe_order(stress_diff, t_mm, f_sigma, load_factor):
    return t_mm * stress_diff * load_factor / f_sigma
