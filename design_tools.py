"""
design_tools.py - design answers built on the EigenStress model.

1. required_radius : smallest fillet radius that keeps a sharp corner's real peak stress below an allowable.
   The ML model predicts in milliseconds, so it simply scans 80 radii over the trained range (0.01-1.0 l);
   a conventional FE workflow would need one refined re-solve per trial radius.
2. fatigue          : notch-sensitivity corrected fatigue stress (Peterson) and an S-N life estimate
   (Basquin through 0.9 su at 1e3 and Se at 1e6 cycles, Goodman mean-stress correction for R = 0),
   used to rank every hot spot by which one cracks first.
3. apply_fillets    : put the required radii into the outline / cutouts and export a DXF for checking.

References: Peterson's Stress Concentration Factors (notch sensitivity q = 1 / (1 + a/r));
Shigley's Mechanical Engineering Design (S-N line through 0.9 su at 1e3 cycles and Se at 1e6, Goodman).
"""
import numpy as np
import pandas as pd

from sing_tool import feature_vector

N_SCAN = 80


# ------------------------------------------------------------------ 1. required radius
def predict_many(model, feats, rhos):
    X = pd.DataFrame(np.vstack([feature_vector(feats, r, model["cols"]) for r in rhos]), columns=model["cols"])
    return np.exp(model["est"].predict(X)) * abs(feats["s_ref"])


def required_radius(model, feats, allow, r_now=None):
    """Smallest radius r (within 0.01-1.0 l) with predicted peak <= allow for r and every larger radius scanned.
    Returns (r_required, status, peak_at_r_required). status: 'ok' | 'any' (even 0.01 l is enough) |
    'impossible' (still above allow at r = l: a fillet alone cannot fix it - add material or reduce load)."""
    ell = feats["ell"]
    rhos = np.exp(np.linspace(np.log(0.01 * ell), np.log(1.0 * ell), N_SCAN))
    pk = predict_many(model, feats, rhos)
    ok = pk <= allow
    if ok.all():
        return float(rhos[0]), "any", float(pk[0])
    if not ok[-1]:
        return np.nan, "impossible", float(pk[-1])
    # first index from which everything larger is OK
    last_bad = int(np.where(~ok)[0].max())
    j = last_bad + 1
    # refine between rhos[j-1] and rhos[j] by interpolation in log space
    a, b = np.log(rhos[j - 1]), np.log(rhos[j])
    pa, pb = pk[j - 1], pk[j]
    t = (pa - allow) / max(pa - pb, 1e-12)
    r = float(np.exp(a + np.clip(t, 0, 1) * (b - a)))
    return r, "ok", float(allow)


# ------------------------------------------------------------------ 2. fatigue
def peterson_a(su):
    """Peterson's material constant a (mm) for steels, from the ultimate strength su (MPa):
    a = 0.0254 (2070 / su)^1.8 mm (Shigley's fit, valid roughly for su = 340-1700 MPa)."""
    return 0.0254 * (2070.0 / max(su, 1.0)) ** 1.8


def notch_sensitivity(rho, su):
    """q = 1 / (1 + a / rho). rho = notch root radius in mm."""
    return 1.0 / (1.0 + peterson_a(su) / max(rho, 1e-9))


def sn_life(sig_a, su, se):
    """Cycles to failure for a fully reversed equivalent amplitude sig_a (MPa). S-N line through
    (1e3, 0.9 su) and (1e6, se). Below se -> inf (infinite life); above 0.9 su -> < 1e3 (low-cycle)."""
    if sig_a <= se:
        return np.inf
    f = 0.9 * su
    b = -np.log10(f / se) / 3.0
    a = f ** 2 / se
    return float((sig_a / a) ** (1.0 / b))


def fatigue(peak, nominal, rho, su, se, cycle="R=-1"):
    """Fatigue assessment of one hot spot under the analysed load applied cyclically.
    peak    : real elastic peak stress (MPa) at the notch root (ML prediction or converged FE peak)
    nominal : stress away from the notch (MPa), so Kt = peak / nominal
    rho     : notch root radius (mm); q from Peterson
    cycle   : 'R=-1' fully reversed (load swings +F to -F) or 'R=0' zero-to-max (0 to +F)
    Returns dict with q, Kt, Kf, fatigue stress, equivalent amplitude, life (cycles)."""
    nominal = max(abs(nominal), 1e-9)
    kt = max(peak / nominal, 1.0)
    q = notch_sensitivity(rho, su)
    kf = 1.0 + q * (kt - 1.0)
    s_eff = kf * nominal                                  # fatigue-effective peak (<= elastic peak)
    if cycle == "R=0":
        sa = sm = s_eff / 2.0
        sar = sa / max(1.0 - sm / su, 1e-6)               # Goodman: equivalent fully reversed amplitude
    else:
        sar = s_eff
    n = sn_life(sar, su, se)
    return {"q": q, "Kt": kt, "Kf": kf, "fatigue_stress": s_eff, "eq_amplitude": sar, "life_cycles": n}


def life_text(n):
    if not np.isfinite(n):
        return "infinite (> 10⁶)"
    if n < 1e3:
        return "< 10³ (low-cycle)"
    e = int(np.floor(np.log10(n)))
    return f"{n / 10 ** e:.1f}×10{str(e).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


# ------------------------------------------------------------------ 3. apply radii -> DXF
def _arc_points(prev, P, nxt, r, max_turn_deg=8.0):
    from fe2d import fillet_points
    t1, cen, t2 = fillet_points(prev, P, nxt, r)
    a1 = np.arctan2(t1[1] - cen[1], t1[0] - cen[0])
    a2 = np.arctan2(t2[1] - cen[1], t2[0] - cen[0])
    da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    k = max(int(np.ceil(abs(np.degrees(da)) / max_turn_deg)), 2)
    return [tuple(cen + r * np.array([np.cos(a1 + da * u), np.sin(a1 + da * u)])) for u in np.linspace(0, 1, k + 1)]


def apply_fillets(ring, radii):
    """ring: list of points; radii: {vertex index: radius}. Returns the ring with those vertices replaced by arcs."""
    out = []
    n = len(ring)
    for i, P in enumerate(ring):
        r = radii.get(i)
        if r and np.isfinite(r) and r > 0:
            out += _arc_points(np.asarray(ring[i - 1], float), np.asarray(P, float),
                               np.asarray(ring[(i + 1) % n], float), float(r))
        else:
            out.append(tuple(map(float, P)))
    return out


def max_fillet(ring, i):
    """Largest radius whose tangent points stay within 45 % of both neighbouring edges."""
    from fe2d import interior_angles
    P = np.asarray(ring[i], float)
    a, c = np.asarray(ring[i - 1], float), np.asarray(ring[(i + 1) % len(ring)], float)
    beta = interior_angles([np.asarray(p, float) for p in ring])[i]
    psi = np.radians(abs(360.0 - beta)) if beta > 180 else np.radians(beta)
    adj = min(np.linalg.norm(P - a), np.linalg.norm(c - P))
    return 0.45 * adj * np.tan(psi / 2)
