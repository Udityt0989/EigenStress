"""
make_dataset.py - paired training data: sharp-corner FE features  ->  real (filleted) peak stress.

Geometry family: "angled bracket" - two legs of width a meeting at an angle psi, so the re-entrant
corner has material angle beta = 360 - psi in [200, 300] deg. One leg is clamped, the other carries
a uniform traction in a random direction (mixes opening and shearing of the corner).

Per geometry:
  * sharp corner, 4 mesh sizes (l/10 ... l/80)  -> features (what the tool will see)
  * filleted corner, 3 radii, converged mesh     -> target peak von Mises stress
Rows = every (sharp mesh, radius) pair, so each target appears with coarse and fine meshes.

Usage: python make_dataset.py --n 600 --out dataset.csv --workers 2
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fe2d import mesh_polygon, solve_plane_stress, signed_area
from corner_tools import corner_features, sampler_from_mesh, corner_length

MESH_DIVS = (10, 20, 40, 80)
FAMILY = "bracket"
RHO_RANGE = (0.01, 0.3)          # fillet radius / l sampled for the targets
N_RADII = 3


def angled_bracket(a, L1, L2, psi_deg):
    psi = np.radians(psi_deg)
    u = np.array([np.sin(psi), np.cos(psi)])          # leg-2 direction (leg 1 runs along +y)
    nrm = np.array([np.cos(psi), -np.sin(psi)])       # leg-2 outward offset direction
    t = -a * (1 + np.cos(psi)) / np.sin(psi)
    Q = np.array([-a, -a * np.sin(psi) + t * np.cos(psi)])   # outer corner
    P = np.zeros(2)
    E2a, E2b = P + L2 * u, P + L2 * u + a * nrm
    poly = [P, E2a, E2b, Q, np.array([-a, L1]), np.array([0.0, L1])]
    if signed_area(poly) < 0:
        poly = poly[::-1]
    idx = int(np.argmin([np.linalg.norm(p) for p in poly]))   # index of the re-entrant corner P
    fixed = [((-a, L1), (0.0, L1))]
    return poly, idx, fixed, (E2a, E2b)


def notched_plate(W, H, x0, depth, omega_deg):
    """Plate W x H, V-notch on the top edge: opening omega, tip at (x0, H - depth). beta = 360 - omega."""
    hw = depth * np.tan(np.radians(omega_deg) / 2)
    poly = [np.array(p, float) for p in
            [(0, 0), (W, 0), (W, H), (x0 + hw, H), (x0, H - depth), (x0 - hw, H), (0, H)]]
    return poly, 4, [((0.0, H), (0.0, 0.0))], (np.array([W, 0.0]), np.array([W, H]))


def stepped_bar(D, d, Lb, two_sided, tilt_deg):
    """Flat bar with a shoulder (one- or two-sided). Step face can lean by tilt_deg. Returns corners to fillet."""
    h = (D - d) / 2 if two_sided else D - d
    dx = h * np.tan(np.radians(tilt_deg))
    if two_sided:
        y0 = h
        pts = [(0, 0), (Lb + dx, 0), (Lb, y0), (2 * Lb, y0), (2 * Lb, y0 + d), (Lb, y0 + d), (Lb + dx, D), (0, D)]
        ic, others = 2, [5]
        end = (np.array([2 * Lb, y0]), np.array([2 * Lb, y0 + d]))
    else:
        pts = [(0, 0), (2 * Lb, 0), (2 * Lb, d), (Lb, d), (Lb + dx, D), (0, D)]
        ic, others = 3, []
        end = (np.array([2 * Lb, 0.0]), np.array([2 * Lb, d]))
    poly = [np.array(p, float) for p in pts]
    return poly, ic, others, [((0.0, D), (0.0, 0.0))], end


def max_fillet(poly, i):
    """Largest radius whose tangent points stay inside 80 % of both adjacent edges."""
    from fe2d import interior_angles as _ia
    n = len(poly)
    psi = np.radians(360.0 - _ia(poly)[i])
    adj = min(np.linalg.norm(poly[i] - poly[i - 1]), np.linalg.norm(poly[(i + 1) % n] - poly[i]))
    return 0.8 * adj * np.tan(psi / 2)


def peak_near(m, s, P, radius):
    d = np.linalg.norm(m.p.T - P, axis=1)
    return float(s["vm"][d < radius].max())


def run_window(seed):
    """Plate with a random convex polygonal window (3-5 sides); target = one window corner, rounded."""
    from fe2d import fillet_points, interior_angles as _ia
    from sing_tool import cutout_corners, part_holes
    rng = np.random.default_rng(seed)
    W, H = rng.uniform(120, 200), rng.uniform(60, 120)
    nv = int(rng.integers(3, 6))
    for _try in range(50):
        ang = np.sort(rng.uniform(0, 2 * np.pi, nv))
        if np.min(np.diff(np.r_[ang, ang[0] + 2 * np.pi])) < np.radians(35):
            continue
        a_, b_ = rng.uniform(0.12, 0.3) * W, rng.uniform(0.15, 0.35) * H
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.4, 0.6) * H
        ring = [(cx + a_ * np.cos(t), cy + b_ * np.sin(t)) for t in ang]
        beta = _ia(ring[::-1])                        # material angles of the clockwise ring
        if beta.min() >= 195 and beta.max() <= 335:
            break
    else:
        return []
    ring = ring[::-1]                                  # clockwise (material on the left)
    outline = [np.array(p, float) for p in [(0, 0), (W, 0), (W, H), (0, H)]]
    fixed = [((0.0, H), (0.0, 0.0))]
    gam, q = rng.uniform(0, 2 * np.pi), rng.uniform(5, 50)
    trac = [(np.array([W, 0.0]), np.array([W, H]), (q * np.cos(gam), q * np.sin(gam)))]
    if rng.random() < 0.35:                            # extra load on the top edge -> biaxial-ish fields
        g2, q2 = rng.uniform(0, 2 * np.pi), rng.uniform(2, 25)
        trac.append((np.array([W, H]), np.array([0.0, H]), (q2 * np.cos(g2), q2 * np.sin(g2))))
    part = {"poly": outline, "poly_holes": [ring], "fixed": fixed, "traction": trac}
    k_ic = int(rng.integers(0, len(ring)))
    cc = {(k, i): (b, ell) for k, i, b, ell in cutout_corners(part)}
    if (0, k_ic) not in cc:
        return []
    beta, ell = cc[(0, k_ic)]
    _, holes = part_holes(part)
    ringcw = holes[0]
    n = len(ringcw)
    Pc = np.asarray(ringcw[k_ic], float)
    psi = 360.0 - beta                                 # window (hole) angle at the corner
    adj = min(np.linalg.norm(Pc - np.asarray(ringcw[k_ic - 1])), np.linalg.norm(np.asarray(ringcw[(k_ic + 1) % n]) - Pc))
    rmax = 0.8 * adj * np.tan(np.radians(psi) / 2)
    radii = np.sort(np.exp(rng.uniform(np.log(RHO_RANGE[0]), np.log(RHO_RANGE[1]), N_RADII))) * ell
    radii = radii[radii < rmax]
    if len(radii) == 0:
        return []
    hg = min(H / 8, ell / 2)
    geo = {"family": "window", "a": H, "L1": W, "L2": float(nv), "seed": seed, "psi": psi,
           "load_angle": np.degrees(gam), "q": q}
    targets = []
    for rho in radii:
        t1, cen, t2 = fillet_points(ringcw[k_ic - 1], Pc, ringcw[(k_ic + 1) % n], rho)
        a1 = np.arctan2(t1[1] - cen[1], t1[0] - cen[0])
        a2 = np.arctan2(t2[1] - cen[1], t2[0] - cen[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi   # shorter arc
        arc = [cen + rho * np.array([np.cos(a1 + da * u), np.sin(a1 + da * u)]) for u in np.linspace(0, 1, 25)]
        rr = [tuple(p) for p in ringcw[:k_ic]] + [tuple(p) for p in arc] + [tuple(p) for p in ringcw[k_ic + 1:]]
        idx = list(range(k_ic, k_ic + 25))
        m = mesh_polygon(outline, h_global=hg, poly_holes=[rr], hole_refine=[(0, i, rho / 12) for i in idx])
        s = solve_plane_stress(m, fixed, trac)
        targets.append(peak_near(m, s, Pc, 3 * rho / np.sin(np.radians(psi) / 2) + 0.05 * ell))
    rows = []
    for k in MESH_DIVS:
        h = ell / k
        m = mesh_polygon(outline, h_global=hg, poly_holes=[ringcw], hole_refine=[(0, k_ic, h)])
        s = solve_plane_stress(m, fixed, trac)
        f = corner_features(sampler_from_mesh(m, s), ringcw, k_ic, h, ell=ell)
        if f is None:
            continue
        j = np.argmin(np.linalg.norm(m.p.T - Pc, axis=1))
        f["vm_corner_node"] = float(s["vm"][j])
        for rho, tgt in zip(radii, targets):
            rows.append({**geo, **f, "rho": rho, "rho_over_ell": rho / ell, "target_peak": tgt})
    return rows


def run_geometry(seed):
    if FAMILY == "window":
        return run_window(seed)
    rng = np.random.default_rng(seed)
    gam = rng.uniform(0, 2 * np.pi)
    q = rng.uniform(5, 50)
    others = []
    if FAMILY == "stepped":
        D = rng.uniform(20, 60)
        d = D * rng.uniform(0.35, 0.9)
        two = bool(rng.random() < 0.5)
        Lb = rng.uniform(1.5, 3.0) * D
        tilt = rng.uniform(-25, 40) if rng.random() < 0.4 else 0.0
        poly, ic, others, fixed, (ea, eb) = stepped_bar(D, d, Lb, two, tilt)
        from fe2d import interior_angles as _ia
        psi = 360.0 - _ia(poly)[ic]
        hg = d / 5
        geo = {"family": "stepped", "a": D, "L1": d, "L2": Lb}
    elif FAMILY == "bracket":
        a = rng.uniform(10, 30)
        L1, L2 = rng.uniform(3, 6, 2) * a
        psi = rng.uniform(50, 170)
        poly, ic, fixed, (ea, eb) = angled_bracket(a, L1, L2, psi)
        hg = a / 4
        geo = {"family": "bracket", "a": a, "L1": L1, "L2": L2}
    else:
        H = rng.uniform(20, 60)
        W = rng.uniform(3, 5) * H
        depth = rng.uniform(0.15, 0.5) * H
        psi = rng.uniform(30, 160)                      # notch opening angle
        hw = depth * np.tan(np.radians(psi) / 2)
        x0 = rng.uniform(max(0.3 * W, hw + 0.1 * H), min(0.7 * W, W - hw - 0.1 * H))
        poly, ic, fixed, (ea, eb) = notched_plate(W, H, x0, depth, psi)
        hg = H / 6
        geo = {"family": "notched_plate", "a": H, "L1": W, "L2": depth}
    trac = [(ea, eb, (q * np.cos(gam), q * np.sin(gam)))]
    ell = corner_length(poly, ic)
    hg = min(hg, ell / 2)
    P = np.asarray(poly[ic])
    geo.update({"seed": seed, "psi": psi, "load_angle": np.degrees(gam), "q": q})

    radii = np.sort(np.exp(rng.uniform(np.log(RHO_RANGE[0]), np.log(RHO_RANGE[1]), N_RADII))) * ell
    rmax = min([max_fillet(poly, ic)] + [max_fillet(poly, o) for o in others])
    radii = radii[radii < rmax]
    if len(radii) == 0:
        return []
    targets = []
    for rho in radii:
        fil = {ic: rho, **{o: rho for o in others}}
        m = mesh_polygon(poly, fillets=fil, refine=[(k, rho / 12) for k in fil], h_global=min(hg, rho * 4 + hg / 2))
        s = solve_plane_stress(m, fixed, trac)
        targets.append(peak_near(m, s, P, 3 * rho / np.sin(np.radians(psi) / 2) + 0.05 * ell))

    rows = []
    for k in MESH_DIVS:
        h = ell / k
        m = mesh_polygon(poly, refine=[(ic, h)] + [(o, h) for o in others], h_global=hg)
        s = solve_plane_stress(m, fixed, trac)
        f = corner_features(sampler_from_mesh(m, s), poly, ic, h)
        if f is None:
            continue
        j = np.argmin(np.linalg.norm(m.p.T - P, axis=1))
        f["vm_corner_node"] = float(s["vm"][j])          # the singular number ANSYS would show
        for rho, tgt in zip(radii, targets):
            rows.append({**geo, **f, "rho": rho, "rho_over_ell": rho / ell, "target_peak": tgt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--family", default="bracket", choices=["bracket", "notched_plate", "stepped", "window"])
    ap.add_argument("--rho-min", type=float, default=0.01)
    ap.add_argument("--rho-max", type=float, default=0.3)
    args = ap.parse_args()
    global FAMILY, RHO_RANGE
    FAMILY = args.family
    RHO_RANGE = (args.rho_min, args.rho_max)
    t0 = time.time()
    rows, fails = [], 0
    with Pool(args.workers, initializer=_set_family, initargs=(args.family, RHO_RANGE)) as pool:
        for i, res in enumerate(pool.imap_unordered(_safe, range(args.seed0, args.seed0 + args.n))):
            if res is None:
                fails += 1
            else:
                rows += res
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{args.n} geometries, {len(rows)} rows, {fails} failed, {time.time() - t0:.0f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"done: {len(rows)} rows, {fails} failed, {time.time() - t0:.0f}s")


def _set_family(f, rr=(0.01, 0.3)):
    global FAMILY, RHO_RANGE
    FAMILY, RHO_RANGE = f, rr


def _safe(seed):
    try:
        return run_geometry(seed)
    except Exception as e:
        print(f"seed {seed} failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()
