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


def peak_near(m, s, P, radius):
    d = np.linalg.norm(m.p.T - P, axis=1)
    return float(s["vm"][d < radius].max())


def run_geometry(seed):
    rng = np.random.default_rng(seed)
    gam = rng.uniform(0, 2 * np.pi)
    q = rng.uniform(5, 50)
    if FAMILY == "bracket":
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

    radii = np.sort(np.exp(rng.uniform(np.log(0.01), np.log(0.3), N_RADII))) * ell
    targets = []
    for rho in radii:
        m = mesh_polygon(poly, fillets={ic: rho}, refine=[(ic, rho / 12)], h_global=hg)
        s = solve_plane_stress(m, fixed, trac)
        targets.append(peak_near(m, s, P, 3 * rho / np.sin(np.radians(psi) / 2) + 0.05 * ell))

    rows = []
    for k in MESH_DIVS:
        h = ell / k
        m = mesh_polygon(poly, refine=[(ic, h)], h_global=hg)
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
    ap.add_argument("--family", default="bracket", choices=["bracket", "notched_plate"])
    args = ap.parse_args()
    global FAMILY
    FAMILY = args.family
    t0 = time.time()
    rows, fails = [], 0
    with Pool(args.workers, initializer=_set_family, initargs=(args.family,)) as pool:
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


def _set_family(f):
    global FAMILY
    FAMILY = f


def _safe(seed):
    try:
        return run_geometry(seed)
    except Exception as e:
        print(f"seed {seed} failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    main()
