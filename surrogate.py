"""
surrogate.py - instant stress and deformation for parametric part families, with no FE at run time.

FE is used OFFLINE only, to make training snapshots. In the app every answer comes from the surrogate.

Physics that makes it small and exact where it can be:
* Linearity: the response to (Fx, Fy) is Fx * (unit-Fx field) + Fy * (unit-Fy field). Superposition is exact,
  so the surrogate only learns the effect of GEOMETRY.
* Dimensional analysis (plane stress, traction loading): stress = s_hat * F / (t L), displacement =
  u_hat * F / (t E). Size L, thickness t and modulus E scale out exactly (nu fixed at 0.3).
* Williams scaling at the fillet: the peak stress grows like rho^(lam - 1) as the fillet shrinks, so the model
  learns s_peak * rho^(1 - lam), which varies slowly, then multiplies the singular part back.

Two parts per family:
1. Fields: proper orthogonal decomposition (POD) of the stress / displacement fields mapped onto a fixed
   reference grid, with radial-basis-function (RBF) interpolation of the POD coefficients over the
   geometry parameters (a classic POD-RBF reduced-order model).
2. Scalars: the tangential stress along each fillet arc (41 points per unit load) and the 2 x 2 compliance at
   the load point, also by RBF. Peak stress = max over the arc of |Fx a(s) + Fy b(s)|, exact for any load
   direction; strain energy = 1/2 F . C . F.

Families
  lbracket : unit square L, horizontal arm height a, vertical arm width b, inner fillet rho. Top edge
             clamped, load on the right end.        params (a, b, rho/min(a, b))
  stepped  : flat bar with shoulder fillets on both sides, narrow width d = 1, wide width D. Wide end
             clamped, load on the narrow end.      params (D/d, r/d)
"""
import json
import time
from dataclasses import dataclass

import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.interpolate import RBFInterpolator

from fe2d import fillet_points

NU = 0.3
LAM_270 = 0.5444837367824639            # Williams eigenvalue for a 270 deg corner
CHANNELS = ("sx", "sy", "txy", "ux", "uy")
N_ARC = 41


# --------------------------------------------------------------------------- families
@dataclass
class Family:
    key: str
    name: str
    pnames: tuple
    lo: tuple
    hi: tuple
    log: tuple            # which params are sampled / interpolated in log space
    grid: tuple           # reference grid (nxi, neta)
    L_name: str           # what the length scale is called in the app

    # ----- parameters -> unit geometry
    def outline(self, p):
        raise NotImplementedError

    def fillets(self, p):
        raise NotImplementedError

    def fixed(self, p):
        raise NotImplementedError

    def load_edge(self, p):
        raise NotImplementedError

    def to_ref(self, x, y, p):
        raise NotImplementedError

    def from_ref(self, xi, eta, p):
        raise NotImplementedError

    def ref_inside(self, XI, ETA):
        return np.ones_like(XI, bool)

    def valid(self, p):
        return True

    def nominal(self, p, Fx, Fy):
        """Nominal stress per unit (F / t L) for Kt, as in Peterson."""
        raise NotImplementedError

    def beam_scales(self, p):
        """Beam-theory size of the answer for each unit load: (stress scale, displacement scale) per load and
        the 2 x 2 compliance estimate. The surrogate learns the FE answer DIVIDED by these, which varies far
        less across the family than the raw answer (a thin arm is much more flexible and stressed)."""
        return {"Fx": (1.0, 1.0), "Fy": (1.0, 1.0)}, np.ones((2, 2))

    # ----- sampling
    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n:
            # Latin-hypercube-ish: stratify the first parameter
            u = rng.random(len(self.pnames))
            p = []
            for k, (a, b, lg) in enumerate(zip(self.lo, self.hi, self.log)):
                p.append(np.exp(np.log(a) + u[k] * (np.log(b) - np.log(a))) if lg else a + u[k] * (b - a))
            p = self.fix(np.array(p))
            if self.valid(p):
                out.append(p)
        return np.array(out)

    def fix(self, p):
        return p

    def z(self, P):
        """Parameters -> normalised interpolation space [0, 1]^k (log where flagged)."""
        P = np.atleast_2d(P).astype(float)
        Z = np.empty_like(P)
        for k, (a, b, lg) in enumerate(zip(self.lo, self.hi, self.log)):
            Z[:, k] = (np.log(P[:, k]) - np.log(a)) / (np.log(b) - np.log(a)) if lg else (P[:, k] - a) / (b - a)
        return Z


class LBracket(Family):
    def __init__(self):
        super().__init__("lbracket", "L-bracket", ("a", "b", "rho_rel"), (0.10, 0.10, 0.02), (0.35, 0.35, 0.60),
                         (True, True, True), (120, 120), "outer size")

    def rho(self, p):
        return p[2] * min(p[0], p[1])

    def outline(self, p):
        a, b = p[0], p[1]
        return [np.array(v, float) for v in [[0, 0], [1, 0], [1, a], [b, a], [b, 1], [0, 1]]]

    def fillets(self, p):
        return {3: self.rho(p)}

    def fixed(self, p):
        return [((p[1], 1.0), (0.0, 1.0))]

    def load_edge(self, p):
        return (1.0, 0.0), (1.0, p[0])

    def to_ref(self, x, y, p):
        a, b = p[0], p[1]
        xi = np.where(x < b, 0.5 * x / b, 0.5 + 0.5 * (x - b) / (1 - b))
        eta = np.where(y < a, 0.5 * y / a, 0.5 + 0.5 * (y - a) / (1 - a))
        return xi, eta

    def from_ref(self, xi, eta, p):
        a, b = p[0], p[1]
        x = np.where(xi < 0.5, 2 * xi * b, b + 2 * (xi - 0.5) * (1 - b))
        y = np.where(eta < 0.5, 2 * eta * a, a + 2 * (eta - 0.5) * (1 - a))
        return x, y

    def ref_inside(self, XI, ETA):
        return (XI <= 0.5 + 1e-9) | (ETA <= 0.5 + 1e-9)

    def inside(self, x, y, p):
        """Physical inside test, including the fillet material."""
        a, b = p[0], p[1]
        r = self.rho(p)
        ins = (x >= 0) & (y >= 0) & (x <= 1) & (y <= 1) & ((x <= b) | (y <= a))
        cx, cy = b + r, a + r
        corner = (x > b) & (x < cx) & (y > a) & (y < cy)
        ins |= corner & (np.hypot(x - cx, y - cy) >= r)
        return ins

    def beam_scales(self, p):
        a, b = p[0], p[1]
        la, lv = 1 - b / 2, 1 - a / 2                  # centre-line arm lengths (unit size)
        Ia, Ib = a ** 3 / 12, b ** 3 / 12
        Cyy = la ** 3 / (3 * Ia) + la ** 2 * lv / Ib    # Castigliano, E = 1, t = 1
        Cxx = lv ** 3 / (3 * Ib) + la / a
        Cxy = la * lv ** 2 / (2 * Ib)
        C = np.array([[Cxx, Cxy], [Cxy, Cyy]])
        sFx = max(1 / a, 6 * lv / b ** 2)
        sFy = max(6 * la / a ** 2, 6 * la / b ** 2)
        return {"Fx": (sFx, Cxx), "Fy": (sFy, Cyy)}, C

    def nominal(self, p, Fx, Fy):
        """Bending + axial stress in the horizontal arm at the joint face x = b (beam theory)."""
        a, b = p[0], p[1]
        return np.abs(6 * Fy * (1 - b) / a ** 2) + np.abs(Fx / a)


class Stepped(Family):
    LW, LN = 2.0, 3.0           # wide and narrow lengths, in units of d

    def __init__(self):
        super().__init__("stepped", "Stepped flat bar (shoulder fillets)", ("D", "r"), (1.1, 0.02), (3.0, 0.30),
                         (False, True), (220, 50), "narrow width d")

    def valid(self, p):
        return p[1] <= 0.95 * (p[0] - 1) / 2

    def outline(self, p):
        D, Lw, Ln = p[0], self.LW, self.LN
        lo, hi = (D - 1) / 2, (D + 1) / 2
        return [np.array(v, float) for v in [[0, 0], [Lw, 0], [Lw, lo], [Lw + Ln, lo], [Lw + Ln, hi], [Lw, hi],
                                             [Lw, D], [0, D]]]

    def fillets(self, p):
        return {2: p[1], 5: p[1]}

    def fixed(self, p):
        return [((0.0, p[0]), (0.0, 0.0))]

    def load_edge(self, p):
        D = p[0]
        return (self.LW + self.LN, (D - 1) / 2), (self.LW + self.LN, (D + 1) / 2)

    def _ybot(self, x, p):
        D, r, Lw = p[0], p[1], self.LW
        lo = (D - 1) / 2
        yb = np.where(x <= Lw, 0.0, lo)
        f = (x > Lw) & (x < Lw + r)
        dx = np.clip(x - Lw - r, -r, r)
        yb = np.where(f, lo - r + np.sqrt(np.maximum(r * r - dx * dx, 0)), yb)
        return yb

    def to_ref(self, x, y, p):
        D = p[0]
        yb = self._ybot(x, p)
        w = D - 2 * yb
        return x / (self.LW + self.LN), (y - yb) / np.maximum(w, 1e-12)

    def from_ref(self, xi, eta, p):
        x = xi * (self.LW + self.LN)
        yb = self._ybot(x, p)
        return x, yb + eta * (p[0] - 2 * yb)

    def inside(self, x, y, p):
        yb = self._ybot(x, p)
        return (x >= 0) & (x <= self.LW + self.LN) & (y >= yb) & (y <= p[0] - yb)

    def nominal(self, p, Fx, Fy):
        """Peterson's nominal stress in the narrow section at the shoulder: P/d + 6M/d^2, M = Fy * Ln."""
        return np.abs(Fx) + np.abs(6 * Fy * self.LN)


FAMILIES = {"lbracket": LBracket(), "stepped": Stepped()}


# --------------------------------------------------------------------------- offline: FE snapshots
def fe_solve(fam, p, h_global=None, div=8):
    """Two unit load cases on the unit geometry (E = 1, force per unit thickness = 1)."""
    from fe2d import mesh_polygon, solve_plane_stress
    poly = fam.outline(p)
    fil = fam.fillets(p)
    size = np.ptp(np.vstack(poly), axis=0).max()
    hg = h_global or size / 45
    m = mesh_polygon(poly, fillets=fil, refine=[(i, r / div) for i, r in fil.items()], h_global=hg)
    (x0, y0), (x1, y1) = fam.load_edge(p)
    Le = np.hypot(x1 - x0, y1 - y0)
    out = {}
    for tag, F in (("Fx", (1.0, 0.0)), ("Fy", (0.0, 1.0))):
        s = solve_plane_stress(m, fam.fixed(p), [((x0, y0), (x1, y1), (F[0] / Le, F[1] / Le))], E=1.0, nu=NU)
        out[tag] = s
    return m, out


def arc_profiles(fam, p, m, sols):
    """Tangential stress along every fillet arc (N_ARC points, tangent point to tangent point)."""
    poly = fam.outline(p)
    n = len(poly)
    xy = m.p.T
    res = []
    for i, r in sorted(fam.fillets(p).items()):
        t1, c, t2 = fillet_points(poly[i - 1], poly[i], poly[(i + 1) % n], r)
        d = np.linalg.norm(xy - c, axis=1)
        on = np.flatnonzero(np.abs(d - r) < 1e-6 * max(r, 1e-9) + 1e-9)
        a1 = np.arctan2(*(t1 - c)[::-1])
        a2 = np.arctan2(*(t2 - c)[::-1])
        span = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        ang = np.arctan2(xy[on, 1] - c[1], xy[on, 0] - c[0])
        sfrac = ((ang - a1 + np.pi) % (2 * np.pi) - np.pi) / span
        keep = (sfrac > -0.01) & (sfrac < 1.01)
        on, sfrac, ang = on[keep], sfrac[keep], ang[keep]
        order = np.argsort(sfrac)
        on, sfrac, ang = on[order], sfrac[order], ang[order]
        tx, ty = -np.sin(ang), np.cos(ang)
        prof = {}
        for tag, s in sols.items():
            stt = s["sx"][on] * tx * tx + 2 * s["txy"][on] * tx * ty + s["sy"][on] * ty * ty
            prof[tag] = np.interp(np.linspace(0, 1, N_ARC), sfrac, stt)
        res.append(prof)
    return res


def compliance(fam, p, m, sols):
    """C[i, j] = mean displacement component i along the load edge under unit load j (E = 1, t = 1)."""
    (x0, y0), (x1, y1) = fam.load_edge(p)
    xy = m.p.T
    L = np.hypot(x1 - x0, y1 - y0)
    tvec = np.array([x1 - x0, y1 - y0]) / L
    rel = xy - np.array([x0, y0])
    along = rel @ tvec
    perp = np.abs(rel @ np.array([-tvec[1], tvec[0]]))
    on = (perp < 1e-7) & (along > -1e-7) & (along < L + 1e-7)
    C = np.zeros((2, 2))
    for j, tag in enumerate(("Fx", "Fy")):
        C[0, j] = sols[tag]["ux"][on].mean()
        C[1, j] = sols[tag]["uy"][on].mean()
    return C


def ref_grid(fam):
    nxi, neta = fam.grid
    XI, ETA = np.meshgrid(np.linspace(0, 1, nxi), np.linspace(0, 1, neta))
    return XI, ETA


def snapshot(fam, p, m, sols):
    """All channels of both load cases on the reference grid (holes filled from the nearest valid value)."""
    from scipy.spatial import cKDTree
    XI, ETA = ref_grid(fam)
    X, Y = fam.from_ref(XI, ETA, p)
    T = Triangulation(m.p[0], m.p[1], m.t.T)
    inside = fam.ref_inside(XI, ETA)
    vec = []
    for tag in ("Fx", "Fy"):
        for ch in CHANNELS:
            v = np.ma.filled(LinearTriInterpolator(T, sols[tag][ch])(X, Y), np.nan)
            bad = ~np.isfinite(v) & inside
            if bad.any():
                good = np.isfinite(v)
                tree = cKDTree(np.c_[X[good], Y[good]])
                _, j = tree.query(np.c_[X[bad], Y[bad]])
                v[bad] = v[good][j]
            v[~inside] = 0.0
            vec.append(v.ravel())
    return np.concatenate(vec)


def generate(fam, n_train, n_test, seed=1, verbose=True):
    P = fam.sample(n_train + n_test, seed=seed)
    snaps, arcs, comps = [], [], []
    t0 = time.time()
    for k, p in enumerate(P):
        m, sols = fe_solve(fam, p)
        snaps.append(snapshot(fam, p, m, sols).astype(np.float32))
        arcs.append(np.array([[pr["Fx"], pr["Fy"]] for pr in arc_profiles(fam, p, m, sols)]))
        comps.append(compliance(fam, p, m, sols))
        if verbose and (k % 10 == 0 or k == len(P) - 1):
            print(f"  {fam.key} {k + 1}/{len(P)}  p={np.round(p, 4)}  nodes={m.p.shape[1]}  {time.time() - t0:.0f}s",
                  flush=True)
    return {"P": P, "snaps": np.array(snaps), "arcs": np.array(arcs), "C": np.array(comps),
            "n_train": n_train}


# --------------------------------------------------------------------------- the surrogate
class Surrogate:
    """POD-RBF field model + RBF arc-profile and compliance models for one family."""

    def __init__(self, fam, energy=0.99999, max_modes=60, smoothing=1e-8):
        self.fam = fam
        self.energy = energy
        self.max_modes = max_modes
        self.smoothing = smoothing

    def _arc_scale(self, P):
        rho = np.array([min(self.fam.fillets(p).values()) for p in P])
        return rho ** (1 - LAM_270)              # Williams: s_peak ~ rho^(lam - 1)

    def _snap_scale(self, P, npts):
        """Per-sample divisor for every entry of a snapshot vector."""
        out = []
        for p in P:
            sc, _ = self.fam.beam_scales(p)
            v = []
            for tag in ("Fx", "Fy"):
                for ch in CHANNELS:
                    v.append(sc[tag][0] if ch in ("sx", "sy", "txy") else sc[tag][1])
            out.append(np.repeat(v, npts))
        return np.array(out, dtype=np.float32)

    def _C_scale(self, P):
        return np.array([self.fam.beam_scales(p)[1] for p in P])

    def fit(self, P, snaps, arcs, C):
        Z = self.fam.z(P)
        npts = snaps.shape[1] // (2 * len(CHANNELS))
        snaps = snaps / self._snap_scale(P, npts)
        C = C / self._C_scale(P)
        sc = np.array([[self.fam.beam_scales(p)[0][t][0] for t in ("Fx", "Fy")] for p in P])
        raw_arcs, raw_C = arcs, C * self._C_scale(P)
        self.mean = snaps.mean(0)
        A = snaps - self.mean
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(min(np.searchsorted(e, self.energy) + 1, self.max_modes, len(S)))
        self.modes = Vt[:k].astype(np.float32)
        self.sv = S
        coef = A @ self.modes.T
        self._coef_fit = coef
        self.n_modes = k
        self._fit_small(P, coef, raw_arcs, raw_C)
        return self

    # ----- unit-load predictions (dimensionless)
    def fields(self, p):
        c = self.rbf_field(self.fam.z(p))[0]
        v = self.mean + c @ self.modes
        v = v * self._snap_scale([p], len(v) // (2 * len(CHANNELS)))[0]
        nxi, neta = self.fam.grid
        v = v.reshape(2, len(CHANNELS), neta, nxi)
        return {tag: {ch: v[j, q] for q, ch in enumerate(CHANNELS)} for j, tag in enumerate(("Fx", "Fy"))}

    def arcs(self, p):
        a = self.rbf_arc(self.fam.z(p))[0].reshape(self.arc_shape)
        sc = self.fam.beam_scales(p)[0]
        a = a * np.array([sc["Fx"][0], sc["Fy"][0]])[None, :, None]
        return a / self._arc_scale([p])[0]

    def compliance(self, p):
        return self.rbf_C(self.fam.z(p))[0].reshape(2, 2) * self._C_scale([p])[0]

    def save(self, path):
        np.savez_compressed(path, key=self.fam.key, P=self._P, snaps_mean=self.mean, modes=self.modes.astype(np.float16),
                            coef=self._coef, arcs=self._arcs, C=self._C, sv=self.sv,
                            meta=json.dumps({"energy": self.energy, "smoothing": self.smoothing,
                                             "n_modes": self.n_modes}))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        fam = FAMILIES[str(d["key"])]
        meta = json.loads(str(d["meta"]))
        s = cls(fam, energy=meta["energy"], smoothing=meta["smoothing"])
        s.max_modes = meta.get("max_modes", s.max_modes)
        s.mean, s.modes, s.sv = d["snaps_mean"], d["modes"].astype(np.float32), d["sv"]
        s.n_modes = meta["n_modes"]
        P, coef, arcs, C = d["P"], d["coef"], d["arcs"], d["C"]
        s._fit_small(P, coef, arcs, C)
        s._P, s._coef, s._arcs, s._C = P, coef, arcs, C
        return s

    def _fit_small(self, P, coef, arcs, C):
        """The RBF parts (cheap): from POD coefficients and the RAW arc / compliance data."""
        Z = self.fam.z(P)
        self.rbf_field = RBFInterpolator(Z, coef, kernel="thin_plate_spline", smoothing=self.smoothing, degree=1)
        sc = np.array([[self.fam.beam_scales(p)[0][t][0] for t in ("Fx", "Fy")] for p in P])
        a = arcs / sc[:, None, :, None] * self._arc_scale(P)[:, None, None, None]
        self.arc_shape = arcs.shape[1:]
        self.rbf_arc = RBFInterpolator(Z, a.reshape(len(P), -1), kernel="thin_plate_spline",
                                       smoothing=self.smoothing, degree=1)
        self.rbf_C = RBFInterpolator(Z, (C / self._C_scale(P)).reshape(len(P), -1), kernel="thin_plate_spline",
                                     smoothing=self.smoothing, degree=1)


def train(fam, data):
    n = data["n_train"]
    s = Surrogate(fam).fit(data["P"][:n], data["snaps"][:n], data["arcs"][:n], data["C"][:n])
    s._P = data["P"][:n]
    s._coef = s._coef_fit
    s._arcs, s._C = data["arcs"][:n], data["C"][:n]
    return s


# --------------------------------------------------------------------------- physical answers
def predict(sur, p, L, t, E, Fx, Fy):
    """Everything the app shows, in physical units (mm, N, MPa)."""
    fam = sur.fam
    sF = 1.0 / (t * L)                    # stress per unit dimensionless stress per N
    uF = 1.0 / (t * E)                    # displacement per unit dimensionless displacement per N
    arcs = sur.arcs(p)                    # (n_fillets, 2, N_ARC)
    comb = Fx * arcs[:, 0] + Fy * arcs[:, 1]
    k = np.unravel_index(np.argmax(np.abs(comb)), comb.shape)
    peak = float(np.abs(comb[k])) * sF
    C = sur.compliance(p)
    u_load = C @ np.array([Fx, Fy]) * uF
    U = 0.5 * np.array([Fx, Fy]) @ C @ np.array([Fx, Fy]) * uF
    nom = fam.nominal(p, Fx, Fy) * sF
    return {"peak": peak, "peak_fillet": int(k[0]), "peak_s": k[1] / (N_ARC - 1), "arc_profile": comb * sF,
            "u_load": u_load, "energy": float(U), "nominal": float(nom), "Kt": peak / nom if nom > 0 else np.nan}


def physical_fields(sur, p, L, t, E, Fx, Fy):
    """Fields on the mapped reference grid in physical coordinates: returns X, Y (mm), inside mask and
    dict of sx, sy, txy (MPa), ux, uy (mm), plus a triangulation of the inside cells."""
    fam = sur.fam
    f = sur.fields(p)
    XI, ETA = ref_grid(fam)
    Xu, Yu = fam.from_ref(XI, ETA, p)
    out = {}
    for ch in CHANNELS:
        v = Fx * f["Fx"][ch] + Fy * f["Fy"][ch]
        out[ch] = v * (1.0 / (t * L) if ch.startswith("s") or ch == "txy" else 1.0 / (t * E))
    X, Y = Xu * L, Yu * L
    inside = fam.ref_inside(XI, ETA) & fam.inside(Xu, Yu, p)
    neta, nxi = XI.shape
    idx = np.arange(neta * nxi).reshape(neta, nxi)
    q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    ins = inside.ravel()
    tri = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    tri = tri[ins[tri].all(1)]
    return X, Y, inside, out, tri


def to_image(X, Y, tri, fields, n_px=700, keys=("sx", "sy", "txy")):
    T = Triangulation(X.ravel(), Y.ravel(), tri)
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    ny = max(int(n_px * (y1 - y0) / (x1 - x0)), 2)
    GX, GY = np.meshgrid(np.linspace(x0, x1, n_px), np.linspace(y0, y1, ny))
    return {k: np.ma.filled(LinearTriInterpolator(T, fields[k].ravel())(GX, GY), np.nan) for k in keys}, \
        (x0, x1, y0, y1)


def required_radius(sur, fam_params_fn, L, t, Fx, Fy, allow, rel_lo=None, rel_hi=None, n=80):
    """Smallest fillet keeping the peak below allow: scan the fillet parameter over its trained range."""
    k = 2 if sur.fam.key == "lbracket" else 1
    lo = rel_lo if rel_lo is not None else sur.fam.lo[k]
    hi = rel_hi if rel_hi is not None else sur.fam.hi[k]
    vals = np.geomspace(lo, hi, n)
    for v in vals:
        p = fam_params_fn(v)
        if not sur.fam.valid(p):
            continue
        if predict(sur, p, L, t, 1.0, Fx, Fy)["peak"] <= allow:
            return v, p
    return None, None
