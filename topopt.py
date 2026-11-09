"""
topopt.py - 2D topology optimisation (SIMP, minimum compliance) on the part's own outline, supports and loads.

    minimise   compliance  c = f^T u           (= make the part as stiff as possible)
    subject to volume <= volfrac * design area
               K(rho) u = f,  K = sum_e (Emin + rho_e^p (1 - Emin)) K_e
Standard method: SIMP penalisation p = 3, density filter (radius rmin), optimality-criteria update
(Sigmund 2001, "A 99 line topology optimization code"; Andreassen et al. 2011), here on an unstructured
triangle mesh of the real part so holes, fixtures and bearing loads are exactly those of the analysis.

Material is always kept (passive) in a ring around every loaded / fixed hole and a band along every fixed
or loaded edge, so supports and loads stay attached to the optimised part.

Then extract_shape() turns the density field into a clean outline + cutouts (+ the original bolt holes)
that EigenStress can analyse, and write_dxf() saves it for CAD.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from skfem import Basis, ElementVector, ElementTriP1, condense, solve

from fe2d import mesh_polygon, boundary_conditions, E_STEEL, NU_STEEL
from sing_tool import hole_fixture_arcs, part_holes, min_wall


def _point_seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def optimise(part, volfrac=0.4, h=None, iters=80, penal=3.0, rmin_factor=2.5, move=0.2, emin=1e-9,
             E=E_STEEL, nu=NU_STEEL, callback=None, tol=0.01):
    poly = [np.asarray(p, float) for p in part["poly"]]
    circles, poly_holes = part_holes(part)
    size = np.ptp(np.vstack(poly), axis=0).max()
    if h is None:
        h = min(size / 90, max(min_wall(part) / 2.5, size / 300))
    fixed_arcs, radial_arcs, bearing_arcs, splits = hole_fixture_arcs(part)
    circ = [(c, r, splits.get(j, []), min(h, 2 * np.pi * r / 32)) for j, (c, r) in enumerate(circles)]
    m = mesh_polygon(poly, h_global=h, circles=circ, poly_holes=poly_holes)

    e = ElementVector(ElementTriP1())
    ib = Basis(m, e)
    fixed = [tuple(map(tuple, x)) for x in part.get("fixed", [])]
    trac = [(x[0], x[1], tuple(x[2])) for x in part.get("traction", [])]
    f, D, Kpen = boundary_conditions(m, ib, e, fixed, trac, fixed_arcs, radial_arcs, bearing_arcs, E)

    # constant-strain triangle stiffness, unit thickness, plane stress
    t = m.t
    X, Y = m.p[0][t], m.p[1][t]                                    # (3, ne)
    A2 = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0])
    area = 0.5 * np.abs(A2)
    b = np.stack([Y[1] - Y[2], Y[2] - Y[0], Y[0] - Y[1]]) / A2       # (3, ne)
    c = np.stack([X[2] - X[1], X[0] - X[2], X[1] - X[0]]) / A2
    ne = t.shape[1]
    B = np.zeros((ne, 3, 6))
    B[:, 0, 0::2] = b.T
    B[:, 1, 1::2] = c.T
    B[:, 2, 0::2] = c.T
    B[:, 2, 1::2] = b.T
    Dm = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ke = area[:, None, None] * np.einsum("eki,kl,elj->eij", B, Dm, B)   # (ne, 6, 6)
    nd = ib.nodal_dofs                                             # (2, nnodes)
    edof = np.stack([nd[0][t[0]], nd[1][t[0]], nd[0][t[1]], nd[1][t[1]], nd[0][t[2]], nd[1][t[2]]], 1)
    iK = np.repeat(edof, 6, axis=1).ravel()
    jK = np.tile(edof, (1, 6)).ravel()

    # passive solid regions: around loaded / fixed holes and along fixed / loaded edges
    cen = np.c_[X.mean(0), Y.mean(0)]
    passive = np.zeros(ne, bool)
    for j, bc in part.get("hole_bc", {}).items():
        cc, rr = circles[int(j)]
        ring = max(0.6 * rr, 2.5 * h)
        passive |= np.linalg.norm(cen - np.asarray(cc), axis=1) < rr + ring
    band = 2.5 * h
    for p_, q_ in fixed:
        passive |= _point_seg_dist(cen, p_, q_) < band
    for p_, q_, _ in trac:
        passive |= _point_seg_dist(cen, p_, q_) < band

    # density filter
    rmin = rmin_factor * h
    tree = cKDTree(cen)
    pairs = tree.query_pairs(rmin, output_type="ndarray")
    ii = np.r_[np.arange(ne), pairs[:, 0], pairs[:, 1]]
    jj = np.r_[np.arange(ne), pairs[:, 1], pairs[:, 0]]
    dd = np.r_[np.zeros(ne), np.linalg.norm(cen[pairs[:, 0]] - cen[pairs[:, 1]], axis=1).repeat(2)]
    w = (rmin - dd) * area[jj]
    H = sp.coo_matrix((w, (ii, jj)), shape=(ne, ne)).tocsr()
    Hs = np.asarray(H.sum(1)).ravel()

    def filt(x):
        xp = H @ x / Hs
        xp[passive] = 1.0
        return xp

    Atot = area.sum()

    def compliance_of(xp_):
        Ee_ = emin + xp_ ** penal * (1 - emin)
        K_ = sp.coo_matrix(((Ee_[:, None, None] * Ke).ravel(), (iK, jK)), shape=(ib.N, ib.N)).tocsr()
        if Kpen is not None:
            K_ = K_ + Kpen
        u_ = solve(*condense(K_, f, D=D)) if len(D) else solve(K_, f)
        return float(f @ u_)
    c_full = compliance_of(np.ones(ne))                         # the original solid part, for comparison
    x = np.full(ne, volfrac)
    x[passive] = 1.0
    hist = []
    for it in range(iters):
        xp = filt(x)
        Ee = emin + xp ** penal * (1 - emin)
        K = sp.coo_matrix(((Ee[:, None, None] * Ke).ravel(), (iK, jK)), shape=(ib.N, ib.N)).tocsr()
        if Kpen is not None:
            K = K + Kpen
        u = solve(*condense(K, f, D=D)) if len(D) else solve(K, f)
        ue = u[edof]
        ce = np.einsum("ei,eij,ej->e", ue, Ke, ue)
        comp = float((Ee * ce).sum())
        dc = -penal * xp ** (penal - 1) * (1 - emin) * ce
        dc[passive] = 0.0
        dv = area.copy()
        dc = H.T @ (dc / Hs)
        dv = H.T @ (dv / Hs)
        l1, l2 = 0.0, 1e12 * max(1.0, -dc.min() / dv.min())
        free = ~passive
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xn = x.copy()
            xn[free] = np.clip(x[free] * np.sqrt(np.maximum(-dc[free], 0) / (dv[free] * lm)),
                               np.maximum(0.0, x[free] - move), np.minimum(1.0, x[free] + move))
            if (filt(xn) * area).sum() > volfrac * Atot:
                l1 = lm
            else:
                l2 = lm
        change = float(np.abs(xn - x).max())
        x = xn
        vol = float((filt(x) * area).sum() / Atot)
        hist.append({"iter": it + 1, "compliance": comp, "volume": vol, "change": change})
        if callback:
            callback(it + 1, iters, comp, vol)
        if it > 15 and change < tol:
            break
    xp = filt(x)
    return {"mesh": m, "density": xp, "passive": passive, "history": hist, "h": h, "area": area,
            "u": u, "volfrac": volfrac, "c_full": c_full}


# ------------------------------------------------------------------ shape extraction
def straighten(ring, max_dev, max_turn=25.0):
    """Remove mild kinks left by contouring: drop a vertex if the boundary only turns by < max_turn degrees
    there and it sits within max_dev of the line joining its neighbours. Real strut junctions stay."""
    P = [np.asarray(p, float) for p in ring]
    changed = True
    while changed and len(P) > 4:
        changed = False
        best, bi = None, None
        for i in range(len(P)):
            a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            v1, v2 = b - a, c - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                best, bi = 0.0, i
                break
            turn = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
            ac = c - a
            dev = abs(ac[0] * (b - a)[1] - ac[1] * (b - a)[0]) / max(np.linalg.norm(ac), 1e-12)
            if turn < max_turn and dev < max_dev and (best is None or turn < best):
                best, bi = turn, i
        if bi is not None:
            P.pop(bi)
            changed = True
    return [tuple(p) for p in P]



def extract_shape(res, part, threshold=0.5, simplify=None, round_r=0.0):
    """
    Density field -> solid region (density >= threshold) as outline + cutouts.
    simplify : Douglas-Peucker tolerance (mm), default 0.6 h - removes the mesh staircase
    round_r  : optional corner rounding radius (mm). 0 keeps sharp corners (for EigenStress to assess).
    Original bolt holes that survive are returned as exact circles (so fixtures can be re-applied).
    Returns geometry dict {outline, circles, poly_holes} + mapping old hole index -> new hole index.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union

    m, xp, h = res["mesh"], res["density"], res["h"]
    simplify = 0.6 * h if simplify is None else simplify
    # nodal densities (area-weighted average of surrounding elements)
    nn = m.p.shape[1]
    acc = np.zeros(nn)
    wts = np.zeros(nn)
    np.add.at(acc, m.t.T, (xp * res["area"])[:, None])
    np.add.at(wts, m.t.T, res["area"][:, None])
    rn = acc / np.maximum(wts, 1e-30)
    T = Triangulation(m.p[0], m.p[1], m.t.T)
    # passive (must-keep) material as a polygon, and anchor points that must end up inside the part
    from shapely.geometry import MultiPolygon
    tri = m.t.T[res["passive"]]
    keep = unary_union([Polygon(m.p[:, k].T) for k in tri]).buffer(0) if len(tri) else None
    anchors = []
    if keep is not None and not keep.is_empty:
        parts_ = keep.geoms if keep.geom_type == "MultiPolygon" else [keep]
        anchors = [pp.representative_point() for pp in parts_]

    def solid_at(thr):
        fig = plt.figure()
        cs = plt.tricontourf(T, rn, levels=[thr, 10.0])
        rings = []
        for path in cs.get_paths():
            rings += [r for r in path.to_polygons() if len(r) >= 4]
        plt.close(fig)
        g_ = None
        for r in rings:
            pg = Polygon(r).buffer(0)
            if not pg.is_empty:
                g_ = pg if g_ is None else g_.symmetric_difference(pg)
        if g_ is None:
            return None
        if keep is not None:
            g_ = g_.union(keep)
        # morphological clean-up: drop slivers and one-element spikes (radius ~ h/2)
        cr = 0.5 * h
        g_ = g_.buffer(-cr, join_style="mitre").buffer(cr, join_style="mitre")
        if keep is not None:
            g_ = g_.union(keep).buffer(0)
        return g_

    geom, used_thr, connected = None, threshold, False
    for thr in [threshold, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]:
        g_ = solid_at(thr)
        if g_ is None or g_.is_empty:
            continue
        main = max(g_.geoms, key=lambda q: q.area) if g_.geom_type == "MultiPolygon" else g_
        if all(main.buffer(h).contains(a_) for a_ in anchors):
            geom, used_thr, connected = main, thr, True
            break
        if geom is None:
            geom = main
    if geom is None or geom.is_empty:
        raise ValueError("optimisation left no material above the threshold")
    if round_r > 0:
        geom = geom.buffer(round_r, join_style="round").buffer(-2 * round_r, join_style="round") \
                   .buffer(round_r, join_style="round")
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
    from shapely.geometry import LinearRing
    circles, _ = part_holes(part)
    min_area = (2.5 * h) ** 2
    new_circles, poly_holes, hole_map = [], [], {}
    # 1) recognise the original bolt holes on the UNsimplified shape
    for ring in geom.interiors:
        R = Polygon(ring)
        if R.area < min_area:
            continue                                               # tiny pinholes from the threshold
        matched = None
        for j, (c, r) in enumerate(circles):
            if j in hole_map:
                continue
            if Point(c).within(R) and abs(R.area - np.pi * r * r) < 0.25 * np.pi * r * r:
                matched = j
                break
        if matched is not None:
            hole_map[matched] = len(new_circles)
            new_circles.append(circles[matched])
            continue
        # 2) real cutouts: simplify + straighten each ring on its own
        rs = LinearRing(ring.coords).simplify(simplify)
        hr = straighten([tuple(p) for p in np.asarray(rs.coords)[:-1]], max_dev=1.2 * h)
        if len(hr) < 3 or Polygon(hr).area < min_area:
            continue
        hx = np.asarray(hr)
        if 0.5 * np.sum(hx[:, 0] * np.roll(hx[:, 1], -1) - np.roll(hx[:, 0], -1) * hx[:, 1]) > 0:
            hr = hr[::-1]                                          # cutouts clockwise
        poly_holes.append(hr)
    ext = LinearRing(geom.exterior.coords).simplify(simplify)
    outline = straighten([tuple(p) for p in np.asarray(ext.coords)[:-1]], max_dev=1.2 * h)
    # outline must be counter-clockwise for the solver
    xy = np.asarray(outline)
    if 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) < 0:
        outline = outline[::-1]
    return {"outline": outline, "circles": new_circles, "poly_holes": poly_holes,
            "volume_kept": float(geom.area / Polygon(part["poly"]).area), "threshold": used_thr,
            "connected": connected}, hole_map


def write_dxf(geom, path):
    import ezdxf
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([tuple(map(float, p)) for p in geom["outline"]], close=True)
    for hole in geom["poly_holes"]:
        msp.add_lwpolyline([tuple(map(float, p)) for p in hole], close=True)
    for c, r in geom["circles"]:
        msp.add_circle((float(c[0]), float(c[1])), float(r))
    doc.saveas(path)
