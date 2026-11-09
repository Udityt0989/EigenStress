"""
fe2d.py - small 2D plane-stress FE solver used as the open-source stand-in for ANSYS.

* Geometry: any simple polygon (list of (x, y) in mm). Any vertex can be rounded with a fillet radius.
* Mesh: gmsh, linear triangles, graded refinement around chosen points / fillet arcs.
* Element: 6-node quadratic triangle (P2) - same family as ANSYS PLANE183.
* Stress: exact element-vertex stresses, then averaged at shared nodes (what ANSYS shows as
  "averaged nodal stress"), so singular behaviour looks the same as in ANSYS.
Units: mm, N, MPa.
"""
import numpy as np
import gmsh
from skfem import (MeshTri, Basis, FacetBasis, ElementVector, ElementTriP2, asm,
                   LinearForm, condense, solve)
from skfem.models.elasticity import linear_elasticity, lame_parameters

E_STEEL, NU_STEEL = 210000.0, 0.3


# ---------------------------------------------------------------- geometry helpers
def fillet_points(A, P, B, r):
    """Tangent points and centre for a fillet of radius r at vertex P between edges P->A and P->B."""
    A, P, B = map(np.asarray, (A, P, B))
    ua = (A - P) / np.linalg.norm(A - P)
    ub = (B - P) / np.linalg.norm(B - P)
    phi = np.arccos(np.clip(ua @ ub, -1, 1))          # wedge angle between the two edges
    t = r / np.tan(phi / 2)
    bis = (ua + ub) / np.linalg.norm(ua + ub)
    c = P + bis * r / np.sin(phi / 2)
    return P + t * ua, c, P + t * ub


def interior_angles(poly):
    """Interior (material-side) angle at each vertex in degrees, for a CCW polygon."""
    poly = np.asarray(poly, float)
    n = len(poly)
    ang = np.zeros(n)
    for i in range(n):
        a, p, b = poly[i - 1], poly[i], poly[(i + 1) % n]
        v1, v2 = a - p, b - p
        cross = v2[0] * v1[1] - v2[1] * v1[0]
        ang[i] = np.degrees(np.arctan2(cross, v2 @ v1)) % 360
    return ang


def signed_area(poly):
    x, y = np.asarray(poly, float).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------- meshing
def mesh_polygon(poly, fillets=None, refine=None, h_global=5.0, verbose=False, circles=None, poly_holes=None):
    """
    poly       : CCW list of vertices
    fillets    : {vertex_index: radius}
    refine     : list of (vertex_index, h_local) - graded refinement at that vertex (or its fillet arc)
    circles    : list of ((cx, cy), r) or ((cx, cy), r, split_angles_deg, h_hole) circular holes.
                 split_angles put mesh nodes exactly where a hole fixture starts/ends.
    poly_holes : list of polygonal holes (lists of vertices)
    Returns skfem MeshTri.
    """
    circles = circles or []
    poly_holes = poly_holes or []
    fillets = fillets or {}
    refine = refine or []
    poly = [np.asarray(p, float) for p in poly]
    n = len(poly)

    gmsh.initialize(interruptible=False)   # no signal handler: lets it run inside the web app
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("part")
    geo = gmsh.model.geo
    curves, vertex_entity = [], {}       # vertex_entity[i] = ("point", tag) or ("curve", tag)
    first_pt, prev_pt = None, None

    def add_pt(xy):
        return geo.addPoint(xy[0], xy[1], 0)

    for i in range(n):
        P = poly[i]
        if i in fillets:
            t1, c, t2 = fillet_points(poly[i - 1], P, poly[(i + 1) % n], fillets[i])
            p1, pc, p2 = add_pt(t1), add_pt(c), add_pt(t2)
            if prev_pt is not None:
                curves.append(geo.addLine(prev_pt, p1))
            else:
                first_pt = p1
            arc = geo.addCircleArc(p1, pc, p2)
            curves.append(arc)
            vertex_entity[i] = ("curve", arc)
            prev_pt = p2
        else:
            p = add_pt(P)
            if prev_pt is not None:
                curves.append(geo.addLine(prev_pt, p))
            else:
                first_pt = p
            vertex_entity[i] = ("point", p)
            prev_pt = p
    curves.append(geo.addLine(prev_pt, first_pt))
    loops = [geo.addCurveLoop(curves)]
    hole_curves = []
    for circ in circles:
        (cx, cy), r = circ[0], circ[1]
        splits = circ[2] if len(circ) > 2 and circ[2] is not None else []
        h_hole = circ[3] if len(circ) > 3 and circ[3] is not None else 2 * np.pi * r / 32
        angs = sorted({round(a % 360.0, 9) for a in [0.0, 90.0, 180.0, 270.0] + list(splits)})
        pc = geo.addPoint(cx, cy, 0)
        q = [geo.addPoint(cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a)), 0) for a in angs]
        arcs = [geo.addCircleArc(q[k], pc, q[(k + 1) % len(q)]) for k in range(len(q))]
        loops.append(geo.addCurveLoop(arcs))
        hole_curves.append((arcs, min(h_global, h_hole)))
    for hole in poly_holes:
        pts = [add_pt(np.asarray(p, float)) for p in hole]
        lines = [geo.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
        loops.append(geo.addCurveLoop(lines))
    geo.addPlaneSurface(loops)
    geo.synchronize()

    fields = []
    for arcs, hs in hole_curves:
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", arcs)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 100)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", hs)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_global)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", hs)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 3 * h_global)
        fields.append(ft)
    for vi, h_loc in refine:
        kind, tag = vertex_entity[vi]
        fd = gmsh.model.mesh.field.add("Distance")
        if kind == "point":
            gmsh.model.mesh.field.setNumbers(fd, "PointsList", [tag])
        else:
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [tag])
            gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_loc)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_global)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 1.5 * h_loc)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", 3.0 * h_global)
        fields.append(ft)
    if fields:
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_global)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xy = coords.reshape(-1, 3)[:, :2]
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tri = enodes[list(etypes).index(2)].reshape(-1, 3)
    gmsh.finalize()

    idx = np.full(int(node_tags.max()) + 1, -1)
    idx[node_tags.astype(int)] = np.arange(len(node_tags))
    tri = idx[tri.astype(int)]
    used = np.unique(tri)                           # drop geometry-only nodes (e.g. arc centres)
    remap = np.full(len(xy), -1)
    remap[used] = np.arange(len(used))
    return MeshTri(xy[used].T.copy(), remap[tri].T.copy())


# ---------------------------------------------------------------- solving
def _on_segment(x, p, q, tol):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    rel = np.stack([x[0] - p[0], x[1] - p[1]])
    s = (rel[0] * d[0] + rel[1] * d[1]) / L
    dist = np.abs(rel[0] * d[1] - rel[1] * d[0]) / L
    return (dist < tol) & (s > -tol) & (s < L + tol)


def _arc_facets(m, c, r, a0, a1):
    """Boundary facets lying on the circle (c, r) between angles a0 -> a1 (deg, CCW)."""
    c = np.asarray(c, float)
    span = (a1 - a0) % 360.0 or 360.0

    def on_arc(x):
        d = np.hypot(x[0] - c[0], x[1] - c[1])
        ang = np.degrees(np.arctan2(x[1] - c[1], x[0] - c[0]))
        rel = (ang - a0) % 360.0
        return (np.abs(d - r) < 0.02 * r) & (rel <= span + 1e-9)
    f = m.facets_satisfying(on_arc)
    return np.intersect1d(f, m.boundary_facets())


def solve_plane_stress(m, fixed_edges, traction_edges, E=E_STEEL, nu=NU_STEEL,
                       fixed_arcs=(), radial_arcs=(), bearing_arcs=()):
    """
    fixed_edges    : list of (p, q) segments, fully clamped
    traction_edges : list of (p, q, (tx, ty)) uniform traction in MPa (N/mm^2 per unit thickness)
    fixed_arcs     : list of (c, r, a0, a1) hole arcs fully clamped (bolted / press-fit hole)
    radial_arcs    : list of (c, r, a0, a1) hole arcs held only radially (frictionless pin contact)
    bearing_arcs   : list of (c, r, (Fx, Fy)) bolt bearing load: force per unit thickness (N/mm),
                     cosine pressure on the half of the hole the bolt pulls toward
    Returns dict of averaged nodal stresses (sx, sy, txy, vm) on mesh vertices.
    """
    e = ElementVector(ElementTriP2())
    ib = Basis(m, e)
    lam, mu = lame_parameters(E, nu)
    lam = 2 * lam * mu / (lam + 2 * mu)             # plane-stress modified Lame constant
    K = asm(linear_elasticity(lam, mu), ib)

    tol = 1e-6 * np.ptp(m.p)
    f = np.zeros(K.shape[0])
    for p, q, (tx, ty) in traction_edges:
        facets = m.facets_satisfying(lambda x, p=p, q=q: _on_segment(x, p, q, tol))
        fb = FacetBasis(m, e, facets=facets)

        @LinearForm
        def trac(v, w):
            return tx * v[0] + ty * v[1]
        f += asm(trac, fb)

    # bolt bearing: p(theta) = p0 cos(theta - thetaF) on |theta - thetaF| < 90 deg, p0 = 2F / (pi r)
    for c, r, (Fx, Fy) in bearing_arcs:
        Fm = np.hypot(Fx, Fy)
        if Fm == 0:
            continue
        thF = np.degrees(np.arctan2(Fy, Fx))
        facets = _arc_facets(m, c, r, thF - 90, thF + 90)
        fb = FacetBasis(m, e, facets=facets)
        p0 = 2 * Fm / (np.pi * r)
        cx, cy, ux, uy = c[0], c[1], Fx / Fm, Fy / Fm

        @LinearForm
        def bear(v, w, cx=cx, cy=cy, ux=ux, uy=uy, p0=p0):
            dx, dy = w.x[0] - cx, w.x[1] - cy
            d = np.sqrt(dx ** 2 + dy ** 2)
            nx, ny = dx / d, dy / d                       # from hole centre into the material
            pr = p0 * np.maximum(nx * ux + ny * uy, 0.0)
            return pr * (nx * v[0] + ny * v[1])
        f += asm(bear, fb)

    D = []
    for p, q in fixed_edges:
        facets = m.facets_satisfying(lambda x, p=p, q=q: _on_segment(x, p, q, tol))
        D.append(ib.get_dofs(facets=facets).all())
    for c, r, a0, a1 in fixed_arcs:
        D.append(ib.get_dofs(facets=_arc_facets(m, c, r, a0, a1)).all())
    D = np.unique(np.concatenate(D)) if D else np.array([], dtype=int)

    # frictionless pin: stiff radial springs at every DOF location on the contact arc (penalty method)
    if radial_arcs:
        import scipy.sparse as sp
        kpen = 1e4 * E
        rows, cols, vals = [], [], []
        for c, r, a0, a1 in radial_arcs:
            dofs = ib.get_dofs(facets=_arc_facets(m, c, r, a0, a1))
            ux_ = np.concatenate([dofs.nodal["u^1"], dofs.facet["u^1"]])
            uy_ = np.concatenate([dofs.nodal["u^2"], dofs.facet["u^2"]])
            loc = ib.doflocs[:, ux_]
            n = loc - np.asarray(c, float)[:, None]
            n /= np.linalg.norm(n, axis=0)
            for a, b, w_ in ((ux_, ux_, n[0] * n[0]), (ux_, uy_, n[0] * n[1]),
                             (uy_, ux_, n[1] * n[0]), (uy_, uy_, n[1] * n[1])):
                rows.append(a); cols.append(b); vals.append(kpen * w_)
        K = K + sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=K.shape).tocsr()
    if len(D):
        u = solve(*condense(K, f, D=D))
    else:
        u = solve(K, f)

    # exact strain at each element's 3 vertices (strain is linear in a P2 triangle)
    X = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vb = Basis(m, e, quadrature=(X, np.ones(3) / 6))
    g = vb.interpolate(u).grad                       # (2 comp, 2 dir, nel, 3)
    exx, eyy = g[0, 0], g[1, 1]
    gxy = g[0, 1] + g[1, 0]
    c = E / (1 - nu ** 2)
    sx = c * (exx + nu * eyy)
    sy = c * (eyy + nu * exx)
    txy = E / (2 * (1 + nu)) * gxy

    out = {}
    cnt = np.zeros(m.p.shape[1])
    np.add.at(cnt, m.t.T, 1)
    for name, s in (("sx", sx), ("sy", sy), ("txy", txy)):
        acc = np.zeros(m.p.shape[1])
        np.add.at(acc, m.t.T, s)
        out[name] = acc / cnt
    out["vm"] = np.sqrt(out["sx"] ** 2 - out["sx"] * out["sy"] + out["sy"] ** 2 + 3 * out["txy"] ** 2)
    out["u"] = u
    return out
