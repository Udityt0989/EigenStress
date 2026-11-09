"""
geometry_io.py - turn a 2D CAD drawing (DXF) into the tool's geometry.

Supported DXF entities: LINE, LWPOLYLINE / POLYLINE (incl. bulges), ARC, CIRCLE, SPLINE (approximated).
* The largest closed loop becomes the outer boundary.
* CIRCLEs inside it become holes. Other closed loops inside it become polygonal holes.
* Arcs are split into short straight segments (<= 5 deg turn each), so a real fillet in the drawing is
  never mistaken for a sharp corner (the tool only treats turns > 190 deg as singular candidates).
Units: the drawing is assumed to be in mm unless a scale factor is given.
"""
import math
from collections import defaultdict

import numpy as np

ARC_STEP_DEG = 5.0


def _arc_points(c, r, a0, a1):
    """Points on an arc from angle a0 to a1 (deg, CCW), endpoints included."""
    while a1 <= a0:
        a1 += 360
    n = max(2, int(math.ceil((a1 - a0) / ARC_STEP_DEG)))
    t = np.radians(np.linspace(a0, a1, n + 1))
    return [(c[0] + r * math.cos(x), c[1] + r * math.sin(x)) for x in t]


def _bulge_points(p0, p1, bulge):
    if abs(bulge) < 1e-12:
        return [p0, p1]
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    theta = 4 * math.atan(bulge)                    # included angle, signed
    chord = np.linalg.norm(p1 - p0)
    r = chord / (2 * math.sin(abs(theta) / 2))
    mid = (p0 + p1) / 2
    d = (p1 - p0) / chord
    nrm = np.array([-d[1], d[0]])
    h = r * math.cos(abs(theta) / 2)
    c = mid + nrm * h * (1 if bulge > 0 else -1)
    a0 = math.degrees(math.atan2(*(p0 - c)[::-1]))
    a1 = math.degrees(math.atan2(*(p1 - c)[::-1]))
    if bulge > 0:
        pts = _arc_points(c, r, a0, a1)
    else:
        pts = _arc_points(c, r, a1, a0)[::-1]
    pts[0], pts[-1] = tuple(p0), tuple(p1)
    return pts


def read_dxf(path_or_stream, scale=1.0):
    import ezdxf
    if hasattr(path_or_stream, "read"):
        import io
        data = path_or_stream.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="ignore")
        doc = ezdxf.read(io.StringIO(data))
    else:
        doc = ezdxf.readfile(path_or_stream)
    msp = doc.modelspace()

    chains, circles = [], []                        # chains: list of point lists (open or closed)
    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            chains.append([tuple(e.dxf.start)[:2], tuple(e.dxf.end)[:2]])
        elif t == "ARC":
            chains.append(_arc_points(tuple(e.dxf.center)[:2], e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle))
        elif t == "CIRCLE":
            circles.append((tuple(e.dxf.center)[:2], e.dxf.radius))
        elif t == "LWPOLYLINE":
            pts = list(e.get_points("xyb"))
            closed = e.closed
            seq = []
            n = len(pts)
            for k in range(n if closed else n - 1):
                x0, y0, b = pts[k]
                x1, y1, _ = pts[(k + 1) % n]
                seg = _bulge_points((x0, y0), (x1, y1), b)
                seq += seg if not seq else seg[1:]
            chains.append(seq)
        elif t == "POLYLINE":
            pts = [tuple(v.dxf.location)[:2] for v in e.vertices]
            if e.is_closed:
                pts.append(pts[0])
            chains.append(pts)
        elif t == "SPLINE":
            pts = [tuple(p)[:2] for p in e.flattening(0.01)]
            chains.append(pts)
    chains = [[(x * scale, y * scale) for x, y in c] for c in chains]
    circles = [((c[0] * scale, c[1] * scale), r * scale) for c, r in circles]
    loops = _join_chains(chains)
    if not loops:
        raise ValueError("No closed outline found in the drawing. Make sure the part boundary is a closed loop.")
    loops = [_clean(l) for l in loops]
    loops.sort(key=lambda l: abs(_area(l)), reverse=True)
    outer = loops[0]
    if _area(outer) < 0:
        outer = outer[::-1]
    poly_holes = []
    for l in loops[1:]:
        if _inside(outer, l[0]):
            poly_holes.append(l[::-1] if _area(l) > 0 else l)   # holes clockwise
    hole_circles = [(c, r) for c, r in circles if _inside(outer, c)]
    return {"outline": outer, "circles": hole_circles, "poly_holes": poly_holes}


def _join_chains(chains, tol=None):
    allpts = np.array([p for c in chains for p in c]) if chains else np.zeros((0, 2))
    if len(allpts) == 0:
        return []
    if tol is None:
        tol = 1e-6 * max(np.ptp(allpts, axis=0).max(), 1.0) * 100
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    loops, open_chains = [], []
    for c in chains:
        if len(c) > 2 and np.linalg.norm(np.subtract(c[0], c[-1])) < tol:
            loops.append(c[:-1])
        else:
            open_chains.append(list(c))
    # greedy chaining of open pieces
    ends = defaultdict(list)
    for i, c in enumerate(open_chains):
        ends[key(c[0])].append(i)
        ends[key(c[-1])].append(i)
    used = [False] * len(open_chains)
    for i in range(len(open_chains)):
        if used[i]:
            continue
        used[i] = True
        path = list(open_chains[i])
        while True:
            k = key(path[-1])
            nxt = [j for j in ends[k] if not used[j]]
            if not nxt:
                break
            j = nxt[0]
            used[j] = True
            c = open_chains[j]
            path += (c[1:] if key(c[0]) == k else c[::-1][1:])
        if len(path) > 3 and key(path[0]) == key(path[-1]):
            loops.append(path[:-1])
    return loops


def _clean(loop, tol=1e-9):
    """Drop duplicate points and exactly collinear midpoints."""
    pts = [np.asarray(p, float) for p in loop]
    out = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > 1e-9 * (1 + np.abs(p).max()):
            out.append(p)
    if np.linalg.norm(out[0] - out[-1]) < 1e-9 * (1 + np.abs(out[0]).max()):
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cr) < tol * np.linalg.norm(c - a) ** 2 and np.dot(b - a, c - b) > 0:
                out.pop(i)
                changed = True
                break
    return [tuple(p) for p in out]


def _area(loop):
    x, y = np.asarray(loop, float).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _inside(loop, pt):
    from matplotlib.path import Path
    return Path(np.asarray(loop, float)).contains_point(pt)


def write_sample_dxfs(folder="."):
    """Example drawings (mm) for trying the app."""
    import ezdxf
    t30 = math.tan(math.radians(30))

    def save(name, outline, circles=(), arcs=()):
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_lwpolyline(outline, close=True)
        for c, r in circles:
            msp.add_circle(c, r)
        doc.saveas(f"{folder}/{name}.dxf")

    save("sample_L_bracket_with_holes",
         [(0, 0), (120, 0), (120, 25), (25, 25), (25, 120), (0, 120)],
         circles=[((12.5, 105), 5), ((105, 12.5), 5)])
    save("sample_chamfer_bracket", [(0, 0), (100, 0), (100, 20), (45, 20), (20, 45), (20, 100), (0, 100)])
    save("sample_v_notch_beam",
         [(0, 0), (150, 0), (150, 40), (50 + 15 * t30, 40), (50, 25), (50 - 15 * t30, 40), (0, 40)])
    # stepped bar where ONE shoulder already has a 3 mm fillet drawn (arc via bulge) and one is sharp
    doc = ezdxf.new()
    msp = doc.modelspace()
    b = math.tan(math.radians(90) / 4)
    msp.add_lwpolyline([(0, 0, 0), (60, 0, 0), (60, 7, -b), (63, 10, 0), (120, 10, 0), (120, 30, 0),
                        (60, 30, 0), (60, 40, 0), (0, 40, 0)], format="xyb", close=True)
    doc.saveas(f"{folder}/sample_stepped_bar_one_fillet.dxf")
