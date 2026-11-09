"""
test_parts.py - parts the model has NEVER seen (different shapes, loads and corner angles),
written as tool input files, plus the ground truth: the same part with real fillets, converged mesh.
"""
import json
import numpy as np

from fe2d import mesh_polygon, solve_plane_stress, interior_angles
from corner_tools import corner_length

t30 = np.tan(np.radians(30))
PARTS = {
    "stepped_bar": {   # shoulder in tension: two 270 deg corners
        "name": "Stepped bar in tension (shoulder)",
        "outline": [[0, 0], [60, 0], [60, 10], [120, 10], [120, 30], [60, 30], [60, 40], [0, 40]],
        "fixed": [[[0, 40], [0, 0]]],
        "traction": [[[120, 10], [120, 30], [100, 0]]],
        "corner_radius": 0.5,
    },
    "v_notch_beam": {  # cantilever with a 60 deg V-notch on top: 300 deg corner
        "name": "Cantilever with 60 deg V-notch",
        "outline": [[0, 0], [150, 0], [150, 40], [50 + 15 * t30, 40], [50, 25], [50 - 15 * t30, 40], [0, 40]],
        "fixed": [[[0, 40], [0, 0]]],
        "traction": [[[150, 0], [150, 40], [0, -10]]],
        "corner_radius": 1.0,
    },
    "chamfer_bracket": {  # bracket with a 45 deg inner chamfer: two 225 deg corners
        "name": "Bracket with chamfered inner corner",
        "outline": [[0, 0], [100, 0], [100, 20], [45, 20], [20, 45], [20, 100], [0, 100]],
        "fixed": [[[20, 100], [0, 100]]],
        "traction": [[[100, 0], [100, 20], [-15, -30]]],
        "corner_radius": 0.5,
    },
}


def write_parts():
    for key, p in PARTS.items():
        json.dump(p, open(f"part_{key}.json", "w"), indent=1)


def ground_truth(part, vertices, rho, div=12):
    poly = [np.array(p, float) for p in part["outline"]]
    size = np.ptp(np.vstack(poly), axis=0).max()
    m = mesh_polygon(poly, fillets={i: rho for i in vertices},
                     refine=[(i, rho / div) for i in vertices], h_global=size / 15)
    s = solve_plane_stress(m, [tuple(map(tuple, e)) for e in part["fixed"]],
                           [(e[0], e[1], tuple(e[2])) for e in part["traction"]])
    beta = interior_angles(poly)
    out = {}
    for i in vertices:
        psi = np.radians(360 - beta[i])
        d = np.linalg.norm(m.p.T - poly[i], axis=1)
        out[i] = float(s["vm"][d < 3 * rho / np.sin(psi / 2) + 0.05 * corner_length(poly, i)].max())
    return out


if __name__ == "__main__":
    write_parts()
    for key, p in PARTS.items():
        poly = [np.array(q, float) for q in p["outline"]]
        verts = [i for i, b in enumerate(interior_angles(poly)) if b > 180.5]
        print(key, {i: round(interior_angles(poly)[i]) for i in verts},
              ground_truth(p, verts, p["corner_radius"]))
