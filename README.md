# FEM Stress-Singularity Finder and Real-Stress Predictor

**What it does:** you give it a 2D part (outline, supports, loads). It solves the part at two mesh sizes, flags every point where the FE stress keeps rising as the mesh is refined (a singularity), and labels each one as either a sharp re-entrant corner (geometry) or an artefact of an idealised support or load. For each sharp corner it predicts the finite peak stress the real part sees for a given corner radius.

## The app (recommended)
**Windows:** put the folder where you want it (e.g. `D:\MOS\app files`) and double-click `run_app.bat`. The first run creates a private Python environment (`.venv`) and download cache (`pip-cache`) inside that folder, so nothing is installed on C:. Delete the folder to remove everything. **Mac/Linux:** `./run_app.sh`. The app opens in your browser at http://localhost:8501.

1. **Part:** upload a DXF (in mm, or set the unit factor) or pick a sample. Lines, polylines, arcs (drawn fillets), circles (holes) and splines are read; the largest closed loop is the part outline.
2. **Material and section:** thickness, E, ν, and the real corner radius at sharp corners (default 0.5 mm).
3. **Supports and forces:** click an edge label (E0, E1, …) on the drawing, then either *Fix this edge* or enter Fx/Fy in N and *Apply force*. A force is the total on that edge, spread evenly.
4. **Run:** stress map, sharp corners with their predicted real stress, support/load artefacts, the highest converged stress elsewhere, and a downloadable report (Markdown) and results table (CSV). Changing the corner radius afterwards updates the predictions without re-solving.

To share it online for free: push this folder to GitHub, then deploy `app.py` on Streamlit Community Cloud (`packages.txt` installs the system libraries gmsh needs).

For changing the UI or adding features, see **DEVELOPING.md**.

## Setup (command line only)
```
pip install gmsh scikit-fem scikit-learn pandas matplotlib joblib scipy
# Linux only, if gmsh complains about libGLU:  sudo apt install libglu1-mesa libxcursor1 libxinerama1 libxft2
```

## Run the tool
```
python sing_tool.py part_stepped_bar.json                         # built-in FE solver
python sing_tool.py part.json --ansys coarse.csv fine.csv --h 0.5 0.25   # ANSYS exports (Person A)
```
ANSYS CSV columns: `X, Y, SX, SY, SXY, SEQV` (all nodes). `--h` is the element size at the corners in each mesh. The fine mesh needs h ≤ 0.12·l at each corner (l = distance from the corner to the nearest non-adjacent edge).

## Rebuild everything from scratch (~20 min)
```
python make_dataset.py --family bracket --n 2000 --out dataset.csv
python make_dataset.py --family notched_plate --n 1500 --seed0 100000 --out dataset_notch.csv
python make_dataset.py --family bracket --n 600 --seed0 200000 --out dataset_bracket2.csv
python train_model.py       # -> singularity_model.joblib, test_predictions.csv, results_summary.csv
python test_parts.py        # -> part_*.json (unseen validation parts)
python make_figures.py      # -> fig1..fig4, unseen_parts_results.csv
```

## Files
| File | Role |
|---|---|
| `fe2d.py` | 2D plane-stress FE solver: gmsh mesh, 6-node quadratic triangles (like ANSYS PLANE183), averaged nodal stresses |
| `corner_tools.py` | Williams eigenvalue λ, corner length scale l, bisector sampling, mesh-independent K1 fit, singularity check |
| `make_dataset.py` | Paired data: sharp corner at 4 meshes (input) vs filleted corner at 3 radii (target) |
| `train_model.py` | Naive vs physics formula vs ML, evaluated on held-out geometries |
| `app.py` | The web app (upload DXF, click supports/forces, run) |
| `geometry_io.py` | DXF reader (lines, polylines with bulges, arcs, circles, splines) + sample drawings |
| `sing_tool.py` | Command-line tool (built-in solver or ANSYS CSVs) |
| `test_parts.py` | 3 unseen parts + filleted ground truth |
| `make_figures.py` | Report figures |
| `generate_lbracket_data.py` | PyMAPDL script, starting point for running the data generation in ANSYS |

## Method in one paragraph
Near a sharp corner with material angle β, elastic stress follows σ = K1·r^(λ−1), where λ comes from Williams' equation sin(βλ) + λ·sin β = 0 (λ = 0.545 at 270°). The stress at the corner node is therefore infinite, and the FE value is set purely by the mesh. The stresses a short distance away (0.1l to 0.3l along the corner bisector) do converge. From them the tool fits K1 and records the stress profile; tested, this reading changes by less than 0.5% between meshes of l/10 and l/160. A real corner has a radius ρ, so its peak stress is finite and set by K1, ρ and the corner angle. A gradient-boosted model learns this mapping from about 49,000 FE pairs (sharp vs filleted). All inputs are dimensionless, so one model covers any size, load and material in linear elasticity.

## Results
**Held-out test set:** 814 geometries never used in training, 9,768 cases.

| Method | Mean error | Median | 95th percentile |
|---|---|---|---|
| FE corner-node stress (current practice) | 42.7% | 33.6% | 123% |
| Physics formula (Williams K1 + 3 constants) | 16.4% | 11.7% | 36% |
| **ML model** | **2.1%** | **1.2%** | **5.5%** |

The ML error stays at 2.0–2.4% from coarse (l/10) to fine (l/80) meshes, while the naive error ranges from 33% to 60%.

**Unseen part shapes** (none of these shapes is in the training data):

| Part | Corner | True peak (MPa) | FE fine-mesh node | Model |
|---|---|---|---|---|
| Stepped bar, tension | 270°, r = 0.5 mm | 421 | 399 (−5%) | 442 (+5.0%) |
| Cantilever, 60° V-notch | 300°, r = 1 mm | 1446 | 1716 (+19%) | 1433 (−0.9%) |
| Chamfered bracket, corner 1 | 225°, r = 0.5 mm | 1748 | 1340 (−23%) | 1756 (+0.5%) |
| Chamfered bracket, corner 2 | 225°, r = 0.5 mm | 3303 | 2573 (−22%) | 3450 (+4.5%) |

## Limitations (state these honestly)
- App supports: fixed (clamped) edges and total forces on edges only. No rollers, pressures or point loads yet.
- 2D plane stress, linear elastic material only. Plasticity at the corner would lower real peaks above yield.
- Only sharp re-entrant corners are predicted. Support and point-load singularities are flagged but not predicted, because their "real" value depends on how the real support is built.
- The corner radius ρ must be given (the default is 0.5 mm). Without a length scale, the finite stress is undefined. Alternatives are a material critical distance (theory of critical distances) or a strain-gauge length.
- Validity range: corner angles 190–330°, ρ/l from 0.01 to 0.3, fine mesh h ≤ 0.12·l at the corner. About 2.5% of test cases have errors above 20%. These are mostly very low-stress corners where other features dominate the local field.
- Two corners closer than about 2l interact. The chamfered bracket (+4.5%) shows this effect is small but present.
- Next steps: 3D parts (STEP → ANSYS Mechanical via PyMechanical), plasticity, and validation against strain-gauge tests.
