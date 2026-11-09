# EigenStress: FEM Singularity Finder and Real-Stress Predictor

**What it does:** you give it a 2D part (outline, supports, loads). It solves the part at two mesh sizes, flags every point where the FE stress keeps rising as the mesh is refined (a singularity), and labels each one as either a sharp re-entrant corner (geometry) or an artefact of an idealised support or load. For each sharp corner it predicts the finite peak stress the real part sees for a given corner radius.

## The app (recommended)
**Windows:** put the folder where you want it (e.g. `D:\MOS\app files`) and double-click `run_app.bat`. The first run creates a private Python environment (`.venv`) and download cache (`pip-cache`) inside that folder, so nothing is installed on C:. Delete the folder to remove everything. **Mac/Linux:** `./run_app.sh`. The app opens in your browser at http://localhost:8501.

1. **Part:** upload a DXF (in mm, or set the unit factor) or pick a sample. Lines, polylines, arcs (drawn fillets), circles (holes) and splines are read; the largest closed loop is the part outline.
2. **Material and section:** thickness, E, ν, and the real corner radius at sharp corners (default 0.5 mm).
3. **Supports and forces:** click edges (E0, E1, …) and holes (H0, H1, …) on the drawing to build a selection (click again to remove), then apply to the whole selection at once:
   - **Fix selected:** edges become clamped supports; holes become fixed (bolted / press-fit).
   - **Pin support** (holes): a frictionless pin with 180° contact on the side you choose (90 = the part hangs on the pin).
   - **Add force:** Fx/Fy in N, either a total *split* across the selection (edges by length → uniform pressure; holes equally → bolt group) or the *same force on each* item. On holes, a force is a bolt **bearing load** (cosine pressure on the loaded half). Add several forces to the same holes to load a bolt in several directions; the bolt carries their resultant.
   - The app blocks a run if the supports can't hold the part (e.g. a single pin with nothing else).
4. **Run:** three result tabs: **Stress map** (summary), **Contours + probe** (von Mises, σx, σy, τxy, σ1, σ2, max shear, εx, εy, γxy, ε1; auto or manual range, number of bands, mono or ANSYS-style colours; hover for live values, click to pin probes P1, P2… with all values listed; ▲ max / ▼ min markers that skip singular spikes by default) and **Mesh** (coarse / fine mesh, zoom to any flagged point or hole, mesh statistics). Plus: stress map, sharp corners with their predicted real stress, support/load artefacts (including the edges of pin contact patches on holes), the converged peak stress on every hole, the highest converged stress elsewhere, and a downloadable report (Markdown) and results table (CSV). Changing the corner radius afterwards updates the predictions without re-solving.

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
python validate_reference.py  # -> fig5..fig7, validation_reference.csv (Kirsch, beam theory, Peterson)
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
Model v2: gradient-boosted trees trained on 74,480 sharp-vs-filleted FE pairs from three shape families (angled brackets, V-notched plates, stepped bars/shoulders), corner angles 190–330°, fillet radius 1–100 % of the local feature size l.

**Held-out test set** (1,270 geometries never used for training, 14,920 cases)

| Method | Mean error | Median | 95th percentile |
|---|---|---|---|
| FE corner-node stress (current practice) | 59.3% | 40.2% | 198% |
| Physics formula (Williams K1 + 3 constants) | 20.7% | 16.2% | 49% |
| **EigenStress ML model** | **2.6%** | **1.2%** | **9.6%** |

By fillet size: r/l ≤ 0.3 → 1.8 % mean; 0.3–0.6 → ~4 %; 0.6–1.0 → ~8 %. Error is flat with mesh size (2.9–3.3 % from l/80 to l/10).

## Validation against independent references (`validate_reference.py`)
None of these were used for training.

| Check | Reference | Result |
|---|---|---|
| A. Plate with hole in tension (fig5) | Kirsch exact elasticity solution | Peak hoop stress 302.7 vs 300 MPa (+0.9 %); von Mises field within 5 % of applied stress everywhere within 6 hole radii |
| B. Cantilever, tip load (fig6) | Beam theory σ = My/I, parabolic shear | Bending stress within 0.01 %, shear within 0.5 % |
| C. Stepped flat bar, shoulder fillets (fig7) | Peterson's Kt (Pilkey curve fits), 3 D/d ratios × 4 radii | **EigenStress from the sharp model: 4.9 % mean error**; FE with the fillet modelled: 3.3 %; naive sharp-node reading: 82.6 % |
| D. Bolt / pin / bearing load cases (`validation_hole_loads.csv`) | Filleted FE ground truth | 1.6 % mean, 8 % max over 15 cases |
| E. Unseen part shapes (`unseen_parts_results.csv`) | Filleted FE ground truth | 1.0–5.2 % (mean 2.3 %) |

**Model history (kept honest).** v1 (brackets + notches, r/l ≤ 0.3) scored 28.9 % against Peterson because stepped shoulders with fillets comparable to the step height were outside its training range. v2 added a stepped-bar family and radii up to l, which brought Peterson to 4.9 % without losing accuracy elsewhere. Note that stepped bars are therefore no longer an unseen *shape family*; Peterson remains an independent reference. The largest remaining errors (10–14 %) are for fillets of 0.8 × the step height, at the edge of geometric possibility.

## Limitations (state these honestly)
- App supports: clamped edges, fixed / pinned / bolt-loaded circular holes, and edge or hole forces (single or grouped). Pin contact is a fixed 180° frictionless patch (no gap, no friction): its ends are flagged as artefacts; true contact needs a nonlinear contact analysis. No rollers, pressures or point loads yet.
- 2D plane stress, linear elastic material only. Plasticity at the corner would lower real peaks above yield.
- Only sharp re-entrant corners are predicted. Support and point-load singularities are flagged but not predicted, because their "real" value depends on how the real support is built.
- The corner radius ρ must be given (the default is 0.5 mm). Without a length scale, the finite stress is undefined. Alternatives are a material critical distance (theory of critical distances) or a strain-gauge length.
- Validity range: corner angles 190–330°, ρ/l from 0.01 to 1.0 (best below 0.3), fine mesh h ≤ 0.12·l at the corner. About 2.5% of test cases have errors above 20%. These are mostly very low-stress corners where other features dominate the local field.
- Two corners closer than about 2l interact. The chamfered bracket (+4.5%) shows this effect is small but present.
- Next steps: 3D parts (STEP → ANSYS Mechanical via PyMechanical), plasticity, and validation against strain-gauge tests.
