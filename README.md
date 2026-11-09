# FEM Stress-Singularity Finder and Real-Stress Predictor

**What it does:** you give it a 2D part (outline, supports, loads). It solves the part at two mesh sizes, flags every point where the FE stress keeps rising as the mesh is refined (a singularity), and labels each one as either a sharp re-entrant corner (geometry) or an artefact of an idealised support or load. For each sharp corner it predicts the finite peak stress the real part sees for a given corner radius.

## The app (recommended)
**Windows:** put the folder where you want it (e.g. `D:\MOS\app files`) and double-click `run_app.bat`. The first run creates a private Python environment (`.venv`) and download cache (`pip-cache`) inside that folder, so nothing is installed on C:. Delete the folder to remove everything. **Mac/Linux:** `./run_app.sh`. The app opens in your browser at http://localhost:8501.

1. **Part:** upload a DXF (in mm, or set the unit factor) or pick a sample. Lines, polylines, arcs (drawn fillets), circles (holes) and splines are read; the largest closed loop is the part outline.
2. **Material and section:** thickness, E, ν, the real corner radius at sharp corners (default 0.5 mm), yield strength σy and an optional yielding correction (Neuber's rule with a Ramberg-Osgood curve, hardening exponent n).
3. **Supports and forces:** click edges (E0, E1, …) and holes (H0, H1, …) on the drawing to build a selection (click again to remove), then apply to the whole selection at once:
   - **Fix selected:** edges become clamped supports; holes become fixed (bolted / press-fit).
   - **Pin support** (holes): a frictionless pin with 180° contact on the side you choose (90 = the part hangs on the pin).
   - **Add force:** Fx/Fy in N, either a total *split* across the selection (edges by length → uniform pressure; holes equally → bolt group) or the *same force on each* item. On holes, a force is a bolt **bearing load** (cosine pressure on the loaded half). Add several forces to the same holes to load a bolt in several directions; the bolt carries their resultant.
   - The app blocks a run if the supports can't hold the part (e.g. a single pin with nothing else).
4. **Run:** every sharp corner gets a card showing, side by side, the **singularity** (the FE value at the point, coarse → fine, flagged as not real), the **ML predicted real stress** for the chosen radius, and, if that is above yield, the **actual stress after local yielding** and the plastic strain (Neuber). Three result tabs: **Stress map** (summary), **Contours + probe** (von Mises, σx, σy, τxy, σ1, σ2, max shear, εx, εy, γxy, ε1; auto or manual range, number of bands, mono or ANSYS-style colours; hover for live values, click to pin probes P1, P2… with all values listed; ▲ max / ▼ min markers that skip singular spikes by default) and **Mesh** (coarse / fine mesh, zoom to any flagged point or hole, mesh statistics). Plus: stress map, sharp corners with their predicted real stress, support/load artefacts (including the edges of pin contact patches on holes), the converged peak stress on every hole, the highest converged stress elsewhere, and a downloadable report (Markdown) and results table (CSV). Changing the corner radius afterwards updates the predictions without re-solving.
5. **Optimise shape (topology optimisation):** after a run, section 05 removes material while keeping the part as stiff as possible for the same supports and forces. It uses SIMP (density method, penalty 3, density filter, optimality-criteria updates) on a triangle mesh. Holes with supports or loads and a thin band along clamped/loaded edges are always kept. Choose the material to keep (e.g. 40 %), the resolution (coarse/medium/fine) and whether to start from a solid part or the current one. The density field is turned into a clean outline (Taubin smoothing removes the mesh zig-zag) and every corner is filleted with the radius you set (default 3 mm; 0 keeps sharp corners so EigenStress predicts their real stress). The result shows the density field, the extracted shape (outline + cutouts, bolt holes kept as exact circles), material saved and stiffness compared with the solid part, and a **DXF download**. **Analyse optimised shape with EigenStress →** loads the new shape with the same supports and forces mapped onto it, so the corners created by the optimiser are checked for singularities and their real stress is predicted in one click. **Back to the original drawing** in the sidebar returns to the original.
6. **Design checks** (sidebar → *Design checks*), shown on every result:
   - **Required radius:** for an allowable stress σy / safety factor, the smallest fillet radius each sharp corner needs (or "a fillet alone cannot get there"). The model predicts in milliseconds, so it scans 80 radii per corner instead of one refined FE re-solve per trial. **Download DXF with required fillets** writes the part with every corner rounded to its required radius; upload it and run it to check.
   - **Fatigue → where it cracks first:** every hot spot (sharp corners, hole edges, the max elsewhere) ranked by fatigue life if the applied loads repeat (reversed ±F or on/off 0→F). Peterson notch sensitivity q = 1/(1 + a/r) turns the elastic peak into a fatigue stress (Kf), then an S-N line through 0.9 σu at 10³ and Se at 10⁶ cycles, with Goodman for on/off loading (Shigley). A steel-type estimate for comparing locations, not a certified life.
7. **ANSYS results instead of the built-in solver** (sidebar → *Stresses from: ANSYS results*): upload the ANSYS result of the same part, either the `.rst` file (Workbench: `<project>_files/dp0/SYS/MECH/file.rst`) or a CSV with node coordinates and stresses. 2D and 3D plate models both work: the plane matching the DXF is found automatically, the outer face of a 3D model is used, coordinates are shifted onto the DXF, and m / Pa models are converted to mm / MPa. Upload a coarse result too, and singular points (sharp corners and the idealised supports/loads inside the ANSYS model) are confirmed by mesh refinement exactly as with the built-in solver. Reading `.rst` needs `pip install ansys-dpf-core` (uses the ANSYS installation) or `pip install ansys-mapdl-reader` (no ANSYS needed); `setup_ansys.bat` installs both. The online app takes CSV.
8. **Calibrated 90 % range on every prediction** (split conformal prediction, `build_conformal.py` → `conformal.json`): bins by corner kind × ρ/l from the held-out test residuals, with a ±7.1 % floor calibrated on 68 independent validation cases. Coverage: 90.6 % on held-out geometries, 94 % (leave-one-out) on the independent cases. *Required radius with 90 % confidence* sizes fillets on the top of the range.
9. **Singularity fingerprint:** each flagged corner's measured exponent λ = 1 − ln(σ_fine/σ_coarse)/ln(h_c/h_f) is shown next to Williams' λ; a departure of more than 15 % is flagged (mesh not resolving the corner, or a neighbouring feature dominating). Across 5,845 training geometries 90 % of corners grow within 7 % of theory (cutouts 15 %).

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

## Rebuild everything from scratch (~1.5 h on 2 cores; add `--workers N` to use more)
```
python make_dataset.py --family bracket --n 2000 --out dataset.csv
python make_dataset.py --family notched_plate --n 1500 --seed0 100000 --out dataset_notch.csv
python make_dataset.py --family bracket --n 600 --seed0 200000 --out dataset_bracket2.csv
python make_dataset.py --family stepped --n 900 --seed0 300000 --rho-min 0.01 --rho-max 1.0 --out dataset_stepped.csv
python make_dataset.py --family bracket --n 800 --seed0 400000 --rho-min 0.3 --rho-max 1.0 --out dataset_bracket_bigr.csv
python make_dataset.py --family notched_plate --n 700 --seed0 500000 --rho-min 0.3 --rho-max 1.0 --out dataset_notch_bigr.csv
python make_dataset.py --family window --n 1500 --seed0 600000 --rho-min 0.01 --rho-max 1.0 --out dataset_window.csv
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
| `app.py` | The web app (upload DXF, click supports/forces, run, optimise shape) |
| `design_tools.py` | Required radius per corner, Peterson/Shigley fatigue ranking, fillets → DXF, calibrated 90 % range |
| `build_conformal.py` | Builds `conformal.json` (prediction-interval calibration) and prints both coverage checks |
| `ansys_io.py` | Reads ANSYS `.rst` (DPF or mapdl-reader) or CSV results, maps them onto the DXF plane |
| `topopt.py` | Topology optimisation (SIMP on P1 triangles, density filter, OC update), shape extraction and DXF export |
| `geometry_io.py` | DXF reader (lines, polylines with bulges, arcs, circles, splines) + sample drawings |
| `sing_tool.py` | Command-line tool (built-in solver or ANSYS CSVs) |
| `test_parts.py` | 3 unseen parts + filleted ground truth |
| `make_figures.py` | Report figures |
| `ansys_validate.py` | Automatic EigenStress-vs-ANSYS validation via PyMAPDL (sharp runs at 2/1/0.5 mm, filleted ground truth, comparison table). `--solver python` runs the same pipeline with the built-in solver |
| `generate_lbracket_data.py` | PyMAPDL script, starting point for running the data generation in ANSYS |

## Method in one paragraph
Near a sharp corner with material angle β, elastic stress follows σ = K1·r^(λ−1), where λ comes from Williams' equation sin(βλ) + λ·sin β = 0 (λ = 0.545 at 270°). The stress at the corner node is therefore infinite, and the FE value is set purely by the mesh. The stresses a short distance away (0.1l to 0.3l along the corner bisector) do converge. From them the tool fits K1 and records the stress profile; tested, this reading changes by less than 0.5% between meshes of l/10 and l/160. A real corner has a radius ρ, so its peak stress is finite and set by K1, ρ and the corner angle. A gradient-boosted model learns this mapping from about 49,000 FE pairs (sharp vs filleted). All inputs are dimensionless, so one model covers any size, load and material in linear elasticity.

## Results
Model v3: gradient-boosted trees trained on about 91,000 sharp-vs-filleted FE pairs from four shape families: angled brackets, V-notched plates, stepped bars/shoulders, and plates with polygonal **cutouts (windows)**, 3–5 sides, any orientation, mixed loads. Corner angles 190–330°, fillet radius 1–100 % of the local feature size l.

**Held-out test set** (1,566 geometries never used for training, 18,240 cases)

| Method | Mean error | Median | 95th percentile |
|---|---|---|---|
| FE corner-node stress (current practice) | 57.1% | 39.2% | 192% |
| Physics formula (Williams K1 + 3 constants) | 31.7% | 24.1% | 80% |
| **EigenStress ML model** | **4.1%** | **1.6%** | **18%** |

Brackets, notches and steps: 2.9 % mean (1.3 % median). Cutout corners: 9.3 % mean. By fillet size:

| r/l | Brackets, notches, steps | Cutout corners |
|---|---|---|
| ≤ 0.3 | 2.0 % | 6.5 % |
| 0.3–0.6 | 4.3 % | 19 % |
| 0.6–1.0 | 8.9 % | 23 % |

Error is flat with mesh size. Typical design radii (0.5–2 mm on parts of tens of mm) are in the r/l ≤ 0.3 band.

## Validation against independent references (`validate_reference.py`)
None of these were used for training.

| Check | Reference | Result |
|---|---|---|
| A. Plate with hole in tension (fig5) | Kirsch exact elasticity solution | Peak hoop stress 302.7 vs 300 MPa (+0.9 %); von Mises field within 5 % of applied stress everywhere within 6 hole radii |
| B. Cantilever, tip load (fig6) | Beam theory σ = My/I, parabolic shear | Bending stress within 0.01 %, shear within 0.5 % |
| C. Stepped flat bar, shoulder fillets (fig7) | Peterson's Kt (Pilkey curve fits), 3 D/d ratios × 4 radii | **EigenStress from the sharp model: 4.75 % mean error**; FE with the fillet modelled: 3.3 %; naive sharp-node reading: 82.6 % |
| D. Bolt / pin / bearing load cases (`validation_hole_loads.csv`) | Filleted FE ground truth | 1.9 % mean, 8.5 % max over 15 cases |
| E. Unseen part shapes (`unseen_parts_results.csv`) | Filleted FE ground truth | mean 2.5 %, max 4.1 % |
| G. Cutout corners: square window in tension, rotated window in bending, r = 0.5/1/2 mm (`validation_cutout_corners.csv`) | Filleted FE ground truth | **3.2 % mean, 8.1 % max** (v2 without the window family: 10–28 %; sharp-node reading: 22 %) |
| F. Sharp FE results read from files, 4 parts × 3 radii (`python_validation_results.csv`) | Filleted FE ground truth | 1.3 % mean, 4.8 % max (sharp 0.5 mm reading: 21 %). The same run with ANSYS: `ansys_validate.py` |
| **H. ANSYS Mechanical APDL 2025 R1** (PLANE183 6-node triangles), 4 parts × 3 radii, 18 corners (`ansys_validation_results.csv`) | ANSYS filleted FE (mesh change ≤ 2 % between r/8 and r/16) | **EigenStress on the ANSYS sharp results: 2.3 % mean, 5.5 % max** (sharp 0.5 mm reading: 20 % mean, 47 % max). ANSYS and the built-in solver agree on the filleted peaks within 0.7 % |
| I. Required radius (`validation_required_radius.csv`): for 17 corner / target pairs on the 4 validation parts, the corner was filleted at the radius EigenStress said it needs and re-solved with FE | Filleted FE peak vs target stress | **1.2 % mean, 2.7 % max** (one target correctly reported as unreachable by a fillet) |
| J. Calibrated 90 % ranges (`build_conformal.py`) | Held-out half of the test set (9,136 cases); 68 independent cases leave-one-out | 90.6 % and 94 % coverage |
| K. **Hard out-of-distribution test** (`hard_test_lever.py` → `hard_test_lever.csv`): all 84 sharp corners of the topology-optimised lever (`samples/lever_optimised_sharp.dxf`), thin struts, close corners, bearing loads, ρ = 0.5 and 1 mm, every corner filleted and re-solved | Filleted FE (ρ/8 and ρ/16) | **11–15 % mean, 6 % median** (raw sharp value 41–51 %); 85 % inside the 90 % range. The 2–5 % figures above are for isolated corners with small fillets |

**Model history (kept honest).** v1 (brackets + notches, r/l ≤ 0.3) scored 28.9 % against Peterson because stepped shoulders with fillets comparable to the step height were outside its training range. v2 added a stepped-bar family and radii up to l, which brought Peterson to 4.9 % without losing accuracy elsewhere. v2 underpredicted corners of cutouts (windows) in tension by 10–28 %, which matters for topology-optimised parts. v3 added a window family; cutout-corner validation dropped to 3.2 % while every other check stayed within 0.5 % of v2. (`singularity_model_v2.joblib` is kept for comparison.) Note that stepped bars are therefore no longer an unseen *shape family*; Peterson remains an independent reference. The largest remaining errors (10–14 %) are for fillets of 0.8 × the step height, at the edge of geometric possibility.

## Limitations (state these honestly)
- App supports: clamped edges, fixed / pinned / bolt-loaded circular holes, and edge or hole forces (single or grouped). Pin contact is a fixed 180° frictionless patch (no gap, no friction): its ends are flagged as artefacts; true contact needs a nonlinear contact analysis. No rollers, pressures or point loads yet.
- 2D plane stress. The ML prediction is linear-elastic; above yield the app applies Neuber's rule (a standard engineering estimate, typically conservative for plane stress), not a full elastic-plastic FE analysis. Plasticity at the corner would lower real peaks above yield.
- Only sharp re-entrant corners are predicted. Support and point-load singularities are flagged but not predicted, because their "real" value depends on how the real support is built.
- The corner radius ρ must be given (the default is 0.5 mm). Without a length scale, the finite stress is undefined. Alternatives are a material critical distance (theory of critical distances) or a strain-gauge length.
- Validity range: corner angles 190–330°, ρ/l from 0.01 to 1.0 (best below 0.3; for cutout corners, about 20 % error above 0.3), fine mesh h ≤ 0.12·l at the corner. About 2.5% of test cases have errors above 20%. These are mostly very low-stress corners where other features dominate the local field.
- Two corners closer than about 2l interact. The chamfered bracket (+4.5%) shows this effect is small but present.
- Topology optimisation is 2D, linear-elastic, minimum compliance (stiffness) at a fixed volume, single load case. It does not include a stress constraint; that is why the optimised shape is then run through EigenStress. The fillets are polyline arcs (no vertex turns more than 8°); rebuild them as true arcs in CAD for manufacture.
- Next steps: 3D parts (STEP → ANSYS Mechanical via PyMechanical), plasticity, and validation against strain-gauge tests.
