# Developing the app

## Run while editing
Double-click `run_app.bat`. Leave it running and edit the code in VS Code. **Every time you save a `.py` file, the browser reloads the app automatically** (set in `.streamlit/config.toml`).

## Where things live
| Want to change… | Edit | Notes |
|---|---|---|
| Layout, text, buttons, tables, colours of the UI | `app.py` | Sections are marked `# ---- sidebar`, `# ---- geometry`, `# ---- run`. Theme colours are in `.streamlit/config.toml` and at the top of `app.py`. |
| The interactive drawing (edge labels, arrows, click behaviour) | `app.py` → `geometry_figure()` | Plotly figure. |
| The results stress map | `app.py` → `result_figure()` | Matplotlib figure. |
| Contours, probe, max/min | `app.py` → `FIELDS`, `field_values()`, `contour_grid()`, `contour_figure()`, `probe_values()` | Plotly contour + invisible click layer for probes. Add a field by adding it to `FIELDS` and `field_values()`. |
| Mesh view | `app.py` → `mesh_figure()`, `mesh_stats()` | |
| The downloadable report | `app.py` → `report_text()` | Markdown text. |
| Which file types can be uploaded / how DXF is read | `geometry_io.py` | `read_dxf()` returns `outline`, `circles`, `poly_holes`. |
| Support types, loads, material model | `fe2d.py` → `solve_plane_stress()` (`fixed_arcs`, `radial_arcs`, `bearing_arcs` for holes), then `app.py` → `build_part()` and the UI | E.g. a roller support = clamp only one displacement component. |
| Force groups / multi-select | `app.py` → `edge_loads_from()`, `hole_forces_from()`, `hole_bc_full()` and the `03 · Supports + forces` block | Forces are stored as groups in `st.session_state.forces`. |
| Hole singularity scan | `sing_tool.py` → `scan_holes()` | Compares coarse vs fine stress around each hole. |
| How singular points are flagged / classified | `sing_tool.py` → `analyse()` | Threshold `RATIO_THRESHOLD`, angle `CANDIDATE_ANGLE`. |
| What the model sees at a corner | `corner_tools.py` → `corner_features()` | **Changing this needs regenerating data and retraining** (see README). |
| The ML model | `train_model.py` → produces `singularity_model.joblib` | |

Rule of thumb: UI changes (`app.py`, `geometry_io.py`) are safe. Changes to `corner_tools.py`, `fe2d.py` or `make_dataset.py` change the numbers, so retrain and re-check `make_figures.py` results afterwards.

## Team workflow (two people, no clashes)
```
git pull                                  # always start with the latest version
git checkout -b ui-results-table          # one branch per change, short descriptive name
# ...edit, save, check in the browser...
git add -A
git commit -m "Results table: show real stress first"
git push -u origin ui-results-table
```
Then on GitHub, open a **Pull Request** from that branch into `main`. The other person checks it, then clicks **Merge**. Afterwards:
```
git checkout main
git pull
```
Avoid both editing the same function at the same time. If you do, Git will ask you to resolve a conflict: open the file, keep the right lines, delete the `<<<<<<<` `=======` `>>>>>>>` markers, then `git add` + `git commit`.

## Feature ideas (good first changes)
- Roller / pin supports, pressure loads (`fe2d.py` + `build_part()` + UI)
- Upload a JSON/CSV outline as well as DXF (`geometry_io.py`)
- Per-corner radius editing in the results table (`analyse()` already supports `corner_radius_by_vertex`)
- PDF report export
- Show the stress-vs-mesh divergence plot for each flagged corner
