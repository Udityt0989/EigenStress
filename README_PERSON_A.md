# EigenStress – ANSYS runs (Person A)

This folder is the complete EigenStress project. You only need the two ANSYS scripts:

| Script | What it does | Time |
|---|---|---|
| `ansys_validate.py` | 4 validation parts: sharp-corner runs (h = 2 / 1 / 0.5 mm) + filleted "truth" runs (r = 0.5 / 1 / 2 mm), then compares EigenStress with the ANSYS truth | ~5 min |
| `ansys_make_dataset.py` | Training data with ANSYS as the solver (same parts and CSV format as our Python data) | 10–30 s per part |

Everything is 2D plane stress, PLANE183 **6-node triangles**, E = 210 GPa, ν = 0.3, thickness 1 mm. ANSYS 2025 R1 (v251) with PyMAPDL.

## 1. Setup (once)
1. Python 3.10–3.12 must be installed (python.org, tick **Add python.exe to PATH**).
2. Unzip this folder somewhere without spaces in the path if possible, e.g. `D:\EigenStress`.
3. Double-click **`setup_ansys.bat`**. It creates `.venv` inside the folder and installs everything (including `ansys-mapdl-core`).

All commands below are run in **cmd** (not PowerShell) inside the folder:
```
cd /d D:\EigenStress
```

## 2. Validation run (do this first)
```
.venv\Scripts\python.exe ansys_validate.py --parts lbracket
```
About 1 min. If it works, run all 4 parts:
```
.venv\Scripts\python.exe ansys_validate.py
```
**Check before sending:** in `ansys_out\ansys_validation_results.csv`, the column `fillet_mesh_change_%` must be within about **±2 %** on every row. (The previous run had 55–140 % because of quad elements and an abrupt mesh jump at the fillet; both are fixed in this version.)

**Send back:** the whole `ansys_out` folder, zipped. It now also contains the fillet CSVs (`<part>_fillet_r<r>_div8/16.csv`).

## 3. Training data
Quick test first (2 parts, ~1 min):
```
.venv\Scripts\python.exe ansys_make_dataset.py --family bracket --n 2 --out ansys_test.csv
```
Then the real runs (each can be stopped with Ctrl+C and restarted with the same command; finished parts are skipped):
```
.venv\Scripts\python.exe ansys_make_dataset.py --family bracket       --n 200 --seed0 0      --out ansys_bracket.csv
.venv\Scripts\python.exe ansys_make_dataset.py --family notched_plate --n 150 --seed0 100000 --out ansys_notch.csv
.venv\Scripts\python.exe ansys_make_dataset.py --family stepped       --n 150 --seed0 300000 --rho-max 1.0 --out ansys_stepped.csv
.venv\Scripts\python.exe ansys_make_dataset.py --family window        --n 150 --seed0 600000 --rho-max 1.0 --out ansys_window.csv
```
**Send back:** the four `ansys_*.csv` files.

Options:
- `--workers 2` – two ANSYS sessions in parallel (needs 2 solver licences), about half the time.
- `--exec "C:\Program Files\ANSYS Inc\v251\ansys\bin\winx64\ANSYS251.exe"` – if PyMAPDL cannot find ANSYS.
- A line "seed N failed: …" is fine now and then (that part is skipped). If **every** seed fails, send the message.

## 4. Use your own ANSYS models in the app
`run_app.bat` starts the app. In the sidebar choose **Stresses from → ANSYS results**, upload the DXF of the part
and its ANSYS result (`file.rst` from `<project>_files\dp0\SYS\MECH\`, or a CSV export). Upload a second result
with a coarser mesh to confirm singularities by refinement. 2D or 3D plate models, any origin, mm or m units.

## 5. If something goes wrong
- `ansys-mapdl-core` cannot start MAPDL → use the `--exec` option above (check the path on your PC), and make sure no other ANSYS session holds the licence.
- Any Python error → send a screenshot of the full red text.

## What not to change
- Mesh settings, element type, units and part definitions: EigenStress compares numbers across ANSYS and the Python solver, so they must be identical.
- `singularity_model.joblib` (the trained model).
