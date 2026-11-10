"""
train_surrogate.py - offline: FE snapshots -> POD-RBF surrogates, plus a held-out accuracy check.
Usage: python train_surrogate.py            (writes surrogate_<family>.npz and surrogate_validation.csv)
"""
import sys
import numpy as np
import pandas as pd

from surrogate import FAMILIES, generate, train, Surrogate, CHANNELS, LAM_270

SIZES = {"lbracket": (260, 40), "stepped": (140, 30)}


def evaluate(sur, data):
    n = data["n_train"]
    rows = []
    for p, snap, arcs, C in zip(data["P"][n:], data["snaps"][n:], data["arcs"][n:], data["C"][n:]):
        f = sur.fields(p)
        nch = len(CHANNELS)
        true = snap.reshape(2, nch, -1)
        errs = {}
        for j, tag in enumerate(("Fx", "Fy")):
            for q, ch in enumerate(CHANNELS):
                pr, tr = f[tag][ch].ravel(), true[j, q]
                errs[f"{tag}_{ch}"] = np.linalg.norm(pr - tr) / (np.linalg.norm(tr) + 1e-30)
        a_pr = sur.arcs(p)
        C_pr = sur.compliance(p)
        row = {f"p{k}": v for k, v in enumerate(p)}
        for j, tag in enumerate(("Fx", "Fy")):
            pk_t = np.abs(arcs[:, j]).max()
            pk_p = np.abs(a_pr[:, j]).max()
            row[f"peak_{tag}_err_%"] = 100 * (pk_p / pk_t - 1)
        row["defl_err_%"] = 100 * (C_pr[1, 1] / C[1, 1] - 1)
        row["Cxx_err_%"] = 100 * (C_pr[0, 0] / C[0, 0] - 1)
        row["field_stress_relL2_%"] = 100 * np.mean([errs[f"{t}_{c}"] for t in ("Fx", "Fy") for c in ("sx", "sy", "txy")])
        row["field_disp_relL2_%"] = 100 * np.mean([errs[f"{t}_{c}"] for t in ("Fx", "Fy") for c in ("ux", "uy")])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(FAMILIES)
    out = []
    for key in keys:
        fam = FAMILIES[key]
        ntr, nte = SIZES[key]
        print(f"== {fam.name}: {ntr} training + {nte} held-out FE runs")
        cache = f"surrogate_data_{key}.npz"
        try:
            data = dict(np.load(cache))
            data["n_train"] = int(data["n_train"])
            print("   using cached FE snapshots", cache)
        except FileNotFoundError:
            data = generate(fam, ntr, nte, seed=7)
            np.savez(cache, **data)
        sur = train(fam, data)
        print(f"   POD modes kept: {sur.n_modes}")
        sur.save(f"surrogate_{key}.npz")
        df = evaluate(Surrogate.load(f"surrogate_{key}.npz"), data)
        df.insert(0, "family", key)
        out.append(df)
        print(df.drop(columns="family").abs().describe().loc[["mean", "50%", "max"]].round(2).to_string())
    pd.concat(out).to_csv("surrogate_validation.csv", index=False)
