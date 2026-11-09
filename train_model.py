"""
train_model.py - train and evaluate the singular-stress -> real-peak-stress model.

Compared methods (all evaluated on geometries never used for training):
  1. NAIVE    : report the FE stress at the corner node (what an engineer reads off the contour)
  2. PHYSICS  : sigma_peak = C(beta) * |K1| * rho^(lam-1)   (Williams amplitude + notch-radius scaling,
                only C(beta) fitted - 3 coefficients)
  3. ML       : gradient-boosted trees on dimensionless, mesh-independent features
Target for (2)/(3): log(sigma_peak / |s_ref|)  -> dimensionless, so one model covers any size and load.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

from corner_tools import PROFILE_R

FEATURES = (["lam", "beta", "log_rho_over_ell", "K1_hat", "T_hat"]
            + [f"{p}_{r}" for p in ("ptt", "prt", "pvm") for r in PROFILE_R])


def prep(df):
    df = df.copy()
    df["log_rho_over_ell"] = np.log(df["rho_over_ell"])
    df["y"] = np.log(df["target_peak"] / df["s_ref"].abs())
    return df


def physics_design(df):
    """log(peak) - log|K1| - (lam-1) log(rho) = c0 + c1*beta + c2*beta^2"""
    b = df["beta"].to_numpy() / 300
    X = np.stack([b, b ** 2], 1)
    off = np.log(df["K1"].abs()) + (df["lam"] - 1) * np.log(df["rho"])
    return X, off


def pct_err(pred, true):
    return 100 * (pred - true) / true


def summary(name, e):
    a = np.abs(e)
    return {"method": name, "mean |err| %": a.mean(), "median |err| %": np.median(a),
            "95th pct |err| %": np.percentile(a, 95), "max |err| %": a.max()}


def main():
    parts = []
    for f in ("dataset.csv", "dataset_notch.csv", "dataset_bracket2.csv"):
        try:
            d = pd.read_csv(f)
        except FileNotFoundError:
            continue
        if "family" not in d:
            d["family"] = "bracket"
        parts.append(d)
    df = prep(pd.concat(parts, ignore_index=True))
    df = df[np.isfinite(df[FEATURES + ["y"]]).all(axis=1)]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(df, groups=df["seed"]))
    dtr, dte = df.iloc[tr], df.iloc[te]
    print(f"rows: train {len(dtr)}  test {len(dte)}  (geometries: {dtr.seed.nunique()} / {dte.seed.nunique()})")

    # physics baseline
    Xp, off = physics_design(dtr)
    phys = LinearRegression().fit(Xp, np.log(dtr["target_peak"]) - off)
    Xpt, offt = physics_design(dte)
    pred_phys = np.exp(phys.predict(Xpt) + offt)

    # ML
    est = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=20, l2_regularization=1e-3, random_state=0)
    est.fit(dtr[FEATURES], dtr["y"])
    pred_ml = np.exp(est.predict(dte[FEATURES])) * dte["s_ref"].abs()

    true = dte["target_peak"].to_numpy()
    res = pd.DataFrame([
        summary("Naive: FE corner-node stress", pct_err(dte["vm_corner_node"].to_numpy(), true)),
        summary("Physics formula (Williams K1, 3 constants)", pct_err(pred_phys, true)),
        summary("ML model (gradient boosting)", pct_err(pred_ml, true)),
    ])
    print(res.round(2).to_string(index=False))

    for fam, g in dte.assign(e=pct_err(pred_ml, true)).groupby("family"):
        print(f"  ML on {fam}: mean |err| {g.e.abs().mean():.2f}%  95th {g.e.abs().quantile(.95):.2f}%")
    out = dte[["seed", "family", "beta", "h_over_ell", "rho_over_ell", "vm_corner_node", "target_peak"]].copy()
    out["pred_phys"], out["pred_ml"] = pred_phys, pred_ml
    out.to_csv("test_predictions.csv", index=False)
    by_mesh = out.assign(err_ml=np.abs(pct_err(out.pred_ml, out.target_peak)),
                         err_naive=np.abs(pct_err(out.vm_corner_node, out.target_peak))) \
        .groupby(out.h_over_ell.round(4))[["err_naive", "err_ml"]].mean()
    print("\nmean |error| % by mesh size at the corner (h / l):\n", by_mesh.round(2))
    res.to_csv("results_summary.csv", index=False)

    # final model on all data for the tool
    est_all = HistGradientBoostingRegressor(**est.get_params()).fit(df[FEATURES], df["y"])
    joblib.dump({"est": est_all, "cols": FEATURES, "phys": phys}, "singularity_model.joblib")
    print("saved singularity_model.joblib")


if __name__ == "__main__":
    main()
