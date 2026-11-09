"""
build_conformal.py - calibrated 90 % prediction intervals for EigenStress (split conformal, Mondrian by bin).

Calibration data: test_predictions.csv = predictions of the model trained WITHOUT these geometries.
Score = |ln(predicted / true peak)|. Bins = corner kind (outline / cutout) x rho/l band, because the error
grows with the fillet size and is larger at cutout corners. For each bin the interval half-width is the
finite-sample conformal quantile  q = score_(ceil((n+1)(1-alpha)))  ->  interval = [pred e^-q, pred e^+q].
Real parts differ from the training families (neighbouring corners, other solvers), so the test residuals alone
under-cover them (75 % on 53 independent cases). A FLOOR is therefore calibrated on the independent validation
cases (ANSYS, Peterson, unseen parts, hole loads, cutout corners; 68 cases): final half-width = max(bin q, floor q),
checked by leave-one-out on those cases.
Writes conformal.json (used by the app) and prints both coverage checks.
"""
import json
import numpy as np
import pandas as pd

ALPHA = 0.10
BANDS = [0.0, 0.1, 0.3, 0.6, 1.0001]
LABELS = ["<=0.1", "0.1-0.3", "0.3-0.6", ">0.6"]


def prep(d):
    d = d.copy()
    d["score"] = np.abs(np.log(d.pred_ml / d.target_peak))
    d["kind"] = np.where(d.family == "window", "cutout", "outline")
    d["band"] = pd.cut(d.rho_over_ell, BANDS, labels=LABELS).astype(str)
    return d


def cq(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1])


def build(d):
    return {f"{k}|{b}": {"q": cq(g.score), "n": int(len(g))} for (k, b), g in d.groupby(["kind", "band"])}


def external_cases():
    """Independent validation cases -> DataFrame(set, score, rl, kind)."""
    import warnings
    warnings.filterwarnings("ignore")
    from ansys_validate import PARTS, prep_part
    from corner_tools import corner_length
    ell = {}
    for _, s_ in PARTS.items():
        if s_ is None:
            continue
        p_ = prep_part(s_)
        ell[s_["name"]] = {i: corner_length(p_["poly"], i) for i in range(len(p_["poly"]))}
    rows = []
    a = pd.read_csv("ansys_validation_results.csv")
    rows += [("ANSYS 2025 R1", r.EigenStress_pred, r.fillet_peak_truth, r.radius_mm / ell[r.part][int(r.corner[1:])], "outline")
             for r in a.itertuples()]
    c = pd.read_csv("validation_cutout_corners.csv")
    rows += [("cutout corners", r["EigenStress"], r["truth fillet"], r["rho"] / r["ell"], "cutout") for _, r in c.iterrows()]
    u = pd.read_csv("unseen_parts_results.csv")
    rows += [("unseen parts", r.model_MPa, r.true_peak_MPa, r.radius_mm / ell[r.part][int(r.corner)], "outline")
             for r in u.itertuples()]
    pt = pd.read_csv("peterson_table.csv")
    rows += [("Peterson", r["Kt EigenStress (from sharp model)"], r["Kt Peterson"], 1 / r["h/r"], "outline")
             for _, r in pt.iterrows()]
    h = pd.read_csv("validation_hole_loads.csv")
    rows += [("hole loads", r.model, r["true peak"], 0.05, "outline") for _, r in h.iterrows()]
    e = pd.DataFrame(rows, columns=["set", "pred", "true", "rl", "kind"])
    e["score"] = np.abs(np.log(e.pred / e.true))
    return e


if __name__ == "__main__":
    d = prep(pd.read_csv("test_predictions.csv"))
    # coverage check: calibrate on half the geometries, test on the other half
    rng = np.random.default_rng(0)
    seeds = d.seed.unique()
    a = set(rng.choice(seeds, len(seeds) // 2, replace=False))
    tab = build(d[d.seed.isin(a)])
    ev = d[~d.seed.isin(a)]
    hit = ev.apply(lambda r: r.score <= tab[f"{r.kind}|{r.band}"]["q"], axis=1)
    print(f"held-out coverage of the 90 % interval: {100 * hit.mean():.1f} % ({len(ev)} cases)")
    full = build(d)
    e = external_cases()
    floor = cq(e.score)
    lab = lambda rl: LABELS[int(np.clip(np.searchsorted(BANDS, rl, side="left") - 1, 0, 3))]
    loo = [e.score.iat[i] <= max(full[f"{e.kind.iat[i]}|{lab(e.rl.iat[i])}"]["q"], cq(e.score.drop(e.index[i])))
           for i in range(len(e))]
    e["inside"] = loo
    print(f"independent cases (leave-one-out, with floor ±{100 * (np.exp(floor) - 1):.1f} %): "
          f"{100 * np.mean(loo):.0f} % inside ({len(e)} cases)")
    print(e.groupby("set").inside.agg(["sum", "count"]).to_string())
    json.dump({"alpha": ALPHA, "bands": BANDS, "labels": LABELS, "table": full, "floor": floor,
               "floor_n": int(len(e))}, open("conformal.json", "w"), indent=1)
    for k, v in full.items():
        print(f"  {k:18s} ±{100 * (np.exp(v['q']) - 1):5.1f} %  (n={v['n']})")
