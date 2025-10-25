# -*- coding: utf-8 -*-
import os, sys, json, numpy as np, pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def balanced_group_kfold(groups, n_splits=5, seed=1337):
    rng = np.random.RandomState(seed)
    fam_counts = Counter(groups)
    fams = list(fam_counts.keys())
    rng.shuffle(fams)
    fams.sort(key=lambda f: fam_counts[f], reverse=True)
    folds = [set() for _ in range(n_splits)]
    load = [0]*n_splits
    for f in fams:
        i = min(range(n_splits), key=lambda k: load[k])
        folds[i].add(f); load[i]+=fam_counts[f]
    fam_to_fold = { }
    for k, fs in enumerate(folds):
        for f in fs: fam_to_fold[f] = k
    return np.array([fam_to_fold[g] for g in groups], dtype=int)

def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Clean categories
    for c in ["family","method","context_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("NA").astype(str).str.strip()
    df["family"] = df["family"].replace({"": "Other"}).fillna("Other")

    # Numerics
    for col in ["excitation_nm","emission_nm","stokes_shift_nm","contrast_normalized"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    med = df[["excitation_nm","emission_nm","stokes_shift_nm"]].median()
    df[["excitation_nm","emission_nm","stokes_shift_nm"]] = df[["excitation_nm","emission_nm","stokes_shift_nm"]].fillna(med)

    # Target/log
    y_log = np.log1p(df["contrast_normalized"].values.astype(float))
    groups = df["family"].values

    num_cols = ["excitation_nm","emission_nm","stokes_shift_nm"]
    cat_cols = ["method","context_type","family"]
    X = df[num_cols + cat_cols]
    sw = df["sample_weight"].values if "sample_weight" in df.columns else None
    if sw is not None:
        sw = np.nan_to_num(sw, nan=1.0)

    pre = ColumnTransformer([
        ("num","passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2), cat_cols)
    ])

    rf = RandomForestRegressor(n_estimators=1200, min_samples_leaf=2, n_jobs=-1, random_state=1337)
    pipe = Pipeline([("prep", pre), ("rf", rf)])

    fold_idx = balanced_group_kfold(groups, n_splits=5, seed=1337)
    y_pred_log = np.zeros_like(y_log, dtype=float)

    for k in range(5):
        tr = fold_idx != k
        te = fold_idx == k
        pipe.fit(X.iloc[tr], y_log[tr], rf__sample_weight=(sw[tr] if sw is not None else None))
        y_pred_log[te] = pipe.predict(X.iloc[te])

    # Metrics (original scale)
    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)
    r2  = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mae_mean = float(mean_absolute_error(y_true, np.full_like(y_true, y_true.mean())))
    mae_med  = float(mean_absolute_error(y_true, np.full_like(y_true, np.median(y_true))))
    delta_mae = float((mae_mean - mae) / max(mae_mean, 1e-9) * 100.0)

    # Simple split-conformal (global 90%)
    resid = np.abs(y_true - y_pred)
    q = float(np.quantile(resid, 0.90))
    pi_low  = y_pred - q
    pi_high = y_pred + q
    covered = ((y_true >= pi_low) & (y_true <= pi_high)).mean()
    coverage = float(covered*100.0)
    ece = float(abs(covered - 0.90))

    # Save
    metrics = {
        "r2": r2, "mae": mae,
        "baseline_mae_mean": mae_mean, "baseline_mae_median": mae_med,
        "delta_mae_percent": delta_mae,
        "coverage_90_percent": coverage, "ece_abs_error": ece
    }
    with open(os.path.join(out_dir, "cv_metrics_v2_2_2.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        "fold": fold_idx, "family": df["family"],
        "y_true": y_true, "y_pred": y_pred,
        "pi_low": pi_low, "pi_high": pi_high
    }).to_csv(os.path.join(out_dir, "cv_predictions_uq_v2_2_2.csv"), index=False, encoding="utf-8")

if __name__ == "__main__":
    # PowerShell-friendly: python script.py --data "<path>" --out "<dir>"
    args = sys.argv
    data_arg = args[args.index("--data")+1] if "--data" in args else None
    out_arg  = args[args.index("--out")+1]  if "--out"  in args else ".\\outputs_v2_2_2"
    if not data_arg or not os.path.exists(data_arg):
        print("NO-RUN: data file not found"); sys.exit(1)
    main(data_arg, out_arg)
