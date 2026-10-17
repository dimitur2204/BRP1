#!/usr/bin/env python3
"""S1b -- are the controls with a TotalSegmentator mask a random subset?

Every cancer case has a mask but only ~63% of controls do, so the cohort has
6.3% events instead of NLST's ~4%. Ranking metrics (C, time-dependent AUC)
do not depend on prevalence as long as the dropped controls are like the kept
ones. This script checks that, among the S1 population just before the mask
step (baseline series of study year 0, resampled CT on disk, follow-up > 0
days), split into controls WITH a mask (kept) and WITHOUT one (dropped):

  balance    per covariate: kept vs dropped mean/share and the standardised
             mean difference (SMD; |SMD| < 0.1 is the usual "balanced" rule)
               clinical  age, male, white, current smoker (cigsmok), positive
                         T0 screen (scr_res0 4-6), follow-up years
               scanner   manufacturer, kernel, slice thickness, kVp, mA,
                         diagnostic quality (scan list ctdxqual), image count
               admin     split of the older lcrisk_discovery experiment that the
                         scan list comes from, pid number
  prediction cross-validated AUC (5-fold logistic regression) of "has a mask"
             from each covariate set. ~0.5 = mask availability is unrelated to
             those variables.
  weights    inverse-probability weights for the kept patients, from the
             clinical + scanner model fitted on all controls: w = P(mask) /
             P(mask | x) for controls (mean ~1), trimmed to the 1st-99th
             percentile; events get n_events / n_events_with_mask (~1). Not
             used by S5 yet; they allow a weighted C/AUC that estimates
             performance on all NLST controls with a CT, if the balance table
             shows real differences.

Outputs:
  data/mask_selection_balance.csv, data/mask_selection_auc.csv,
  data/mask_ipw.csv (pid, event, p_mask, w), figs/mask_selection.png (SMD plot)

Usage (login node, CPU, a few minutes; checks file existence by pid name only):
    env/.venv/bin/python src/s1b_mask_selection.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import s1_candidates
from nlst import CT_DIR, DATA, FIGS, PRSN_CSV, SCAN_LIST, SEG_DIR

BALANCE_CSV = DATA / "mask_selection_balance.csv"
AUC_CSV = DATA / "mask_selection_auc.csv"
IPW_CSV = DATA / "mask_ipw.csv"
POSITIVE_SCREEN = (4, 5, 6)  # prsn dictionary: scr_resX = 4, 5 or 6 = positive screen
TOP_KERNELS = 8


def has_files(pid: str) -> tuple[str, int, int]:
    return (pid, int((CT_DIR / f"{pid}_yr0.nii.gz").exists()),
            int((SEG_DIR / f"{pid}_yr0" / "seg.nii.gz").exists()))


def population() -> pd.DataFrame:
    """The S1 population right before the mask step, with covariates."""
    p = pd.read_csv(PRSN_CSV, dtype={"pid": str},
                    usecols=["pid", "age", "gender", "race", "cigsmok", "scr_res0",
                             "candx_days", "canc_free_days", "scr_days0"])
    p["event"] = p["candx_days"].notna().astype(int)
    p["time"] = np.where(p["event"] == 1, p["candx_days"], p["canc_free_days"]) - p["scr_days0"].fillna(0)
    scans = pd.read_csv(SCAN_LIST, dtype={"pid": str},
                        usecols=["pid", "series_uid", "ctdxqual", "image_count", "split"])
    df = p.merge(scans.rename(columns={"split": "old_split"}), on="pid")
    df = df.merge(s1_candidates.load_series_meta(), on="series_uid", how="left")
    df = df[df["series_study_yr"] == 0]
    print(f"checking CT/mask files for {len(df)} pids ...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        files = pd.DataFrame(list(ex.map(has_files, df["pid"])), columns=["pid", "has_ct", "has_seg"])
    df = df.merge(files, on="pid")
    df = df[(df["has_ct"] == 1) & (df["time"] > 0)].copy()

    df["male"] = (df["gender"] == 1).astype(float)
    df["white"] = (df["race"] == 1).astype(float)
    df["screen_positive_t0"] = df["scr_res0"].isin(POSITIVE_SCREEN).astype(float)
    df["followup_years"] = df["time"] / 365.25
    df["diag_quality_1"] = (df["ctdxqual"] == 1).astype(float)
    df["pid_num"] = pd.to_numeric(df["pid"], errors="coerce")
    top = df["kernel"].value_counts().index[:TOP_KERNELS]
    df["kernel_grp"] = df["kernel"].where(df["kernel"].isin(top), "other").fillna("other")
    df["manufacturer"] = df["manufacturer"].fillna("unknown")
    return df


NUMERIC = {"clinical": ["age", "followup_years"],
           "scanner": ["slice_thickness_mm", "kvp", "tube_current_ma", "image_count"],
           "admin": ["pid_num"]}
BINARY = {"clinical": ["male", "white", "cigsmok", "screen_positive_t0"],
          "scanner": ["diag_quality_1"]}
CATEGORICAL = {"scanner": ["manufacturer", "kernel_grp"], "admin": ["old_split"]}


def smd_num(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def smd_prop(p1: float, p0: float) -> float:
    s = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return float((p1 - p0) / s) if s > 0 else 0.0


def balance(ct: pd.DataFrame) -> pd.DataFrame:
    k, d = ct[ct.has_seg == 1], ct[ct.has_seg == 0]
    rows = []
    for group, cols in NUMERIC.items():
        for c in cols:
            rows.append({"group": group, "covariate": c, "kept": k[c].mean(), "dropped": d[c].mean(),
                         "missing_kept": k[c].isna().mean(), "missing_dropped": d[c].isna().mean(),
                         "smd": smd_num(k[c].dropna(), d[c].dropna())})
    for group, cols in BINARY.items():
        for c in cols:
            p1, p0 = k[c].mean(), d[c].mean()
            rows.append({"group": group, "covariate": c, "kept": p1, "dropped": p0,
                         "missing_kept": k[c].isna().mean(), "missing_dropped": d[c].isna().mean(),
                         "smd": smd_prop(p1, p0)})
    for group, cols in CATEGORICAL.items():
        for c in cols:
            for lvl in ct[c].value_counts().index:
                p1, p0 = (k[c] == lvl).mean(), (d[c] == lvl).mean()
                rows.append({"group": group, "covariate": f"{c}={lvl}", "kept": p1, "dropped": p0,
                             "missing_kept": 0.0, "missing_dropped": 0.0, "smd": smd_prop(p1, p0)})
    out = pd.DataFrame(rows)
    out["imbalanced"] = out["smd"].abs() >= 0.1
    return out


def design(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    cols = [c for g in groups for c in NUMERIC.get(g, []) + BINARY.get(g, [])]
    x = df[cols].astype(float)
    cats = [c for g in groups for c in CATEGORICAL.get(g, [])]
    if cats:
        x = pd.concat([x, pd.get_dummies(df[cats].astype(str), drop_first=True, dtype=float)], axis=1)
    return x


def model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def prediction_auc(ct: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    y = ct["has_seg"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    # follow-up years is left out of the prediction sets: it is part of the outcome
    sets = {"clinical": ["clinical"], "scanner": ["scanner"], "clinical+scanner": ["clinical", "scanner"],
            "admin (old split, pid)": ["admin"]}
    for name, groups in sets.items():
        x = design(ct, groups).drop(columns=["followup_years"], errors="ignore")
        p = cross_val_predict(model(), x, y, cv=cv, method="predict_proba")[:, 1]
        rows.append({"covariates": name, "n_features": x.shape[1], "cv_auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def weights(df: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    x_ct = design(ct, ["clinical", "scanner"]).drop(columns=["followup_years"])
    m = model().fit(x_ct, ct["has_seg"])
    kept = df[df.has_seg == 1].copy()
    x_k = design(kept, ["clinical", "scanner"]).drop(columns=["followup_years"])
    x_k = x_k.reindex(columns=x_ct.columns, fill_value=0.0)
    kept["p_mask"] = m.predict_proba(x_k)[:, 1]
    is_ct = kept.event == 0
    w = ct["has_seg"].mean() / kept.loc[is_ct, "p_mask"]
    lo, hi = np.percentile(w, [1, 99])
    kept.loc[is_ct, "w"] = w.clip(lo, hi)
    ev = df[df.event == 1]
    kept.loc[~is_ct, "p_mask"] = ev["has_seg"].mean()
    kept.loc[~is_ct, "w"] = len(ev) / ev["has_seg"].sum()
    return kept[["pid", "event", "p_mask", "w"]]


def smd_plot(bal: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = bal.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(b) + 1.2))
    colors = {"clinical": "tab:blue", "scanner": "tab:orange", "admin": "tab:grey"}
    ax.scatter(b["smd"], range(len(b)), c=[colors[g] for g in b["group"]], s=18)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b["covariate"], fontsize=7)
    for v in (-0.1, 0.1):
        ax.axvline(v, color="grey", ls="--", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SMD, controls with mask (kept) - without (dropped)")
    ax.set_title("Are the controls with a TotalSegmentator mask a random subset?\n"
                 "blue clinical, orange scanner, grey administrative; |SMD| < 0.1 = balanced", fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "mask_selection.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = population()
    ct = df[df.event == 0]
    ev = df[df.event == 1]
    print(f"population: {len(df)} pts, {len(ev)} events ({ev.has_seg.sum()} with mask), "
          f"{len(ct)} controls ({ct.has_seg.sum()} with mask = {ct.has_seg.mean():.1%}); "
          f"event rate {df.event.mean():.2%} -> {df[df.has_seg == 1].event.mean():.2%} with masks")

    bal = balance(ct)
    pd.set_option("display.width", 200)
    print("\nbalance, controls kept vs dropped:")
    print(bal.round(3).to_string(index=False))
    auc = prediction_auc(ct)
    print("\ncan the covariates predict who has a mask? (5-fold CV AUC, 0.5 = no)")
    print(auc.round(3).to_string(index=False))

    w = weights(df, ct)
    wc = w.loc[w.event == 0, "w"]
    ess = wc.sum() ** 2 / (wc ** 2).sum()
    print(f"\nIPW (controls): mean {wc.mean():.3f}, range {wc.min():.2f}-{wc.max():.2f}, "
          f"effective n {ess:.0f} of {len(wc)}")

    bal.to_csv(BALANCE_CSV, index=False)
    auc.to_csv(AUC_CSV, index=False)
    w.to_csv(IPW_CSV, index=False)
    smd_plot(bal)
    print(f"\nwrote {BALANCE_CSV}, {AUC_CSV}, {IPW_CSV}, {FIGS / 'mask_selection.png'}")


if __name__ == "__main__":
    main()
