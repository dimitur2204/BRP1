#!/usr/bin/env python3
"""R3 -- patient QC exclusions, feature transform + scanner harmonization,
kernel-leakage probe, redundancy pruning, univariate Cox association.

Steps (all fitted on the TRAIN split only, then applied to val/test, so the
held-out test set never informs any preprocessing):
  1. Exclusions (thresholds from the R2 outlier review, see PROGRESS.md):
     no heart mask; heart < MIN_HEART_ML (non-covering/garbage scans);
     >= METAL_MIN_VOX metal voxels; heart cut off by the scan z-extent;
     unknown kernel group.
  2. Yeo-Johnson + standardize every feature (radiomics are heavily skewed,
     e.g. calcium volume).
  3. ComBat (neuroCombat) with batch = manufacturer x kernel_group,
     preserving age / sex / smoking. Batch-only adjustment is applied to
     val/test with the train estimates (neuroCombatFromTraining).
  4. Leakage probe: L2-logistic on train, AUC on val+test for predicting
     kernel group (sharp vs soft) and manufacturer (macro one-vs-rest), per
     feature family, before vs after ComBat. Sanity check that biology
     survives: heart volume still predicts sex after ComBat.
  5. Redundancy: greedy |Spearman| > CORR_MAX pruning on harmonized train,
     keeping features in a priority order (cardiac > shape > first-order >
     texture), so the interpretable feature wins a tie.
  6. Univariate Cox per kept feature on train+val (discovery), stratified by
     split (the mentor's enrichment gives train 33% vs val ~8% events, so the
     baseline hazards differ), HR per SD:
       unadj     feature
       adj       feature + age + sex + smoking
       adj_tech  adj + log core noise + kernel group (residual-scanner check)
     BH-FDR within each model. Features with adj FDR < 0.05 are then
     replicated on the untouched test split (same adj model).
Outputs: data/radiomics/r3_* tables, data/radiomics/heart_features_v1_r3.csv
(harmonized features + covariates, what R4 trains on), figs/r3_feature_qc/.

Usage:
    env/.venv/bin/python src/r3_feature_qc.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from neuroCombat import neuroCombat, neuroCombatFromTraining
from scipy.stats import false_discovery_control, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PowerTransformer


warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT = Path(__file__).resolve().parents[1]
COHORT_CSV = PROJECT / "data" / "pid_lists" / "heart_cohort_v1.csv"
FEATURES_CSV = PROJECT / "data" / "radiomics" / "heart_features_v1.csv"
OUT_DATA = PROJECT / "data" / "radiomics"
OUT_FIG = PROJECT / "figs" / "r3_feature_qc"

MIN_HEART_ML = 250.0
METAL_MIN_VOX = 5
CORR_MAX = 0.95
FDR = 0.05
CLIN = ["age", "male", "cigsmok"]

# technical, not predictors: measured noise is a scanner covariate; heart
# volume duplicates pyradiomics shape volume; calc_any is binary (not ComBat-able)
NON_PREDICTORS = {"card_core_noise_sd_hu", "card_calc_any"}
FAMILY_ORDER = ["card", "shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]

# dataviz reference palette (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ab"


def family(f: str) -> str:
    return "card" if f.startswith("card_") else f.split("_")[1]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ── 1. load + exclusions ─────────────────────────────────────────────────────
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    feats = pd.read_csv(FEATURES_CSV, dtype={"pid": str})
    df = cohort.merge(feats, on="pid", how="left")
    df["male"] = (df["sex"] == "M").astype(int)
    df["time_yr"] = df["time"] / 365.25

    steps = [
        ("R0 cohort", np.ones(len(df), bool)),
        ("heart mask extracted", df["status"] == "ok"),
        (f"heart volume >= {MIN_HEART_ML:.0f} mL", df["card_heart_volume_ml"] >= MIN_HEART_ML),
        (f"metal voxels < {METAL_MIN_VOX}", df["qc_metal_voxels"] < METAL_MIN_VOX),
        ("heart not cut off at scan edge", df["qc_heart_touches_z_edge"] == 0),
        ("known kernel group", df["kernel_group"] != "other"),
    ]
    keep = np.ones(len(df), bool)
    rows = []
    for name, cond in steps:
        keep &= np.asarray(cond.fillna(False) if hasattr(cond, "fillna") else cond)
        k = df[keep]
        row = {"step": name, "n": len(k), "events": int(k["event"].sum())}
        for s in ("train", "val", "test"):
            row[f"{s}_n"] = int((k["split"] == s).sum())
            row[f"{s}_events"] = int(k.loc[k["split"] == s, "event"].sum())
        rows.append(row)
    return df[keep].reset_index(drop=True), pd.DataFrame(rows)


# ── 2-3. transform + ComBat ──────────────────────────────────────────────────
def transform_and_harmonize(df: pd.DataFrame, feat: list[str]):
    tr = df["split"] == "train"
    pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(df.loc[tr, feat])
    X = pd.DataFrame(pt.transform(df[feat]), columns=feat, index=df.index)

    batch = df["manufacturer"].str.split().str[0] + "_" + df["kernel_group"]
    covars = df.loc[tr, ["age", "male", "cigsmok"]].assign(batch=batch[tr].values).reset_index(drop=True)
    fit = neuroCombat(dat=X.loc[tr].T.values, covars=covars, batch_col="batch",
                      categorical_cols=["male", "cigsmok"], continuous_cols=["age"])
    H = X.copy()
    H.loc[tr] = fit["data"].T
    te = ~tr
    H.loc[te] = neuroCombatFromTraining(dat=X.loc[te].T.values, batch=batch[te].values,
                                        estimates=fit["estimates"])["data"].T
    # re-standardize on train so HRs stay "per SD"
    mu, sd = H.loc[tr].mean(), H.loc[tr].std()
    return X, (H - mu) / sd, batch


# ── 4. leakage probe ─────────────────────────────────────────────────────────
def leakage(df, X, H, feat) -> pd.DataFrame:
    tr, ev = df["split"] == "train", df["split"] != "train"
    fams = {"all": feat, **{f: [c for c in feat if family(c) == f] for f in
                            ["card", "shape", "firstorder"]},
            "texture": [c for c in feat if family(c) in {"glcm", "glrlm", "glszm", "gldm", "ngtdm"}]}
    targets = {"kernel (sharp vs soft)": (df["kernel_group"] == "sharp").astype(int),
               "manufacturer": df["manufacturer"]}
    rows = []
    for tname, y in targets.items():
        for fname, cols in fams.items():
            for stage, M in (("before ComBat", X), ("after ComBat", H)):
                clf = LogisticRegression(C=0.1, max_iter=2000).fit(M.loc[tr, cols], y[tr])
                p = clf.predict_proba(M.loc[ev, cols])
                auc = (roc_auc_score(y[ev], p[:, 1]) if p.shape[1] == 2
                       else roc_auc_score(y[ev], p, multi_class="ovr", average="macro"))
                rows.append({"target": tname, "features": fname, "stage": stage,
                             "n_features": len(cols), "auc_val_test": auc})
    # biology must survive harmonization: heart volume -> sex
    for stage, M in (("before ComBat", X), ("after ComBat", H)):
        rows.append({"target": "sex (biology check)", "features": "card_heart_volume_ml",
                     "stage": stage, "n_features": 1,
                     "auc_val_test": roc_auc_score(df.loc[ev, "male"], M.loc[ev, "card_heart_volume_ml"])})
    return pd.DataFrame(rows)


# ── 5. redundancy ────────────────────────────────────────────────────────────
def prune(H_train: pd.DataFrame, feat: list[str]) -> tuple[list[str], dict]:
    order = sorted(feat, key=lambda f: (FAMILY_ORDER.index(family(f)), f))
    rho = pd.DataFrame(np.abs(spearmanr(H_train[order]).correlation), index=order, columns=order)
    kept, dropped_for = [], {}
    for f in order:
        hit = [k for k in kept if rho.loc[f, k] > CORR_MAX]
        if hit:
            dropped_for[f] = hit[0]
        else:
            kept.append(f)
    return kept, dropped_for


# ── 6. univariate Cox ────────────────────────────────────────────────────────
def cox_hr(d: pd.DataFrame, f: str, extra: list[str], strata) -> dict:
    cols = [f, *extra, "time_yr", "event"] + ([strata] if strata else [])
    cph = CoxPHFitter(penalizer=1e-4).fit(d[cols], "time_yr", "event", strata=[strata] if strata else None)
    s = cph.summary.loc[f]
    return {"hr": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"], "p": s["p"]}


def univariate(df: pd.DataFrame, H: pd.DataFrame, kept: list[str]) -> pd.DataFrame:
    d = pd.concat([df[["split", "time_yr", "event", *CLIN, "kernel_group"]], H[kept]], axis=1)
    d["log_noise"] = np.log(df["card_core_noise_sd_hu"].clip(lower=1))
    d["sharp"] = (d["kernel_group"] == "sharp").astype(int)
    disc, test = d[d["split"] != "test"], d[d["split"] == "test"]
    models = {"unadj": [], "adj": CLIN, "adj_tech": [*CLIN, "log_noise", "sharp"]}

    rows = []
    for f in kept:
        row = {"feature": f, "family": family(f)}
        for m, extra in models.items():
            r = cox_hr(disc, f, extra, "split")
            row.update({f"{m}_{k}": v for k, v in r.items()})
        rows.append(row)
    res = pd.DataFrame(rows)
    for m in models:
        res[f"{m}_fdr"] = false_discovery_control(res[f"{m}_p"], method="bh")

    # replication of adj-FDR hits (plus the top 10 by adj p, for the forest plot)
    rep = set(res.loc[res["adj_fdr"] < FDR, "feature"]) | set(res.nsmallest(10, "adj_p")["feature"])
    for f in rep:
        r = cox_hr(test, f, CLIN, None)
        for k, v in r.items():
            res.loc[res["feature"] == f, f"test_adj_{k}"] = v
    # replication threshold: Bonferroni over the features taken to the test set
    res["replicated"] = res["test_adj_p"] < 0.05 / len(rep)
    return res.sort_values("adj_p").reset_index(drop=True), d


def clinical_reference(d: pd.DataFrame) -> pd.DataFrame:
    disc = d[d["split"] != "test"]
    cph = CoxPHFitter().fit(disc[[*CLIN, "time_yr", "event", "split"]], "time_yr", "event", strata=["split"])
    s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["hr", "lo", "hi", "p"]
    return s


# ── figures ──────────────────────────────────────────────────────────────────
def short(f: str) -> str:
    return f.replace("original_", "").replace("card_", "cardiac_")


def fig_leakage(lk: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    for ax, tgt in zip(axes, ["kernel (sharp vs soft)", "manufacturer"]):
        style(ax)
        t = lk[lk.target == tgt]
        fams = t["features"].unique()
        y = np.arange(len(fams))
        for i, (stage, col) in enumerate((("before ComBat", GRAY), ("after ComBat", BLUE))):
            v = t[t.stage == stage].set_index("features").loc[fams, "auc_val_test"]
            ax.barh(y + (i - 0.5) * 0.36, v, height=0.34, color=col, label=stage, zorder=3)
            for yy, vv in zip(y + (i - 0.5) * 0.36, v):
                ax.text(vv + 0.005, yy, f"{vv:.2f}", va="center", fontsize=7, color=INK2)
        ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(y, fams)
        ax.set_xlim(0.4, 1.05)
        ax.set_xlabel("AUC on val+test (0.5 = features carry no scanner information)", fontsize=8, color=INK2)
        ax.set_title(f"Can features predict {tgt}?", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.55, 1.0), ncol=2)
    bio = lk[lk.target == "sex (biology check)"].set_index("stage")["auc_val_test"]
    fig.text(0.01, 0.01, f"Biology check: heart volume predicts sex with AUC {bio['before ComBat']:.2f} "
             f"before / {bio['after ComBat']:.2f} after ComBat (should stay high).", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_FIG / "r3_scanner_leakage.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_volcano(res: pd.DataFrame, n_disc: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    style(ax)
    sig = res["adj_fdr"] < FDR
    x, y = np.log(res["adj_hr"]), -np.log10(res["adj_p"])
    ax.scatter(x[~sig], y[~sig], s=22, color=GRAY, zorder=3, label=f"FDR ≥ {FDR}")
    ax.scatter(x[sig], y[sig], s=30, color=BLUE, edgecolor=SURFACE, lw=1, zorder=4, label=f"FDR < {FDR}")
    rep = res["replicated"].fillna(False).astype(bool)
    ax.scatter(x[rep], y[rep], s=120, facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=5,
               label="replicated in test (Bonferroni)")
    lab = res.head(4).index.union(res.index[rep])
    for i, idx in enumerate(sorted(lab, key=lambda j: -y[j])):
        dy = 8 if i % 2 == 0 else -10
        ax.annotate(short(res.loc[idx, "feature"]), (x[idx], y[idx]), xytext=(-6 if x[idx] > 0 else 6, dy),
                    textcoords="offset points", fontsize=7, color=INK,
                    ha="right" if x[idx] > 0 else "left")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("log hazard ratio per SD (adjusted for age, sex, smoking)", fontsize=9, color=INK2)
    ax.set_ylabel("−log10 p", fontsize=9, color=INK2)
    ax.set_title(f"Heart features vs lung-cancer incidence — discovery (train+val, "
                 f"n={n_disc[0]}, {n_disc[1]} events)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "r3_volcano.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_forest(res: pd.DataFrame, clin: pd.DataFrame, n_test: tuple[int, int]) -> None:
    top = res.dropna(subset=["test_adj_hr"]).head(12)
    labels = [short(f) for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(top) + 1.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(top))[::-1]
    for off, pre, col, lab in ((0.14, "adj", BLUE, "discovery (train+val)"),
                               (-0.14, "test_adj", ORANGE, f"test replication (n={n_test[0]}, {n_test[1]} events)")):
        ax.errorbar(top[f"{pre}_hr"], y + off,
                    xerr=[top[f"{pre}_hr"] - top[f"{pre}_lo"], top[f"{pre}_hi"] - top[f"{pre}_hr"]],
                    fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.4, capsize=0, label=lab, zorder=3)
    ax.axvline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ticks = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    labels = [l + ("  ★" if r else "") for l, r in zip(labels, top["replicated"].fillna(False))]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("hazard ratio per SD, adjusted for age, sex, smoking (95% CI)", fontsize=9, color=INK2)
    ax.set_title("Top heart features by discovery p-value  (★ = replicated in test, Bonferroni)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2)
    c = clin
    fig.text(0.01, 0.01, "Clinical reference HRs (discovery): age/yr {:.2f}, male {:.2f}, current smoker {:.2f}. "
             "FDR across all kept features.".format(c.loc["age", "hr"], c.loc["male", "hr"], c.loc["cigsmok", "hr"]),
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT_FIG / "r3_forest_top.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df, attrition = load()
    print(attrition.to_string(index=False), "\n")

    feat = [c for c in df.columns if c.startswith(("original_", "card_")) and c not in NON_PREDICTORS]
    X, H, batch = transform_and_harmonize(df, feat)
    print("ComBat batches (train n):", batch[df.split == "train"].value_counts().to_dict(), "\n")

    lk = leakage(df, X, H, feat)
    print(lk.pivot_table(index=["target", "features"], columns="stage", values="auc_val_test").round(3), "\n")

    kept, dropped_for = prune(H[df.split == "train"], feat)
    print(f"redundancy pruning |rho|>{CORR_MAX}: {len(feat)} -> {len(kept)} features")

    res, d = univariate(df, H, kept)
    clin = clinical_reference(d)
    n_disc = (int((df.split != "test").sum()), int(df.loc[df.split != "test", "event"].sum()))
    n_test = (int((df.split == "test").sum()), int(df.loc[df.split == "test", "event"].sum()))
    print("clinical reference (discovery, stratified by split):\n", clin.round(3), "\n")
    for m in ("unadj", "adj", "adj_tech"):
        print(f"{m}: {int((res[f'{m}_fdr'] < FDR).sum())}/{len(res)} features FDR<{FDR}")
    cols = ["feature", "adj_hr", "adj_lo", "adj_hi", "adj_p", "adj_fdr", "adj_tech_fdr",
            "test_adj_hr", "test_adj_lo", "test_adj_hi", "test_adj_p"]
    with pd.option_context("display.width", 220, "display.max_colwidth", 48):
        print(res[cols].head(15).round(4).to_string(index=False))

    attrition.to_csv(OUT_DATA / "r3_attrition.csv", index=False)
    lk.to_csv(OUT_DATA / "r3_scanner_leakage.csv", index=False)
    res.to_csv(OUT_DATA / "r3_univariate_cox.csv", index=False)
    clin.to_csv(OUT_DATA / "r3_clinical_reference_cox.csv")
    pd.DataFrame({"feature": kept}).to_csv(OUT_DATA / "r3_selected_features.csv", index=False)
    pd.Series(dropped_for, name="kept_instead").rename_axis("dropped").to_csv(OUT_DATA / "r3_dropped_redundant.csv")
    meta = df[["pid", "split", "time", "time_yr", "event", "age", "sex", "male", "race", "cigsmok",
               "manufacturer", "kernel", "kernel_group", "slice_thickness_mm", "card_core_noise_sd_hu"]]
    pd.concat([meta, H[feat]], axis=1).to_csv(OUT_DATA / "heart_features_v1_r3.csv", index=False)

    fig_leakage(lk)
    fig_volcano(res, n_disc)
    fig_forest(res, clin, n_test)
    print(f"\nwrote data/radiomics/r3_*.csv, heart_features_v1_r3.csv, figs/r3_feature_qc/*.png")


if __name__ == "__main__":
    main()
