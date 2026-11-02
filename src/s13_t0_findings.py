#!/usr/bin/env python3
"""S13 -- does a run's eta re-detect the nodules a radiologist saw on the T0 scan?

Why: `clinical_cox/joint_s2` is the best run so far (test dC +0.049) and its gain
is concentrated in the first year (dAUC 1 y +0.103 vs +0.02-0.04 for s0/s1). 89-91%
of cancers diagnosed within 1 y already had a nodule >= 4 mm on the T0 CT (median
15-18 mm = 6-7 voxels of 2.5 mm), so the 2.5 mm input is fine enough to see them.
The project goal is general lung features, not nodule detection
(PROGRESS 2026-10-26), so we need to know which one a run learned.

T0 findings (public IDC-780 `ctab`, study_yr 0, one row per abnormality):
  any_nodule  a non-calcified nodule/mass >= 4 mm (sct_ab_desc 51)
  max_dia     its longest diameter [mm], largest nodule; 0 if none
  dia_bin     none / 4-6 / 6-10 / 10-20 / >20 mm
  nodule_lobe lobe of the largest nodule (TotalSegmentator label; lingula -> LUL)
  spiculated  any nodule with spiculated margins
  emphysema, fibrosis, consolidation, adenopathy, atelectasis, many_nodules
              (59, 61, 58, 56, 54, 62): reported abnormalities, 0/1
Written to data/t0_findings.csv (labels only, all cohort patients).

Per run and split (train/val only; test is refused, s5 is the only test reader):
  by_nodule_size.csv  mean eta (in val-SD units, centred on val controls without a
                      nodule) per dia_bin, for events and controls separately. A
                      nodule detector scores controls with large BENIGN nodules
                      high too; a general lung score does not.
  associations.csv    Spearman rho of eta with max_dia and the other findings;
                      AUC of eta for "nodule >= 10 mm" among controls
  strata_c.csv        Harrell C of eta in strata: all; no T0 nodule >= 4 mm;
                      events within 1 y removed; no nodule and no event within 1 y
  added_value.csv     5-fold CV on the split (Cox fitted on 4/5, scored on 1/5,
                      out-of-fold scores pooled): C of clinical, clinical + T0
                      nodules (any_nodule + log1p(max_dia)), and each + eta, with a
                      paired bootstrap CI of each dC (no refit inside the
                      bootstrap). Plus the full-split Cox HR of eta per SD and the
                      LR p for eta on top of clinical + T0 nodules.
  summary.csv         one row per run x split with the key numbers
  by_nodule_size.png  the first table as a figure

Usage (login node, CPU, ~2-5 min):
    env/.venv/bin/python src/s13_t0_findings.py
    env/.venv/bin/python src/s13_t0_findings.py --runs cosine,clinical_cox/joint_s2 --splits val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import harrell_c
from nlst import COHORT_CSV, DATA, PRSN_CSV, RUNS_DIR

CTAB_CSV = PRSN_CSV.parent / "nlst_780_ctab_idc_20210527.csv"
T0_CSV = DATA / "t0_findings.csv"
RUNS = "cosine,clinical_cox/joint,clinical_cox/joint_s1,clinical_cox/joint_s2,clinical_cox/offset"
CLINICAL = ["age", "male", "cigsmok", "lung_ml", "lung_mean_hu"]
T0_NODULE = ["any_nodule", "log_max_dia"]
DIA_BINS = [-1, 0.5, 6, 10, 20, np.inf]
DIA_LABELS = ["none", "4-6", "6-10", "10-20", ">20"]
# ctab sct_epi_loc -> TotalSegmentator lobe label (10 LUL, 11 LLL, 12 RUL, 13 RML, 14 RLL)
EPI_LOC_TO_LOBE = {1: 12, 2: 13, 3: 14, 4: 10, 5: 10, 6: 11}
OTHER_FINDINGS = {59: "emphysema", 61: "fibrosis", 58: "consolidation", 56: "adenopathy",
                  54: "atelectasis", 62: "many_nodules"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS, help="comma-separated run names under runs/")
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--ctab", default=str(CTAB_CSV))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", default=str(RUNS_DIR / "t0_findings"))
    a = ap.parse_args()
    if "test" in a.splits.split(","):
        ap.error("test predictions are only read by s5_evaluate.py")
    return a


def t0_findings(cohort: pd.DataFrame, ctab_csv: str) -> pd.DataFrame:
    """One row per cohort patient: what the radiologist reported on the T0 CT."""
    ct = pd.read_csv(ctab_csv, dtype={"pid": str})
    ct = ct[ct.study_yr == 0]
    nod = ct[ct.sct_ab_desc == 51].assign(dia=lambda d: pd.to_numeric(d.sct_long_dia, errors="coerce"))
    nod = nod.sort_values("dia", ascending=False, na_position="last")
    g = nod.groupby("pid").agg(n_nodules=("dia", "size"), max_dia=("dia", "max"),
                               epi_loc=("sct_epi_loc", "first"),
                               spiculated=("sct_margins", lambda s: float((pd.to_numeric(s, errors="coerce") == 1).any())))
    out = cohort[["pid"]].merge(g, on="pid", how="left")
    out["any_nodule"] = out.n_nodules.notna().astype(float)
    out["n_nodules"] = out.n_nodules.fillna(0).astype(int)
    out["max_dia"] = out.max_dia.fillna(0.0)  # a nodule without a measured diameter counts as 4 mm below
    out.loc[(out.any_nodule == 1) & (out.max_dia == 0), "max_dia"] = 4.0
    out["dia_bin"] = pd.cut(out.max_dia, DIA_BINS, labels=DIA_LABELS).astype(str)
    out["nodule_lobe"] = pd.to_numeric(out.pop("epi_loc"), errors="coerce").map(EPI_LOC_TO_LOBE)
    out["spiculated"] = out.spiculated.fillna(0.0)
    for code, name in OTHER_FINDINGS.items():
        out[name] = out.pid.isin(ct.loc[ct.sct_ab_desc == code, "pid"]).astype(float)
    return out


def auc(score: np.ndarray, label: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if not n1 or not n0:
        return np.nan
    return float((rankdata(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def design(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = {"male": lambda: (d.sex == "M").astype(float),
                "log_max_dia": lambda: np.log1p(d.max_dia)}.get(c, lambda: d[c].astype(float))()
    return X


def cv_scores(d: pd.DataFrame, models: dict, folds: int, rng) -> dict:
    """Out-of-fold Cox linear predictors (folds stratified by event)."""
    from lifelines import CoxPHFitter
    fold = np.empty(len(d), int)
    for e in (0, 1):
        idx = np.flatnonzero(d.event.to_numpy() == e)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    lp = {name: np.empty(len(d)) for name in models}
    for name, cols in models.items():
        X = design(d, cols).assign(time=d.time.to_numpy(), event=d.event.to_numpy())
        for k in range(folds):
            tr, te = fold != k, fold == k
            cph = CoxPHFitter().fit(X[tr], "time", "event")
            lp[name][te] = cph.predict_log_partial_hazard(X.loc[te, cols]).to_numpy()
    return lp


def added_value(d: pd.DataFrame, a, rng) -> tuple[pd.DataFrame, dict]:
    from lifelines import CoxPHFitter
    from scipy.stats import chi2
    models = {"clinical": CLINICAL, "clinical+eta": CLINICAL + ["eta"],
              "clinical+t0": CLINICAL + T0_NODULE, "clinical+t0+eta": CLINICAL + T0_NODULE + ["eta"]}
    lp = cv_scores(d, models, a.folds, rng)
    t, e = d.time.to_numpy(), d.event.to_numpy()
    rows = [{"model": n, "cv_c": harrell_c(lp[n], t, e)} for n in models]
    boot = {n: [] for n in models}
    for _ in range(a.n_boot):
        i = rng.integers(0, len(d), len(d))
        for n in models:
            boot[n].append(harrell_c(lp[n][i], t[i], e[i]))
    for hi, lo in (("clinical+eta", "clinical"), ("clinical+t0", "clinical"), ("clinical+t0+eta", "clinical+t0")):
        diff = np.asarray(boot[hi]) - np.asarray(boot[lo])
        rows.append({"model": f"delta {hi} - {lo}", "cv_c": harrell_c(lp[hi], t, e) - harrell_c(lp[lo], t, e),
                     "lo": np.nanpercentile(diff, 2.5), "hi": np.nanpercentile(diff, 97.5),
                     "p_boot_le0": float(np.nanmean(diff <= 0))})
    # full-split fit: is eta still associated with risk once the T0 nodules are in the model?
    fits = {}
    for n in ("clinical+t0", "clinical+t0+eta"):
        X = design(d, models[n]).assign(time=t, event=e)
        fits[n] = CoxPHFitter().fit(X, "time", "event")
    s = fits["clinical+t0+eta"].summary.loc["eta"]
    lr = 2 * (fits["clinical+t0+eta"].log_likelihood_ - fits["clinical+t0"].log_likelihood_)
    extra = {"hr_eta_adj_t0": s["exp(coef)"], "hr_lo": s["exp(coef) lower 95%"],
             "hr_hi": s["exp(coef) upper 95%"], "lr_p_eta_over_clinical_t0": float(chi2.sf(lr, 1))}
    return pd.DataFrame(rows), extra


def analyse(pred: pd.DataFrame, t0: pd.DataFrame, split: str, a, rng) -> dict:
    d = pred[pred.split == split].merge(t0, on="pid", how="left").reset_index(drop=True)
    val = pred[pred.split == "val"].merge(t0, on="pid")
    ref = val[(val.event == 0) & (val.any_nodule == 0)]
    d["eta"] = (d.eta - ref.eta.mean()) / val.eta.std()  # 0 = typical nodule-free val control, 1 = 1 val SD
    t, e = d.time.to_numpy(), d.event.to_numpy()
    early = (e == 1) & (t <= 365)

    by_size = (d.assign(group=np.where(e == 1, "events", "controls"))
               .groupby(["group", "dia_bin"]).eta.agg(["size", "mean", "sem"]).reset_index())
    ctrl = d[e == 0]
    assoc = [{"finding": "max_dia (spearman)", "value": d.eta.corr(d.max_dia, method="spearman")},
             {"finding": "max_dia among controls (spearman)", "value": ctrl.eta.corr(ctrl.max_dia, method="spearman")},
             {"finding": "AUC: nodule >= 10 mm among controls", "value": auc(ctrl.eta.to_numpy(), (ctrl.max_dia >= 10).to_numpy())},
             {"finding": "AUC: any nodule among controls", "value": auc(ctrl.eta.to_numpy(), ctrl.any_nodule.to_numpy())},
             {"finding": "AUC: spiculated among controls", "value": auc(ctrl.eta.to_numpy(), ctrl.spiculated.to_numpy())}]
    assoc += [{"finding": f"{f} (spearman)", "value": d.eta.corr(d[f], method="spearman")}
              for f in [*OTHER_FINDINGS.values(), "age", "lung_ml", "lung_mean_hu"]]

    no_nod = (d.any_nodule == 0).to_numpy()
    strata = {"all": np.ones(len(d), bool), "no T0 nodule": no_nod, "no event <= 1 y": ~early,
              "no T0 nodule, no event <= 1 y": no_nod & ~early, "T0 nodule >= 4 mm": ~no_nod}
    sc = pd.DataFrame([{"stratum": k, "n": int(m.sum()), "events": int(e[m].sum()),
                        "c_eta": harrell_c(d.eta[m], t[m], e[m]),
                        "c_max_dia": harrell_c(d.max_dia[m], t[m], e[m])} for k, m in strata.items()])
    av, extra = added_value(d, a, rng)
    summary = {"split": split, "n": len(d), "events": int(e.sum()),
               **{f"c_{r.stratum}": r.c_eta for r in sc.itertuples()},
               "auc_ctrl_nodule_ge10": assoc[2]["value"], "rho_max_dia": assoc[0]["value"],
               "rho_emphysema": d.eta.corr(d.emphysema, method="spearman"),
               **{f"cv_{r.model}": r.cv_c for r in av.itertuples()}, **extra}
    return {"by_nodule_size": by_size, "associations": pd.DataFrame(assoc), "strata_c": sc,
            "added_value": av, "summary": summary}


def plot(sizes: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted({r for r, _ in sizes})
    splits = sorted({s for _, s in sizes}, reverse=True)
    fig, axes = plt.subplots(len(splits), 2, figsize=(10, 3.6 * len(splits)), squeeze=False)
    x = np.arange(len(DIA_LABELS))
    for i, sp in enumerate(splits):
        for j, grp in enumerate(["controls", "events"]):
            ax = axes[i, j]
            for k, r in enumerate(runs):
                b = sizes[(r, sp)]
                b = b[b.group == grp].set_index("dia_bin").reindex(DIA_LABELS)
                ax.errorbar(x + (k - len(runs) / 2) * 0.06, b["mean"], yerr=1.96 * b["sem"], marker="o",
                            capsize=2, label=r)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xticks(x, [f"{l}\n(n={int(n) if n == n else 0})" for l, n in
                              zip(DIA_LABELS, sizes[(runs[0], sp)].query("group == @grp").set_index("dia_bin")
                                  .reindex(DIA_LABELS)["size"])], fontsize=7)
            ax.set_title(f"{sp}: {grp}", fontsize=9)
            ax.set_xlabel("largest T0 nodule [mm]", fontsize=8)
            ax.set_ylabel("eta [val SD] vs nodule-free controls", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    t0 = t0_findings(cohort, a.ctab)
    t0.to_csv(T0_CSV, index=False)
    print(f"T0 findings -> {T0_CSV}: {int(t0.any_nodule.sum())}/{len(t0)} with a nodule >= 4 mm")
    covs = cohort[["pid", "age", "sex", "cigsmok", "lung_ml", "lung_mean_hu"]]

    tables = {k: [] for k in ("by_nodule_size", "associations", "strata_c", "added_value")}
    summary, sizes = [], {}
    for run in a.runs.split(","):
        pred = pd.read_csv(Path(a.runs_dir) / run / "pred.csv", dtype={"pid": str}, usecols=["pid", "split", "time", "event", "eta"])
        pred = pred[pred.split.isin(["train", "val"])].merge(covs, on="pid")  # test rows never kept
        for split in a.splits.split(","):
            r = analyse(pred, t0, split, a, rng)
            for k in tables:
                tables[k].append(r[k].assign(run=run, split=split))
            summary.append({"run": run, **r["summary"]})
            sizes[(run, split)] = r["by_nodule_size"]
            print(f"\n== {run} / {split}")
            print(r["strata_c"].round(3).to_string(index=False))
            print(r["added_value"].round(3).to_string(index=False))
            print({k: round(v, 3) for k, v in r["summary"].items() if isinstance(v, float)
                   and k in ("auc_ctrl_nodule_ge10", "rho_max_dia", "rho_emphysema", "hr_eta_adj_t0",
                             "lr_p_eta_over_clinical_t0")})
    for k, v in tables.items():
        pd.concat(v).to_csv(out / f"{k}.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    plot(sizes, out / "by_nodule_size.png")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
