#!/usr/bin/env python3
"""C4 -- evaluate the lungs / sternum CNNs on the held-out TEST set, once.

Reads runs/grid/*/pred.csv and runs/final/*/pred.csv (never the "grid"-tag
tuning runs' test predictions -- those were only used for selection on val).

Model groups (per organ):
  old_ckpt       the saved legacy checkpoint (trained on 179 pts), just re-scored
  legacy_subset  legacy recipe retrained on the same 179 old-subset train pts
  legacy_scale   legacy recipe (64^3 squash, old windows, BCE) on all 2,191 train pts
  new_main       new recipe (iso input + mask channel + proper windows + Cox +
                 aug + early stopping), chosen config, 5 seeds
  new_bce        new_main with BCE loss instead of Cox
  new_noaug      new_main without augmentation
  new_lc_*       new_main trained on subset_v1 (179) / 25% / 50% of train
  null           new_main trained on permuted (time, event) -> should be ~0.5

Per group: Harrell C (stratified bootstrap 95% CI), Uno C (tau = 6 y),
time-dependent AUC at 1/3/5 y, binary event AUC (for comparison with the old
numbers only), train/val/test C (overfitting gap), seed spread. Multi-seed
groups are ensembled (mean of per-seed val-standardised eta).

Plus:
  - paired-bootstrap Delta C for the key contrasts (new vs legacy vs old, lungs vs sternum)
  - KM tertile curves + multivariate log-rank (mentor convention)
  - "why" analyses: lead time (events <= 1 y vs 1-y landmark), kernel strata,
    Spearman of the score with image covariates (lung volume, LAA-950, MLD,
    Perc15 / sternum volume, HU) and -- when available -- age/sex/smoking;
    a Cox model of the score adjusted for those; and a covariate-only Cox
    baseline ("how far does a hand-crafted density/size meter get?")
  - clinical adjustment (age, sex, cigsmok): runs only if the cohort CSV has
    those columns filled (see docs/cnn3d_explained.md, "Clinical covariates")
  - learning-curve figure, decomposition table

Outputs: data/c4_*.csv, data/c4_summary.json, figs/c4_*.png

Usage:
    env/.venv/bin/python src/c4_evaluate.py [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from cnn3d import harrell_c, CACHE_DIR, RUNS_DIR, PROJECT

warnings.filterwarnings("ignore", category=RuntimeWarning)
COHORT_CSV = PROJECT / "data" / "pid_lists" / "cnn_cohort_v1.csv"
DATA_OUT = PROJECT / "data"
FIG_OUT = PROJECT / "figs"
ORGANS = ["lungs", "sternum"]
TAU_DAYS = 6 * 365.25
TD_YEARS = [1, 3, 5]
IMAGE_COVS = {"lungs": ["volume_ml", "laa950_pct", "mld_hu", "perc15_hu"],
              "sternum": ["volume_ml", "mean_hu", "median_hu"]}
GROUP_ORDER = ["old_ckpt", "legacy_subset", "legacy_scale", "new_lc_subset_v1", "new_lc_0.25",
               "new_lc_0.5", "new_main", "new_noaug", "new_bce", "null"]


# ── loading ──────────────────────────────────────────────────────────────────

def group_name(cfg: dict) -> str | None:
    tag = cfg["tag"]
    if cfg["set"] == "grid":
        return {"score_old": "old_ckpt", "legacy_subset": "legacy_subset",
                "legacy_scale": "legacy_scale"}.get(tag)          # "grid" tuning runs: excluded
    if tag == "lc":
        return f"new_lc_{cfg['train_subset']}"
    return {"main": "new_main", "bce": "new_bce", "noaug": "new_noaug", "null": "null"}[tag]


def load_groups(runs_dir: Path) -> dict:
    """{(organ, group): {"cfgs": [...], "preds": [DataFrame per seed]}}"""
    groups: dict = {}
    for res in sorted(runs_dir.glob("*/*/result.json")):
        cfg = json.loads(res.read_text())
        g = group_name(cfg)
        if g is None:
            continue
        pred = pd.read_csv(res.parent / "pred.csv", dtype={"pid": str})
        d = groups.setdefault((cfg["organ"], g), {"cfgs": [], "preds": []})
        d["cfgs"].append(cfg)
        d["preds"].append(pred)
    return groups


def ensemble(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean of per-seed val-standardised eta, then re-standardised on val."""
    base = preds[0][["pid", "split", "time", "event"]].copy()
    zs = []
    for p in preds:
        assert (p["pid"].values == base["pid"].values).all()
        v = p.loc[p["split"] == "val", "eta"]
        zs.append((p["eta"] - v.mean()) / v.std())
    s = np.mean(zs, axis=0)
    v = s[base["split"].values == "val"]
    base["score"] = (s - v.mean()) / v.std()
    return base


# ── metrics ──────────────────────────────────────────────────────────────────

def surv(df):
    return Surv.from_arrays(df["event"].astype(bool).values, df["time"].values)


def boot_idx(event: np.ndarray, n_boot: int, seed: int = 0):
    """Event-stratified bootstrap: resample cases and controls separately, so
    every replicate keeps the observed number of events."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(event == 1)[0], np.where(event == 0)[0]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def c_with_ci(df: pd.DataFrame, col: str, n_boot: int):
    c = harrell_c(df["time"], df[col], df["event"])
    t, s, e = df["time"].values, df[col].values, df["event"].values
    bs = [harrell_c(t[i], s[i], e[i]) for i in boot_idx(e, n_boot)]
    return c, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def all_metrics(ens: pd.DataFrame, train_ref: pd.DataFrame, n_boot: int) -> dict:
    te = ens[ens["split"] == "test"]
    c, lo, hi = c_with_ci(te, "score", n_boot)
    out = {"test_n": len(te), "test_events": int(te["event"].sum()),
           "harrell_c": c, "c_lo": lo, "c_hi": hi}
    try:
        out["uno_c"] = concordance_index_ipcw(surv(train_ref), surv(te), te["score"].values, tau=TAU_DAYS)[0]
    except ValueError:
        out["uno_c"] = np.nan
    times = [y * 365.25 for y in TD_YEARS]
    try:
        aucs, _ = cumulative_dynamic_auc(surv(train_ref), surv(te), te["score"].values, times)
        out.update({f"td_auc_{y}y": a for y, a in zip(TD_YEARS, aucs)})
    except ValueError:
        out.update({f"td_auc_{y}y": np.nan for y in TD_YEARS})
    out["binary_auc"] = roc_auc_score(te["event"], te["score"])
    for s in ("train", "val"):
        d = ens[ens["split"] == s]
        out[f"{s}_c"] = harrell_c(d["time"], d["score"], d["event"])
    return out


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int):
    """Delta C (a - b) on the test pids both models scored, paired bootstrap."""
    m = a[a.split == "test"][["pid", "time", "event", "score"]].merge(
        b[b.split == "test"][["pid", "score"]], on="pid", suffixes=("_a", "_b"))
    t, e = m["time"].values, m["event"].values
    sa, sb = m["score_a"].values, m["score_b"].values
    d = harrell_c(t, sa, e) - harrell_c(t, sb, e)
    bs = np.array([harrell_c(t[i], sa[i], e[i]) - harrell_c(t[i], sb[i], e[i])
                   for i in boot_idx(e, n_boot, seed=1)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return d, np.percentile(bs, 2.5), np.percentile(bs, 97.5), min(p, 1.0), len(m)


def cox_hr(df: pd.DataFrame, cols: list[str], key: str = "score") -> tuple:
    d = df[["time", "event"] + cols].dropna()
    cph = CoxPHFitter(penalizer=1e-4).fit(d, "time", "event")
    s = cph.summary.loc[key]
    return float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"]), len(d)


# ── plots ────────────────────────────────────────────────────────────────────

def km_panel(ax, te: pd.DataFrame, title: str) -> float:
    te = te.copy()
    labels = ["Low", "Mid", "High"]
    te["group"] = pd.qcut(te["score"].rank(method="first"), 3, labels=labels)  # tie-safe tertiles
    for lab, col in zip(labels, ["tab:blue", "tab:orange", "tab:red"]):
        sub = te[te["group"] == lab]
        KaplanMeierFitter().fit(sub["time"] / 365.25, sub["event"],
                                label=f"{lab} (n={len(sub)}, ev={int(sub.event.sum())})"
                                ).plot_survival_function(ax=ax, color=col, ci_show=False)
    p = multivariate_logrank_test(te["time"], te["group"], te["event"]).p_value
    ax.set_title(f"{title}\nlog-rank p={p:.2g}", fontsize=9)
    ax.set_xlabel("years from baseline CT")
    ax.set_ylabel("lung-cancer-free")
    ax.legend(fontsize=7, loc="lower left")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = ap.parse_args()

    cohort = pd.read_csv(COHORT_CSV, dtype={"pid": str})
    clin_ok = cohort[["age", "sex", "cigsmok"]].notna().all(axis=1).all()
    cohort["male"] = (cohort["sex"] == "M").astype(float).where(cohort["sex"].notna())
    cohort["sharp_kernel"] = (cohort["kernel_group"] == "sharp").astype(float)
    clin_cols = ["age", "male", "cigsmok"]
    print(f"clinical covariates available: {clin_ok}")

    groups = load_groups(Path(args.runs_dir))
    print("groups found:", sorted(groups))
    ens = {k: ensemble(v["preds"]) for k, v in groups.items()}
    train_ref = cohort[cohort.split == "train"]  # censoring distribution for Uno C / td-AUC

    # 1) per-group metrics ----------------------------------------------------
    rows = []
    for (organ, g), e in ens.items():
        cfgs = groups[(organ, g)]["cfgs"]
        m = all_metrics(e, train_ref, args.n_boot)
        seed_c = [harrell_c(p[p.split == "test"].time, p[p.split == "test"].eta, p[p.split == "test"].event)
                  for p in groups[(organ, g)]["preds"]]
        train_n = cfgs[0].get("train_n", 179 if g == "old_ckpt" else np.nan)
        rows.append({"organ": organ, "group": g, "n_seeds": len(cfgs), "train_n": train_n,
                     "train_events": cfgs[0].get("train_events", 83 if g == "old_ckpt" else np.nan),
                     **m, "seed_c_mean": np.mean(seed_c), "seed_c_sd": np.std(seed_c, ddof=1) if len(seed_c) > 1 else np.nan})
        # old checkpoint on the OLD test subset (127 pts / 18 events): reproduces the 0.687 claim
        if g == "old_ckpt":
            old = e[e.pid.isin(cohort.loc[cohort.in_subset_v1 == 1, "pid"])]
            te = old[old.split == "test"]
            c, lo, hi = c_with_ci(te, "score", args.n_boot)
            rows.append({"organ": organ, "group": "old_ckpt@old_test", "n_seeds": 1, "train_n": 179,
                         "test_n": len(te), "test_events": int(te.event.sum()), "harrell_c": c,
                         "c_lo": lo, "c_hi": hi, "binary_auc": roc_auc_score(te.event, te.score)})
    met = pd.DataFrame(rows)
    met["order"] = met["group"].map({g: i for i, g in enumerate(GROUP_ORDER)}).fillna(-1)
    met = met.sort_values(["organ", "order"]).drop(columns="order")
    met.to_csv(DATA_OUT / "c4_test_metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(met.round(3).to_string(index=False))

    # 2) paired Delta C ------------------------------------------------------
    contrasts = [(o, "new_main", o, "legacy_scale") for o in ORGANS] + \
                [(o, "new_main", o, "old_ckpt") for o in ORGANS] + \
                [(o, "legacy_scale", o, "old_ckpt") for o in ORGANS] + \
                [(o, "new_main", o, "new_bce") for o in ORGANS] + \
                [(o, "new_main", o, "new_noaug") for o in ORGANS] + \
                [("lungs", "new_main", "sternum", "new_main")]
    drows = []
    for oa, ga, ob, gb in contrasts:
        if (oa, ga) in ens and (ob, gb) in ens:
            d, lo, hi, p, n = paired_delta(ens[(oa, ga)], ens[(ob, gb)], args.n_boot)
            drows.append({"a": f"{oa}:{ga}", "b": f"{ob}:{gb}", "delta_c": d, "lo": lo, "hi": hi,
                          "p_boot": p, "n_paired": n})
    dfd = pd.DataFrame(drows)
    dfd.to_csv(DATA_OUT / "c4_delta_c.csv", index=False)
    print("\nPaired Delta C:\n", dfd.round(3).to_string(index=False))

    # 3) "why" analyses on the headline model ---------------------------------
    why_rows, corr_rows, adj_rows = [], [], []
    for organ in ORGANS:
        idx = pd.read_csv(Path(args.cache_dir) / f"{organ}_index.csv", dtype={"pid": str})
        for g in ("new_main", "legacy_scale", "old_ckpt", "new_bce"):
            if (organ, g) not in ens:
                continue
            te = ens[(organ, g)]
            te = te[te.split == "test"].merge(idx[["pid"] + IMAGE_COVS[organ]], on="pid") \
                .merge(cohort[["pid", "kernel_group", "sharp_kernel", "manufacturer"] + clin_cols + ["sex"]], on="pid")
            # lead time
            early = te[(te.event == 0) | (te.time <= 365.25)]
            late = te[te.time > 365.25]
            why_rows.append({"organ": organ, "group": g, "analysis": "events <= 1y vs all controls (binary AUC)",
                             "n": len(early), "events": int(early.event.sum()),
                             "value": roc_auc_score(early.event, early.score)})
            why_rows.append({"organ": organ, "group": g, "analysis": "1-y landmark (time > 1y) Harrell C",
                             "n": len(late), "events": int(late.event.sum()),
                             "value": harrell_c(late.time, late.score, late.event)})
            for kg in ("soft", "sharp"):
                s = te[te.kernel_group == kg]
                why_rows.append({"organ": organ, "group": g, "analysis": f"kernel={kg} Harrell C",
                                 "n": len(s), "events": int(s.event.sum()),
                                 "value": harrell_c(s.time, s.score, s.event)})
            if g != "new_main":
                continue
            # what does the score correlate with? (test set)
            probes = IMAGE_COVS[organ] + ["sharp_kernel"] + (clin_cols if clin_ok else [])
            for cov in probes:
                rho, p = spearmanr(te["score"], te[cov], nan_policy="omit")
                corr_rows.append({"organ": organ, "covariate": cov, "spearman_rho": rho, "p": p})
            # HR per SD of the score: unadjusted -> + kernel -> + image covariates (-> + clinical)
            te["score_sd"] = te["score"] / te["score"].std()
            sets = [("unadjusted", []), ("+ kernel", ["sharp_kernel"]),
                    ("+ kernel + image covariates", ["sharp_kernel"] + IMAGE_COVS[organ])]
            if clin_ok:
                sets += [("+ age/sex/smoking", clin_cols),
                         ("+ age/sex/smoking + kernel", clin_cols + ["sharp_kernel"]),
                         ("+ age/sex/smoking + kernel + image covariates",
                          clin_cols + ["sharp_kernel"] + IMAGE_COVS[organ])]
            for name, cols in sets:
                hr, lo, hi, p, n = cox_hr(te, ["score_sd"] + cols, "score_sd")
                adj_rows.append({"organ": organ, "adjustment": name, "hr_per_sd": hr, "lo": lo, "hi": hi,
                                 "p": p, "n": n})

    pd.DataFrame(why_rows).to_csv(DATA_OUT / "c4_why_leadtime_kernel.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(DATA_OUT / "c4_why_correlations.csv", index=False)
    pd.DataFrame(adj_rows).to_csv(DATA_OUT / "c4_hr_adjusted.csv", index=False)
    print("\nWhy (lead time / kernel):\n", pd.DataFrame(why_rows).round(3).to_string(index=False))
    print("\nScore correlations:\n", pd.DataFrame(corr_rows).round(3).to_string(index=False))
    print("\nHR per SD, adjusted:\n", pd.DataFrame(adj_rows).round(3).to_string(index=False))

    # 4) reference models fit on TRAIN (no CNN) + CNN combinations fit on VAL --
    # A CNN's train-set predictions are overfit (train C >> val C), so a Cox
    # model combining "CNN score + X" must be fit where the CNN did not learn:
    # the val set. Reference models without a CNN are fit on train as usual.
    ref_rows = []
    for organ in ORGANS:
        if (organ, "new_main") not in ens:
            continue
        idx = pd.read_csv(Path(args.cache_dir) / f"{organ}_index.csv", dtype={"pid": str})
        df = ens[(organ, "new_main")].merge(idx[["pid"] + IMAGE_COVS[organ]], on="pid") \
            .merge(cohort[["pid", "sharp_kernel"] + clin_cols], on="pid")
        specs = [("image covariates only", IMAGE_COVS[organ] + ["sharp_kernel"], False)]
        if clin_ok:
            specs += [("clinical only (age, sex, smoking)", clin_cols, False),
                      ("clinical + image covariates", clin_cols + IMAGE_COVS[organ] + ["sharp_kernel"], False)]
        specs += [("CNN score only", ["score"], True),
                  ("CNN + image covariates", ["score"] + IMAGE_COVS[organ] + ["sharp_kernel"], True)]
        if clin_ok:
            specs += [("clinical + CNN", clin_cols + ["score"], True),
                      ("clinical + CNN + image covariates", clin_cols + ["score"] + IMAGE_COVS[organ] + ["sharp_kernel"], True)]
        test = df[df.split == "test"].dropna(subset=specs[0][1])
        preds = {}
        for name, cols, uses_cnn in specs:
            fit_on = df[df.split == ("val" if uses_cnn else "train")][["time", "event"] + cols].dropna()
            cph = CoxPHFitter(penalizer=1e-3).fit(fit_on, "time", "event")
            t = test[["pid", "time", "event"] + cols].dropna().copy()
            t["score"] = cph.predict_log_partial_hazard(t[cols]).values
            preds[name] = t
            c, lo, hi = c_with_ci(t, "score", args.n_boot)
            ref_rows.append({"organ": organ, "model": name, "fit_on": "val" if uses_cnn else "train",
                             "test_n": len(t), "harrell_c": c, "c_lo": lo, "c_hi": hi})
        base = "clinical only (age, sex, smoking)" if clin_ok else "image covariates only"
        for name in preds:
            if name != base:
                a = preds[name].assign(split="test")
                b = preds[base].assign(split="test")
                d, lo, hi, p, n = paired_delta(a, b, args.n_boot)
                ref_rows.append({"organ": organ, "model": f"Delta C: {name} - {base}", "fit_on": "",
                                 "test_n": n, "harrell_c": d, "c_lo": lo, "c_hi": hi, "p_boot": p})
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(DATA_OUT / "c4_reference_models.csv", index=False)
    print("\nReference / combined models:\n", ref.round(3).to_string(index=False))

    # 5) figures ---------------------------------------------------------------
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for organ in ORGANS:
        gs = [g for g in ("old_ckpt", "legacy_scale", "new_main") if (organ, g) in ens]
        if not gs:
            continue
        fig, axes = plt.subplots(1, len(gs), figsize=(5.2 * len(gs), 4.3), squeeze=False)
        for ax, g in zip(axes[0], gs):
            te = ens[(organ, g)]
            km_panel(ax, te[te.split == "test"], f"{organ}: {g}")
        fig.tight_layout()
        fig.savefig(FIG_OUT / f"c4_km_{organ}.png", dpi=110)
        plt.close(fig)

    lc = met[met.group.str.startswith("new_lc_") | (met.group == "new_main") |
             met.group.isin(["legacy_subset", "legacy_scale"])].copy()
    if len(lc):
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for organ, col in zip(ORGANS, ["tab:blue", "tab:gray"]):
            n = lc[(lc.organ == organ) & lc.group.str.startswith("new")].sort_values("train_n")
            ax.errorbar(n.train_n, n.seed_c_mean, yerr=n.seed_c_sd.fillna(0), marker="o", color=col,
                        capsize=3, label=f"{organ}: new recipe (mean +- sd over seeds)")
            l = lc[(lc.organ == organ) & lc.group.str.startswith("legacy")].sort_values("train_n")
            ax.plot(l.train_n, l.seed_c_mean, "x--", color=col, label=f"{organ}: legacy recipe")
        ax.axhline(0.5, color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("training patients (log scale)")
        ax.set_ylabel("test Harrell C (2,0xx pts)")
        ax.set_title("Learning curve: does more data help?")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_OUT / "c4_learning_curve.png", dpi=110)
        plt.close(fig)

    # headline bar chart with CIs
    head = met[met.group.isin(["old_ckpt@old_test", "old_ckpt", "legacy_scale", "new_main", "new_bce", "null"])]
    if len(head):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        labels = [f"{r.organ}\n{r.group}" for r in head.itertuples()]
        cols = ["tab:blue" if o == "lungs" else "tab:gray" for o in head.organ]
        ax.bar(range(len(head)), head.harrell_c - 0.5, bottom=0.5, color=cols)
        ax.errorbar(range(len(head)), head.harrell_c, yerr=[head.harrell_c - head.c_lo, head.c_hi - head.harrell_c],
                    fmt="none", color="k", capsize=3)
        ax.axhline(0.5, color="k", lw=0.7)
        ax.set_xticks(range(len(head)), labels, fontsize=7)
        ax.set_ylabel("test Harrell C (95% bootstrap CI)")
        fig.tight_layout()
        fig.savefig(FIG_OUT / "c4_headline.png", dpi=110)
        plt.close(fig)

    # keep one training-curve figure per headline model in git (runs/ is ignored)
    cdir = FIG_OUT / "c4_training_curves"
    cdir.mkdir(exist_ok=True)
    for (organ, g), v in groups.items():
        if g in ("new_main", "legacy_scale", "new_bce", "null"):
            src = Path(args.runs_dir) / v["cfgs"][0]["set"] / v["cfgs"][0]["run_id"] / "curves.png"
            if src.exists():
                shutil.copy(src, cdir / f"{organ}_{g}.png")

    json.dump({"clinical_available": bool(clin_ok), "groups": [f"{o}:{g}" for o, g in sorted(groups)],
               "n_boot": args.n_boot}, open(DATA_OUT / "c4_summary.json", "w"), indent=1)
    print("\nwrote data/c4_*.csv, figs/c4_*.png")


if __name__ == "__main__":
    main()
