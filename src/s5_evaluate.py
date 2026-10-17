#!/usr/bin/env python3
"""S5 -- evaluate a trained run on the held-out test split (and val, for reference).

This is the only script that reads test predictions. Metrics per split:
  c_index         Harrell C of eta, with a 95% patient-bootstrap CI
  c_after_1y      Harrell C among patients still event-free and in follow-up
                  at 1 year. Drops the cancers already visible on the T0 scan
                  (27% of events are diagnosed within a year), so this is the
                  "predicts future cancer" number.
  auc_{1,2,3,5}y  cumulative/dynamic time-dependent AUC (sksurv, IPCW with the
                  train split's censoring distribution)
  c_age           Harrell C of age alone (reference for how much the image adds)
KM figure: test patients in eta tertiles, cut points taken from val.

Adjustment for sex and smoking (`--adjust`, default "sex,cigsmok"; test split).
Does the score still separate risk once sex and current/former smoking are
accounted for?
  km_test_strata.png      KM by eta tertile inside each sex x smoking group
                          (4 panels). Separation within every panel = not
                          explained by those two variables.
  km_test_adjusted.png    left: unadjusted KM by tertile; right: covariate-
                          adjusted curves from a Cox model
                          tertile + adjust covariates, standardised to the
                          test population (each curve = mean predicted
                          survival if every test patient were in that
                          tertile), with patient-bootstrap 95% bands.
  adjusted_hr.csv         Cox hazard ratios, unadjusted vs adjusted:
                          eta per val-SD, and mid/high vs low tertile.

Added value over clinical variables (always run, test split). Does eta improve
discrimination beyond what simple variables give? Three Cox models are fitted on
val (--fit-split; train would overweight eta, which the CNN fitted there) and
scored on test:
  clinical       --clinical covariates (default age, sex, cigsmok, lung_ml,
                 lung_mean_hu: the lung size/density the CNN is known to track)
  clinical+eta   the same + eta per val-SD
  eta            eta alone
  added_value.csv      C (95% bootstrap CI), C after 1 y, AUC 1/2/3/5 y per model,
                       and the row "delta": clinical+eta minus clinical, with a
                       paired-bootstrap CI (both models on the same resampled test
                       patients) and p_boot_le0 = share of resamples with delta <= 0
  added_value_fit.csv  the fitted HRs + likelihood-ratio test of adding eta (on val)

Usage:
    env/.venv/bin/python src/s5_evaluate.py --run baseline [--n-boot 1000]
    env/.venv/bin/python src/s5_evaluate.py --run cosine --added-only --out eval_added
-> runs/{run}/{--out, default eval}/metrics.csv, km_test.png, km_test_strata.png,
   km_test_adjusted.png, adjusted_hr.csv, added_value.csv, added_value_fit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import harrell_c
from nlst import COHORT_CSV, RUNS_DIR

YEARS = (1, 2, 3, 5)
TERTILES = ["low", "mid", "high"]
SMOKING = {0: "former smoker", 1: "current smoker"}  # NLST cigsmok
SEX = {"F": "female", "M": "male"}


def covariates(df: pd.DataFrame, adjust: list[str]) -> pd.DataFrame:
    """Numeric design columns for the adjustment covariates (sex -> male 0/1)."""
    out = pd.DataFrame(index=df.index)
    for c in adjust:
        out["male" if c == "sex" else c] = (df[c] == "M").astype(float) if c == "sex" else df[c].astype(float)
    return out


def tertile_dummies(grp: np.ndarray, index) -> pd.DataFrame:
    return pd.DataFrame({"mid_vs_low": (grp == 1).astype(float),
                         "high_vs_low": (grp == 2).astype(float)}, index=index)


def surv(df: pd.DataFrame):
    from sksurv.util import Surv
    return Surv.from_arrays(df["event"].astype(bool).to_numpy(), df["time"].to_numpy(float))


def split_metrics(df: pd.DataFrame, train: pd.DataFrame, n_boot: int, rng) -> dict:
    from sksurv.metrics import cumulative_dynamic_auc
    eta, t, e = df["eta"].to_numpy(), df["time"].to_numpy(), df["event"].to_numpy()
    c = harrell_c(eta, t, e)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        boot.append(harrell_c(eta[i], t[i], e[i]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
    late = df[df["time"] > 365]
    times = [y * 365.25 for y in YEARS]
    auc = []
    for tp in times:  # NaN where a horizon is outside the split's follow-up (tiny smoke splits)
        try:
            auc.append(float(cumulative_dynamic_auc(surv(train), surv(df), eta, [tp])[0][0]))
        except ValueError:
            auc.append(np.nan)
    out = {"n": len(df), "events": int(e.sum()),
           "c_index": c, "c_lo": lo, "c_hi": hi,
           "c_after_1y": harrell_c(late["eta"], late["time"], late["event"]),
           "events_after_1y": int(late["event"].sum()),
           "c_age": harrell_c(df["age"], t, e)}
    out.update({f"auc_{y}y": a for y, a in zip(YEARS, auc)})
    return out


def km_figure(test: pd.DataFrame, cuts, path: Path, title: str) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    grp = np.digitize(test["eta"], cuts)  # 0 low, 1 mid, 2 high
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g, name in enumerate(["low", "mid", "high"]):
        d = test[grp == g]
        KaplanMeierFitter(label=f"{name} risk (n={len(d)}, {int(d.event.sum())} events)") \
            .fit(d["time"] / 365.25, d["event"]).plot_survival_function(ax=ax, ci_show=True)
    p = multivariate_logrank_test(test["time"], grp, test["event"]).p_value
    ax.set_xlabel("years since T0 scan"); ax.set_ylabel("lung-cancer-free probability")
    ax.set_title(f"{title}\ntest split, eta tertiles from val, log-rank p = {p:.2g}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(p)


def km_strata_figure(test: pd.DataFrame, grp: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for r, sex in enumerate(SEX):
        for c, smk in enumerate(SMOKING):
            ax = axes[r, c]
            m = ((test["sex"] == sex) & (test["cigsmok"] == smk)).to_numpy()
            d, g = test[m], grp[m]
            for k, name in enumerate(TERTILES):
                dk = d[g == k]
                if len(dk):
                    KaplanMeierFitter(label=f"{name} (n={len(dk)}, {int(dk.event.sum())} ev)") \
                        .fit(dk["time"] / 365.25, dk["event"]).plot_survival_function(ax=ax, ci_show=True)
            p = multivariate_logrank_test(d["time"], g, d["event"]).p_value
            ax.set_title(f"{SEX[sex]}, {SMOKING[smk]}: n={len(d)}, {int(d.event.sum())} events, "
                         f"log-rank p={p:.2g}", fontsize=9)
            ax.set_xlabel("years since T0 scan"); ax.set_ylabel("lung-cancer-free probability")
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{title}: test split, eta tertiles (cut points from val) within sex x smoking",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def adjusted_curves(df: pd.DataFrame, grp: np.ndarray, adj: pd.DataFrame, t_grid: np.ndarray):
    """Direct-standardised survival per tertile from Cox(tertile + adj), shape (3, len(t_grid))."""
    from lifelines import CoxPHFitter
    X = pd.concat([tertile_dummies(grp, df.index), adj], axis=1)
    cph = CoxPHFitter().fit(X.assign(time=df["time"], event=df["event"]), "time", "event")
    curves = []
    for k in range(3):
        Xk = X.copy()
        Xk[["mid_vs_low", "high_vs_low"]] = tertile_dummies(np.full(len(df), k), df.index).to_numpy()
        curves.append(cph.predict_survival_function(Xk, times=t_grid).mean(axis=1).to_numpy())
    return np.array(curves)


def km_adjusted_figure(test: pd.DataFrame, grp: np.ndarray, adj: pd.DataFrame, adjust: list[str],
                       path: Path, title: str, n_boot: int, rng) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    t_grid = np.linspace(0, test["time"].max(), 200)
    est = adjusted_curves(test, grp, adj, t_grid)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(test), len(test))
        d = test.iloc[i].reset_index(drop=True)
        try:
            boot.append(adjusted_curves(d, grp[i], adj.iloc[i].reset_index(drop=True), t_grid))
        except Exception:  # rare non-converging resample
            pass
    lo, hi = (np.percentile(boot, [2.5, 97.5], axis=0) if boot else (None, None))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = ["C0", "C1", "C2"]
    for k, name in enumerate(TERTILES):
        d = test[grp == k]
        KaplanMeierFitter(label=f"{name} (n={len(d)}, {int(d.event.sum())} ev)") \
            .fit(d["time"] / 365.25, d["event"]).plot_survival_function(ax=a1, ci_show=True, color=colors[k])
        a2.plot(t_grid / 365.25, est[k], color=colors[k], label=name)
        if lo is not None:
            a2.fill_between(t_grid / 365.25, lo[k], hi[k], color=colors[k], alpha=0.2, lw=0)
    a1.set_title("unadjusted Kaplan-Meier", fontsize=9)
    a2.set_title(f"adjusted for {' + '.join(adjust)} (Cox, standardised to the test\n"
                 f"population; {len(boot)}x bootstrap 95% bands)", fontsize=9)
    for ax in (a1, a2):
        ax.set_xlabel("years since T0 scan")
        ax.legend(fontsize=8, loc="lower left")
    a1.set_ylabel("lung-cancer-free probability")
    fig.suptitle(f"{title}: test split, eta tertiles (cut points from val)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hazard_ratios(test: pd.DataFrame, grp: np.ndarray, adj: pd.DataFrame, eta_sd: float,
                  eta_mean: float) -> pd.DataFrame:
    """Cox HRs on test: eta per val-SD and tertiles, each without and with adjustment."""
    from lifelines import CoxPHFitter
    ty = test[["time", "event"]]
    eta_z = pd.DataFrame({"eta_per_sd": (test["eta"] - eta_mean) / eta_sd}, index=test.index)
    terts = tertile_dummies(grp, test.index)
    rows = []
    for exposure in (eta_z, terts):
        for label, X in (("unadjusted", exposure), ("adjusted", pd.concat([exposure, adj], axis=1))):
            s = CoxPHFitter().fit(pd.concat([X, ty], axis=1), "time", "event").summary
            for term in s.index:
                rows.append({"model": label, "term": term, "hr": s.loc[term, "exp(coef)"],
                             "hr_lo": s.loc[term, "exp(coef) lower 95%"],
                             "hr_hi": s.loc[term, "exp(coef) upper 95%"], "p": s.loc[term, "p"]})
    return pd.DataFrame(rows)


def added_value(pred: pd.DataFrame, clinical: list[str], fit_split: str, n_boot: int, rng):
    """Does eta improve discrimination over clinical variables? Cox models fitted on
    `fit_split`, scored on test: clinical, clinical + eta, eta alone. Returns
    (test metrics per model + the clinical -> clinical+eta difference, fitted coefficients)."""
    from lifelines import CoxPHFitter
    from scipy.stats import chi2
    from sksurv.metrics import cumulative_dynamic_auc
    fit = pred[pred.split == fit_split].reset_index(drop=True)
    test = pred[pred.split == "test"].reset_index(drop=True)
    train = pred[pred.split == "train"]
    m, sd = fit["eta"].mean(), fit["eta"].std()
    fill = covariates(fit, clinical).median()  # guard: lifelines rejects NaN

    def design(d: pd.DataFrame, name: str) -> pd.DataFrame:
        X = covariates(d, clinical).fillna(fill)
        z = pd.DataFrame({"eta_per_sd": (d["eta"] - m) / sd}, index=d.index)
        return {"clinical": X, "clinical+eta": pd.concat([X, z], axis=1), "eta": z}[name]

    names = ["clinical", "clinical+eta", "eta"]
    lp, coefs, ll = {}, [], {}
    for name in names:
        Xf = design(fit, name)
        cph = CoxPHFitter().fit(Xf.assign(time=fit["time"], event=fit["event"]), "time", "event")
        lp[name] = cph.predict_log_partial_hazard(design(test, name)).to_numpy()
        ll[name] = cph.log_likelihood_
        s = cph.summary
        coefs += [{"model": name, "term": t, "hr": s.loc[t, "exp(coef)"], "hr_lo": s.loc[t, "exp(coef) lower 95%"],
                   "hr_hi": s.loc[t, "exp(coef) upper 95%"], "p": s.loc[t, "p"]} for t in s.index]
    lr = 2 * (ll["clinical+eta"] - ll["clinical"])
    coefs.append({"model": f"LR test on {fit_split}: clinical+eta vs clinical", "term": "eta_per_sd",
                  "hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p": float(chi2.sf(lr, 1))})

    t, e = test["time"].to_numpy(), test["event"].to_numpy()
    late = (t > 365)
    rows = []
    for name in names:
        row = {"model": name, "c_index": harrell_c(lp[name], t, e),
               "c_after_1y": harrell_c(lp[name][late], t[late], e[late])}
        for y in YEARS:
            try:
                row[f"auc_{y}y"] = float(cumulative_dynamic_auc(surv(train), surv(test), lp[name],
                                                                [y * 365.25])[0][0])
            except ValueError:
                row[f"auc_{y}y"] = np.nan
        rows.append(row)
    res = pd.DataFrame(rows).set_index("model")

    # paired bootstrap of test patients (no refit): both models scored on the same resample
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        i = rng.integers(0, len(test), len(test))
        for n in names:
            boot[n].append(harrell_c(lp[n][i], t[i], e[i]))
    boot = {n: np.asarray(v) for n, v in boot.items()}
    for n in names:
        res.loc[n, ["c_lo", "c_hi"]] = np.nanpercentile(boot[n], [2.5, 97.5]) if n_boot else np.nan
    d = boot["clinical+eta"] - boot["clinical"]
    res["p_boot_le0"] = np.nan
    delta = {c: res.loc["clinical+eta", c] - res.loc["clinical", c] for c in res.columns
             if c not in ("c_lo", "c_hi", "p_boot_le0")}
    delta.update({"c_lo": np.nanpercentile(d, 2.5) if n_boot else np.nan,
                  "c_hi": np.nanpercentile(d, 97.5) if n_boot else np.nan,
                  "p_boot_le0": float(np.nanmean(d <= 0)) if n_boot else np.nan})
    res.loc["delta (clinical+eta - clinical)"] = pd.Series(delta)
    res = res.reset_index()[["model", "c_index", "c_lo", "c_hi", "c_after_1y",
                             *[f"auc_{y}y" for y in YEARS], "p_boot_le0"]]
    return res, pd.DataFrame(coefs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under runs/")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-curves", type=int, default=200,
                    help="bootstrap resamples for the adjusted-curve bands")
    ap.add_argument("--adjust", default="sex,cigsmok",
                    help="comma-separated cohort.csv columns to adjust for (sex -> male 0/1)")
    ap.add_argument("--clinical", default="age,sex,cigsmok,lung_ml,lung_mean_hu",
                    help="covariates of the clinical reference model for the added-value analysis")
    ap.add_argument("--fit-split", choices=["val", "train"], default="val",
                    help="split the clinical / clinical+eta Cox models are fitted on (never test)")
    ap.add_argument("--added-only", action="store_true",
                    help="only the added-value analysis (skips metrics, KM figures, adjusted HRs)")
    ap.add_argument("--out", default="eval", help="output folder under runs/{run}/")
    a = ap.parse_args()
    adjust = [c for c in a.adjust.split(",") if c]
    clinical = [c for c in a.clinical.split(",") if c]

    run = Path(a.runs_dir) / a.run
    pred = pd.read_csv(run / "pred.csv", dtype={"pid": str})
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})[
        ["pid"] + sorted({"age", "sex", "cigsmok", *adjust, *clinical})]
    pred = pred.merge(cohort, on="pid", how="left")
    train = pred[pred.split == "train"]
    rng = np.random.default_rng(0)
    out = run / a.out
    out.mkdir(exist_ok=True)

    av, av_fit = added_value(pred, clinical, a.fit_split, a.n_boot, rng)
    av.to_csv(out / "added_value.csv", index=False)
    av_fit.to_csv(out / "added_value_fit.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"Added value of eta on test (Cox models fitted on {a.fit_split}; "
              f"clinical = {', '.join(clinical)}; {a.n_boot}x paired bootstrap):")
        print(av.round(4).to_string(index=False))
        print(f"\nfitted on {a.fit_split}:")
        print(av_fit.round(4).to_string(index=False), "\n")
    if a.added_only:
        print(f"{a.run}: wrote {out / 'added_value.csv'}, {out / 'added_value_fit.csv'}")
        return

    rows = []
    for s in ("val", "test"):
        rows.append({"split": s, **split_metrics(pred[pred.split == s], train, a.n_boot, rng)})
    res = pd.DataFrame(rows)
    cuts = np.quantile(pred.loc[pred.split == "val", "eta"], [1 / 3, 2 / 3])
    res["km_logrank_p"] = [np.nan, km_figure(pred[pred.split == "test"], cuts, out / "km_test.png", a.run)]
    res.to_csv(out / "metrics.csv", index=False)

    # adjustment for sex / smoking (test split)
    test = pred[pred.split == "test"].reset_index(drop=True)
    grp = np.digitize(test["eta"], cuts)
    adj = covariates(test, adjust)
    val_eta = pred.loc[pred.split == "val", "eta"]
    hr = hazard_ratios(test, grp, adj, val_eta.std(), val_eta.mean())
    hr.to_csv(out / "adjusted_hr.csv", index=False)
    km_strata_figure(test, grp, out / "km_test_strata.png", a.run)
    km_adjusted_figure(test, grp, adj, adjust, out / "km_test_adjusted.png", a.run,
                       a.n_boot_curves, rng)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))
        print(f"\nCox hazard ratios on test (adjusted for {', '.join(adjust)}):")
        print(hr.round(4).to_string(index=False))
    sel = json.loads((run / "config.json").read_text())
    print(f"\n{a.run}: n_train {sel['n_train']} ({sel['events_train']} events); wrote {out}")


if __name__ == "__main__":
    main()
