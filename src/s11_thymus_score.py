#!/usr/bin/env python3
"""S11 -- the published thymic-health score on our cohort (thymus step 0).

The question, before any thymus SSL is built: how much does a thymus model tell
us about time to lung cancer in this cohort, alone, on top of the clinical
variables, and on top of the supervised lung CNN? The score is the AIM-Harvard
thymic-health percentile (Bernatz et al. 2026, Nature; Zenodo 10.5281/zenodo.18306999):
one value per NLST participant, 0-100, higher = less fatty thymus. It is their
full system (thymic-bed segmenter -> SwAV ResNet-50 -> classifier fine-tuned on
expert involution grades), so it is roughly the ceiling for a thymus model here.

TEST PATIENTS ARE NEVER READ: cohort.csv and the reference pred.csv are cut to
train + val as soon as they are loaded. Every analysis uses the complete cases
(score present, and for rows with eta, the reference prediction present).

  coverage         score present per split, and how covered vs missing patients differ
  discrimination   Cox models fitted on TRAIN, Harrell C on VAL with a 1,000x patient
                   bootstrap, and paired-bootstrap delta C for
                     clinical -> clinical + thymus
                     clinical + eta -> clinical + eta + thymus
                   clinical = age, sex, cigsmok, lung_ml, lung_mean_hu (S5's set);
                   eta = the reference CNN's prediction (default runs/cosine)
  hazard ratios    HR per SD of the score (train SD), Cox models:
                     train + val: unadjusted, + age, + age/sex/smoking, + clinical
                     val only:    + clinical + eta (eta is fitted to train patients,
                                  so it is only used where it is out of sample),
                                  with a likelihood-ratio test of adding the score
  correlations     Spearman of the score with age, sex, smoking, lung_ml,
                   lung_mean_hu (train + val) and eta (val)
  histology        cause-specific Cox (other cancers censored), age/sex/smoking
                   adjusted, train + val: adenocarcinoma vs non-adenocarcinoma
                   (squamous, small cell, other), from prsn de_type (ICD-O-3)
  all-cause death  if prsn has death_days and fup_days: HR per SD for death from
                   any cause, time from the T0 scan, train + val

Outputs, runs/thymus_score/ (--out):
  coverage.csv, discrimination.csv, hazard_ratios.csv, correlations.csv,
  summary.json (arguments, counts, headline numbers)

Usage (login node, env/.venv, ~2 min):
    env/.venv/bin/python src/s11_thymus_score.py
    env/.venv/bin/python src/s11_thymus_score.py --ref cosine --n-boot 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2, spearmanr
from sksurv.metrics import concordance_index_censored

from nlst import COHORT_CSV, PRSN_CSV, RUNS_DIR, THYMUS_SCORE_CSV

CLINICAL = ["age", "sex", "cigsmok", "lung_ml", "lung_mean_hu"]
BASIC = ["age", "sex", "cigsmok"]
# ICD-O-3 morphology codes (prsn de_type), as in the thymus-heart-aging E3/E5 analyses
ADENO = {8140, 8250, 8251, 8252, 8253, 8254, 8255, 8260, 8323, 8480, 8481, 8490, 8550}


def cindex(eta: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    if event.sum() == 0:
        return float("nan")
    return float(concordance_index_censored(event.astype(bool), time, eta)[0])


def design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric design columns (sex -> male 0/1); th_z and eta_z pass through."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out["male" if c == "sex" else c] = (df[c] == "M").astype(float) if c == "sex" else df[c].astype(float)
    return out


def fit_cox(df: pd.DataFrame, cols: list[str], time: str = "time", event: str = "event") -> CoxPHFitter:
    X = design(df, cols)
    return CoxPHFitter().fit(X.assign(_t=df[time].to_numpy(float), _e=df[event].to_numpy(int)), "_t", "_e")


def hr_row(cph: CoxPHFitter, term: str, **info) -> dict:
    s = cph.summary.loc[term]
    return {**info, "term": term, "hr_per_sd": s["exp(coef)"], "hr_lo": s["exp(coef) lower 95%"],
            "hr_hi": s["exp(coef) upper 95%"], "p": s["p"], "n": int(cph._n_examples),
            "events": int(cph.event_observed.sum())}


def load(a) -> pd.DataFrame:
    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    cohort = cohort[cohort["split"].isin(["train", "val"])].reset_index(drop=True)  # test never read

    th = pd.read_csv(a.th_csv, dtype={"ID": str}).rename(
        columns={"ID": "pid", "Thymic_health_continuous": "th", "Thymic_health_categories": "th_cat"})
    if not th["pid"].is_unique:
        raise SystemExit(f"{a.th_csv}: duplicate IDs")
    df = cohort.merge(th[["pid", "th", "th_cat"]], on="pid", how="left", validate="1:1")

    ref = Path(a.runs_dir) / a.ref / "pred.csv"
    pred = pd.read_csv(ref, dtype={"pid": str})
    pred = pred[pred["split"].isin(["train", "val"])]
    df = df.merge(pred[["pid", "eta"]], on="pid", how="left", validate="1:1")

    prsn_cols = pd.read_csv(PRSN_CSV, nrows=0).columns
    use = ["pid", "de_type", "scr_days0"] + [c for c in ("death_days", "fup_days") if c in prsn_cols]
    prsn = pd.read_csv(PRSN_CSV, dtype={"pid": str}, usecols=use)
    df = df.merge(prsn, on="pid", how="left", validate="1:1")

    tr = df["split"] == "train"
    df["th_z"] = (df["th"] - df.loc[tr, "th"].mean()) / df.loc[tr, "th"].std()
    df["eta_z"] = (df["eta"] - df.loc[tr, "eta"].mean()) / df.loc[tr, "eta"].std()
    code = pd.to_numeric(df["de_type"], errors="coerce")
    df["histo"] = np.where(df["event"] == 0, "none",
                           np.where(code.isin(ADENO), "adeno", np.where(code.isna(), "unknown", "non_adeno")))
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, d in [("train", df[df.split == "train"]), ("val", df[df.split == "val"]), ("train+val", df)]:
        has = d["th"].notna()
        for label, g in (("with score", d[has]), ("without score", d[~has])):
            rows.append({"split": split, "group": label, "n": len(g), "events": int(g["event"].sum()),
                         "event_rate": g["event"].mean(), "age_mean": g["age"].mean(),
                         "male_frac": (g["sex"] == "M").mean(), "current_smoker_frac": g["cigsmok"].mean(),
                         "lung_ml_mean": g["lung_ml"].mean()})
    return pd.DataFrame(rows)


def discrimination(df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """Cox on train, C on val, same complete-case patients for every model."""
    d = df.dropna(subset=["th_z", "eta_z"] + CLINICAL)
    tr, va = d[d.split == "train"], d[d.split == "val"].reset_index(drop=True)
    models = {"age": ["age"], "clinical": CLINICAL, "thymus": ["th_z"], "clinical+thymus": CLINICAL + ["th_z"],
              "eta": ["eta_z"], "clinical+eta": CLINICAL + ["eta_z"],
              "clinical+eta+thymus": CLINICAL + ["eta_z", "th_z"]}
    t, e = va["time"].to_numpy(float), va["event"].to_numpy(int)
    lp = {}
    for name, cols in models.items():
        # eta was fitted to the train patients, so its train-set coefficient would be inflated:
        # models with eta are fitted on val (as S5 does with --fit-split val) and scored on val
        # with no held-out set, which favours them slightly; the deltas below compare like with like.
        fit_on = va if "eta_z" in cols else tr
        cph = fit_cox(fit_on, cols)
        lp[name] = cph.predict_log_partial_hazard(design(va, cols)).to_numpy()
    boot = {n: np.empty(n_boot) for n in models}
    for b in range(n_boot):
        i = rng.integers(0, len(va), len(va))
        for n in models:
            boot[n][b] = cindex(lp[n][i], t[i], e[i])
    rows = []
    for n in models:
        lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
        rows.append({"model": n, "fit_on": "val" if "eta_z" in models[n] else "train",
                     "val_c": cindex(lp[n], t, e), "c_lo": lo, "c_hi": hi})
    for base, plus in (("clinical", "clinical+thymus"), ("clinical+eta", "clinical+eta+thymus")):
        dd = boot[plus] - boot[base]
        rows.append({"model": f"delta {plus} - {base}", "fit_on": rows[list(models).index(plus)]["fit_on"],
                     "val_c": cindex(lp[plus], t, e) - cindex(lp[base], t, e),
                     "c_lo": np.nanpercentile(dd, 2.5), "c_hi": np.nanpercentile(dd, 97.5),
                     "p_boot_le0": float(np.nanmean(dd <= 0))})
    out = pd.DataFrame(rows)
    out.attrs.update(n_val=len(va), events_val=int(e.sum()), n_train=len(tr), events_train=int(tr["event"].sum()))
    return out


def hazard_ratios(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    d = df.dropna(subset=["th_z"] + CLINICAL)
    for label, cols in (("unadjusted", []), ("+ age", ["age"]), ("+ age, sex, smoking", BASIC),
                        ("+ clinical", CLINICAL)):
        rows.append(hr_row(fit_cox(d, cols + ["th_z"]), "th_z", outcome="lung cancer", scope="train+val",
                           model=label))
    v = d.dropna(subset=["eta_z"])
    v = v[v.split == "val"]
    for label, cols in (("unadjusted", []), ("+ clinical", CLINICAL), ("+ clinical + eta", CLINICAL + ["eta_z"])):
        rows.append(hr_row(fit_cox(v, cols + ["th_z"]), "th_z", outcome="lung cancer", scope="val", model=label))
    base, full = fit_cox(v, CLINICAL + ["eta_z"]), fit_cox(v, CLINICAL + ["eta_z", "th_z"])
    p_lr = float(chi2.sf(2 * (full.log_likelihood_ - base.log_likelihood_), 1))

    # cause-specific: the other histology's cancers are censored at diagnosis
    for group in ("adeno", "non_adeno"):
        g = d.assign(event_cs=((d["event"] == 1) & (d["histo"] == group)).astype(int))
        rows.append(hr_row(fit_cox(g, BASIC + ["th_z"], event="event_cs"), "th_z",
                           outcome=f"lung cancer: {group}", scope="train+val", model="+ age, sex, smoking"))

    if {"death_days", "fup_days"} <= set(d.columns):
        t0 = d["scr_days0"].fillna(0)
        died = d["death_days"].notna()
        g = d.assign(t_death=np.where(died, d["death_days"], d["fup_days"]) - t0, e_death=died.astype(int))
        g = g[g["t_death"] > 0]
        for label, cols in (("unadjusted", []), ("+ age, sex, smoking", BASIC), ("+ clinical", CLINICAL)):
            rows.append(hr_row(fit_cox(g, cols + ["th_z"], time="t_death", event="e_death"), "th_z",
                               outcome="all-cause death", scope="train+val", model=label))
    else:
        print("prsn has no death_days/fup_days: all-cause death skipped", flush=True)
    return pd.DataFrame(rows), p_lr


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["age", "sex", "cigsmok", "lung_ml", "lung_mean_hu", "eta"]:
        d = df[df.split == "val"] if c == "eta" else df
        x = (d[c] == "M").astype(float) if c == "sex" else d[c]
        ok = x.notna() & d["th"].notna()
        r, p = spearmanr(d.loc[ok, "th"], x[ok])
        rows.append({"variable": "male" if c == "sex" else c, "scope": "val" if c == "eta" else "train+val",
                     "spearman_rho": r, "p": p, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--th-csv", default=str(THYMUS_SCORE_CSV), help="published thymic-health scores (Zenodo)")
    ap.add_argument("--ref", default="cosine", help="reference CNN run under runs/ (its pred.csv gives eta)")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="thymus_score", help="output folder under runs/")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.runs_dir) / a.out
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    df = load(a)
    cov = coverage(df)
    print("\n== coverage (train + val only) ==\n" + cov.round(3).to_string(index=False))
    corr = correlations(df)
    print("\n== Spearman with the thymus score ==\n" + corr.round(3).to_string(index=False))
    disc = discrimination(df, a.n_boot, rng)
    print(f"\n== discrimination: C on val ({disc.attrs['n_val']} pts / {disc.attrs['events_val']} events), "
          f"{a.n_boot}x bootstrap ==\n" + disc.round(3).to_string(index=False))
    hr, p_lr = hazard_ratios(df)
    print("\n== HR per SD of the thymus score ==\n" + hr.round(3).to_string(index=False))
    print(f"val LR test, clinical + eta + thymus vs clinical + eta: p = {p_lr:.3g}")

    cov.to_csv(out / "coverage.csv", index=False)
    corr.to_csv(out / "correlations.csv", index=False)
    disc.to_csv(out / "discrimination.csv", index=False)
    hr.to_csv(out / "hazard_ratios.csv", index=False)
    summary = {"args": vars(a), **disc.attrs, "p_lr_val_add_thymus_to_clinical_eta": p_lr,
               "val_c": dict(zip(disc["model"], disc["val_c"]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nwrote {out}/", flush=True)


if __name__ == "__main__":
    main()
