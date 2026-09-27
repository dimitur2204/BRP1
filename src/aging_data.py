"""Shared analysis frame + stats helpers for the thymus x heart aging experiments.

Builds one row per patient: R3-QC'd heart cohort (6,205) + raw cardiac features
+ published NLST thymic health (Bernatz et al., Nature 2026) + baseline (yr0)
radiologist reads from ctab (emphysema = 59, significant CV abnormality = 60).
Definitions are fixed by experiments/E1_thymus_heart_joint/protocol.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

PROJECT = Path(__file__).resolve().parents[1]
R3_CSV = PROJECT / "data" / "radiomics" / "heart_features_v1_r3.csv"
RAW_CSV = PROJECT / "data" / "radiomics" / "heart_features_v1.csv"
TH_CSV = Path("/faststorage/project/aura_thymus/discovery_pipeline/NLST_thymic_health_scores_published.csv")
CTAB_CSV = PROJECT / "data" / "external" / "nlst_780" / "nlst_780" / "nlst_780_ctab_idc_20210527.csv"

CLIN = ["age", "male", "cigsmok"]
TH_CAT = {0: "low", 1: "average", 2: "high"}

# dataviz reference palette (light surface); same tokens as r5_evaluate.py
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8883"


def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def load_frame(require_th: bool = True) -> tuple[pd.DataFrame, dict]:
    """Analysis frame + a small attrition dict (n / events before/after TH join)."""
    r3 = pd.read_csv(R3_CSV)
    keep = ["pid", "split", "time_yr", "event", "age", "male", "sex", "race", "cigsmok",
            "manufacturer", "kernel", "kernel_group", "card_core_noise_sd_hu", "card_calc_mass_proxy"]
    df = r3[keep].rename(columns={"card_calc_mass_proxy": "calc_combat"})
    raw = pd.read_csv(RAW_CSV, usecols=["pid", "card_calc_mass_proxy", "card_calc_volume_ml",
                                        "card_fat_volume_ml", "card_fat_mean_hu", "card_heart_volume_ml"])
    df = df.merge(raw, on="pid", how="left", validate="1:1")

    th = pd.read_csv(TH_CSV).rename(columns={"ID": "pid", "Thymic_health_continuous": "th_pct",
                                             "Thymic_health_categories": "th_cat"})
    df = df.merge(th, on="pid", how="left", validate="1:1")

    ctab = pd.read_csv(CTAB_CSV, usecols=["pid", "study_yr", "sct_ab_desc"])
    c0 = ctab[ctab["study_yr"] == 0]
    df["emph0"] = df["pid"].isin(c0.loc[c0["sct_ab_desc"] == 59, "pid"]).astype(int)
    df["cv0"] = df["pid"].isin(c0.loc[c0["sct_ab_desc"] == 60, "pid"]).astype(int)

    att = {"cohort_n": len(df), "cohort_events": int(df["event"].sum()),
           "th_missing_n": int(df["th_pct"].isna().sum()),
           "th_missing_events": int(df.loc[df["th_pct"].isna(), "event"].sum())}
    if require_th:
        df = df[df["th_pct"].notna()].reset_index(drop=True)
        df["th_cat"] = df["th_cat"].astype(int)
    att.update({"analysis_n": len(df), "analysis_events": int(df["event"].sum())})

    # pre-specified marker scalings (z-scored on the analysis cohort)
    df["log_calc"] = np.log1p(df["card_calc_mass_proxy"])
    df["calc_z"] = zscore(df["log_calc"])
    df["calc_combat_z"] = zscore(df["calc_combat"])
    df["fat_z"] = zscore(df["card_fat_volume_ml"])
    df["th_z"] = zscore(df["th_pct"])
    df["th_avg"] = (df["th_cat"] == 1).astype(int)
    df["th_high"] = (df["th_cat"] == 2).astype(int)
    return df, att


# ── stats helpers ────────────────────────────────────────────────────────────
def fit_cox(d: pd.DataFrame, cols: list[str], strata: list[str] | None = ("split",)) -> CoxPHFitter:
    strata = list(strata) if strata else None
    use = list(cols) + ["time_yr", "event"] + (strata or [])
    return CoxPHFitter().fit(d[use], "time_yr", "event", strata=strata)


def hr_row(cph: CoxPHFitter, var: str) -> dict:
    s = cph.summary.loc[var]
    return {"var": var, "coef": s["coef"], "se": s["se(coef)"], "hr": s["exp(coef)"],
            "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"], "p": s["p"]}


def harrell(t, e, risk) -> float:
    return concordance_index(t, -np.asarray(risk), e)


def paired_bootstrap_c(test: pd.DataFrame, risks: dict[str, np.ndarray], ref: str,
                       n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Harrell C per model with 95% bootstrap CI, and dC vs `ref` (paired resamples)."""
    rng = np.random.default_rng(seed)
    t, e = test["time_yr"].to_numpy(), test["event"].to_numpy()
    names = list(risks)
    point = {m: harrell(t, e, risks[m]) for m in names}
    boots = {m: [] for m in names}
    n = len(test)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if e[i].sum() < 2:
            continue
        for m in names:
            boots[m].append(harrell(t[i], e[i], risks[m][i]))
    rows = []
    for m in names:
        b = np.array(boots[m])
        d = b - np.array(boots[ref])
        rows.append({"model": m, "c": point[m], "c_lo": np.percentile(b, 2.5), "c_hi": np.percentile(b, 97.5),
                     "dc_vs_" + ref: point[m] - point[ref],
                     "dc_lo": np.percentile(d, 2.5), "dc_hi": np.percentile(d, 97.5),
                     "dc_p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if m != ref else np.nan})
    return pd.DataFrame(rows)


def logit(d: pd.DataFrame, y: str, cols: list[str], n_iter: int = 50) -> pd.DataFrame:
    """Unpenalized logistic regression by IRLS (statsmodels isn't in the venv): coef, se, Wald p."""
    from scipy.stats import norm
    X = np.column_stack([np.ones(len(d)), d[cols].to_numpy(float)])
    yv = d[y].to_numpy(float)
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (yv - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return pd.DataFrame({"coef": b, "se": se, "p": 2 * norm.sf(np.abs(b / se))}, index=["const"] + list(cols))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
