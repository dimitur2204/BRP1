#!/usr/bin/env python3
"""R4 -- survival models on the harmonized heart features; writes held-out
test risk scores for R5 to evaluate.

Protocol (the test split is touched exactly once, at the very end):
  1. tune  -- fit each model on TRAIN, pick hyperparameters by Harrell's C on
              VAL (val has only 154 events, so grids are kept small).
  2. refit -- refit the chosen config on TRAIN+VAL. The mentor's enrichment
              makes train (33% events) and val (~8%) differ in baseline
              hazard, handled per model family:
                Cox models  stratified by split (separate baseline hazards)
                DeepSurv    Cox loss computed within each split, summed
                RSF         split indicator as an input (0 for val/test rows,
                            so it is constant on test and can't reorder it)
  3. predict on TEST -> data/r4_test_predictions.csv (one risk column per model).

Models (clinical = age, male, current smoker; heart = the 57 R3 features):
  clinical            Cox, clinical only -- the bar to beat
  clinical_calcium    Cox, clinical + card_calc_mass_proxy (pre-specified from
                      R3 discovery -- chosen without looking at test)
  coxen_heart         elastic-net Cox, heart features only
  coxen_all           elastic-net Cox, heart + clinical
  rsf_all             Random Survival Forest, heart + clinical
  deepsurv_heart      DeepSurv MLP, heart features only
  deepsurv_all        DeepSurv MLP, heart + clinical
  deepsurv_null       deepsurv_all config trained on PERMUTED survival labels
                      (negative control: should give test C ~ 0.5)
DeepSurv = small MLP with the Cox partial-likelihood loss (Katzman et al.
2018), plain torch, full-batch Adam; an ensemble of N_SEEDS seeds is averaged
to reduce run-to-run noise.

Runs on CPU in a few minutes. Usage:
    env/.venv/bin/python src/r4_models.py
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
DATA_CSV = PROJECT / "data" / "radiomics" / "heart_features_v1_r3.csv"
SELECTED_CSV = PROJECT / "data" / "radiomics" / "r3_selected_features.csv"
OUT_PRED = PROJECT / "data" / "r4_test_predictions.csv"
OUT_TUNE = PROJECT / "data" / "r4_val_tuning.csv"
OUT_CONFIG = PROJECT / "data" / "r4_chosen_configs.json"

CLIN = ["age", "male", "cigsmok"]
CALCIUM = "card_calc_mass_proxy"
SEED = 0
N_SEEDS = 5
MAX_EPOCHS, PATIENCE = 400, 40
torch.set_num_threads(8)


def cindex(time, event, risk) -> float:
    return concordance_index(time, -np.asarray(risk), event)


# ── Cox (lifelines) ──────────────────────────────────────────────────────────
def fit_cox(d: pd.DataFrame, cols, penalizer=0.0, l1_ratio=0.0, strata=False) -> CoxPHFitter:
    use = [*cols, "time_yr", "event"] + (["split"] if strata else [])
    return CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio).fit(
        d[use], "time_yr", "event", strata=["split"] if strata else None)


def cox_risk(cph: CoxPHFitter, d: pd.DataFrame, cols) -> np.ndarray:
    # linear predictor; stratification only changes the baseline hazard, not the ranking
    return (d[cols].values @ cph.params_[cols].values)


# ── RSF (scikit-survival) ────────────────────────────────────────────────────
def fit_rsf(X, t, e, leaf, seed=SEED) -> RandomSurvivalForest:
    return RandomSurvivalForest(n_estimators=400, min_samples_leaf=leaf, max_features="sqrt",
                                n_jobs=8, random_state=seed).fit(X, Surv.from_arrays(e.astype(bool), t))


# ── DeepSurv (torch) ─────────────────────────────────────────────────────────
class DeepSurv(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_loss(risk: torch.Tensor, t: torch.Tensor, e: torch.Tensor, strata: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), summed over strata,
    divided by the total number of events."""
    total = risk.new_zeros(())
    for s in torch.unique(strata):
        m = strata == s
        r, tt, ee = risk[m], t[m], e[m]
        order = torch.argsort(tt, descending=True)        # latest times first
        r, ee = r[order], ee[order]
        log_risk_set = torch.logcumsumexp(r, dim=0)       # log sum exp(risk) over subjects with time >= t_i
        total = total - ((r - log_risk_set) * ee).sum()
    return total / e.sum().clamp(min=1)


def train_deepsurv(Xtr, ttr, etr, str_, cfg, seed, epochs=None, Xval=None, tval=None, eval_=None):
    """Full-batch training. With validation data: early-stop on val C and
    return (model, best_epoch, best_c). Without: train exactly `epochs`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepSurv(Xtr.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    X, t, e, s = (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ttr, dtype=torch.float32),
                  torch.tensor(etr, dtype=torch.float32), torch.tensor(str_))
    best = (-1.0, 0, None)
    n_ep = epochs or MAX_EPOCHS
    for ep in range(1, n_ep + 1):
        model.train()
        opt.zero_grad()
        loss = cox_loss(model(X), t, e, s)
        loss.backward()
        opt.step()
        if Xval is not None:
            c = cindex(tval, eval_, predict_deepsurv([model], Xval))
            if c > best[0]:
                best = (c, ep, {k: v.clone() for k, v in model.state_dict().items()})
            elif ep - best[1] >= PATIENCE:
                break
    if Xval is not None:
        model.load_state_dict(best[2])
        return model, best[1], best[0]
    return model


def predict_deepsurv(models, X) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outs = []
        for m in models:
            m.eval()
            outs.append(m(Xt).numpy())
    return np.mean(outs, axis=0)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    df = pd.read_csv(DATA_CSV, dtype={"pid": str})
    heart = pd.read_csv(SELECTED_CSV)["feature"].tolist()
    # clinical inputs standardized on train (features are already per-SD from R3)
    tr_mask = df["split"] == "train"
    df["age"] = (df["age"] - df.loc[tr_mask, "age"].mean()) / df.loc[tr_mask, "age"].std()
    ALL = heart + CLIN

    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    trva = df[df.split != "test"]
    print(f"train {len(tr)}/{int(tr.event.sum())}  val {len(va)}/{int(va.event.sum())}  "
          f"test {len(te)}/{int(te.event.sum())}  | heart features {len(heart)}")

    tuning, configs, preds = [], {}, {"pid": te.pid.values, "time_yr": te.time_yr.values,
                                      "event": te.event.values}

    def log(model, cfg, c):
        tuning.append({"model": model, **{k: str(v) for k, v in cfg.items()}, "val_c": c})

    # ── fixed Cox models ──
    for name, cols in (("clinical", CLIN), ("clinical_calcium", [*CLIN, CALCIUM])):
        c = cindex(va.time_yr, va.event, cox_risk(fit_cox(tr, cols), va, cols))
        log(name, {}, c)
        cph = fit_cox(trva, cols, strata=True)
        preds[name] = cox_risk(cph, te, cols)
        configs[name] = {"cols": cols, "coef": cph.params_.round(4).to_dict()}
        print(f"{name:18s} val C={c:.3f}")

    # ── elastic-net Cox ──
    for name, cols in (("coxen_heart", heart), ("coxen_all", ALL)):
        best = None
        for pen, l1 in itertools.product([0.01, 0.03, 0.1, 0.3], [0.0, 0.5, 1.0]):
            try:
                c = cindex(va.time_yr, va.event, cox_risk(fit_cox(tr, cols, pen, l1), va, cols))
            except Exception:  # lifelines convergence failure on a grid point
                c = np.nan
            log(name, {"penalizer": pen, "l1_ratio": l1}, c)
            if not np.isnan(c) and (best is None or c > best[0]):
                best = (c, pen, l1)
        cph = fit_cox(trva, cols, best[1], best[2], strata=True)
        preds[name] = cox_risk(cph, te, cols)
        nz = int((cph.params_.abs() > 1e-3).sum())
        configs[name] = {"penalizer": best[1], "l1_ratio": best[2], "val_c": best[0], "nonzero_coefs": nz,
                         "top_coefs": cph.params_[cph.params_.abs().sort_values(ascending=False).index[:10]]
                         .round(4).to_dict()}
        print(f"{name:18s} val C={best[0]:.3f}  (pen={best[1]}, l1={best[2]}, {nz} nonzero)")

    # ── RSF ──
    best = None
    for leaf in (15, 30, 60, 120):
        rsf = fit_rsf(tr[ALL].values, tr.time_yr.values, tr.event.values, leaf)
        c = cindex(va.time_yr, va.event, rsf.predict(va[ALL].values))
        log("rsf_all", {"min_samples_leaf": leaf}, c)
        if best is None or c > best[0]:
            best = (c, leaf)
    ind = (trva.split == "train").astype(float).values[:, None]
    rsf = fit_rsf(np.hstack([trva[ALL].values, ind]), trva.time_yr.values, trva.event.values, best[1])
    preds["rsf_all"] = rsf.predict(np.hstack([te[ALL].values, np.zeros((len(te), 1))]))
    configs["rsf_all"] = {"min_samples_leaf": best[1], "val_c": best[0], "n_estimators": 400}
    print(f"{'rsf_all':18s} val C={best[0]:.3f}  (leaf={best[1]})")

    # ── DeepSurv ──
    grid = [dict(hidden=h, dropout=dr, weight_decay=wd, lr=1e-3)
            for h, dr, wd in itertools.product([16, 64], [0.2, 0.5], [1e-4, 1e-2])]
    strata_trva = (trva.split == "train").astype(int).values
    for name, cols in (("deepsurv_heart", heart), ("deepsurv_all", ALL)):
        best = None
        for cfg in grid:
            _, ep, c = train_deepsurv(tr[cols].values, tr.time_yr.values, tr.event.values,
                                      np.zeros(len(tr), int), cfg, SEED,
                                      Xval=va[cols].values, tval=va.time_yr.values, eval_=va.event.values)
            log(name, {**cfg, "best_epoch": ep}, c)
            if best is None or c > best[0]:
                best = (c, cfg, ep)
        c, cfg, ep = best
        models = [train_deepsurv(trva[cols].values, trva.time_yr.values, trva.event.values,
                                 strata_trva, cfg, SEED + k, epochs=ep) for k in range(N_SEEDS)]
        preds[name] = predict_deepsurv(models, te[cols].values)
        configs[name] = {**cfg, "epochs": ep, "val_c": c, "n_seeds": N_SEEDS}
        print(f"{name:18s} val C={c:.3f}  ({cfg}, {ep} epochs)")
        if name == "deepsurv_all":
            # negative control: same config, survival labels permuted within train+val
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(trva))
            null = [train_deepsurv(trva[cols].values, trva.time_yr.values[perm], trva.event.values[perm],
                                   strata_trva[perm], cfg, SEED + k, epochs=ep) for k in range(N_SEEDS)]
            preds["deepsurv_null"] = predict_deepsurv(null, te[cols].values)
            configs["deepsurv_null"] = {**configs[name], "labels": "permuted"}

    pd.DataFrame(tuning).to_csv(OUT_TUNE, index=False)
    pd.DataFrame(preds).to_csv(OUT_PRED, index=False)
    OUT_CONFIG.write_text(json.dumps(configs, indent=2, default=float))
    print(f"\nwrote {OUT_PRED}, {OUT_TUNE}, {OUT_CONFIG}")
    print("(test metrics are computed in R5 -- src/r5_evaluate.py)")


if __name__ == "__main__":
    main()
