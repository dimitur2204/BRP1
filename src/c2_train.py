#!/usr/bin/env python3
"""C2 -- train the lungs / sternum 3D CNNs (one run = one config, one seed).

A "run set" is a list of run configs; a Slurm array task executes a
contiguous chunk of runs of the same organ, so the organ's cache (~5-8 GB)
is loaded to the GPU once per task, not once per run.

Run sets
  grid   (independent of any selection)
         - new recipe, Cox loss, seed 0: lr x weight_decay x dropout grid
         - legacy recipe AT SCALE (old 64^3 input + old windows + BCE + Adam
           1e-3, batch 8, 60 epochs, best-val-AUC checkpoint), seeds 0-2
         - legacy recipe on the old 400-subset training patients only
           (a retrained reproduction of the old model), seed 0
         - score_old: the saved legacy checkpoint models/{organ}_cnn3d.pt,
           scored (no training) on every cohort patient
  final  (reads data/c3_chosen_configs.json, written by c3_select.py)
         - chosen config, seeds 1-5              (the headline model)
         - same with BCE loss, seeds 1-5         (does the objective matter?)
         - same without augmentation, seeds 1-3
         - learning curve: train on the old subset_v1 train pids (179), 25%,
           50% of train (event-stratified), seeds 1-3 (100% = headline)
         - permuted-label null: train (time, event) shuffled, seed 1
  curves (reads data/c3_chosen_configs.json; for figures only, never evaluated)
         - main, bce, null (seed 1) and legacy_scale (seed 0), re-trained for
           the full epoch budget (early stopping's epoch is recorded, not
           obeyed) with diagnostics and a checkpoint per epoch -> plotted by
           c2_plot_curves.py. No test predictions are written.

Every run writes runs/{run_id}/: config.json, history.json, pred.csv (eta for
EVERY cohort patient, all splits, at the selected epoch), curves.png and
model.pt. Test predictions are written for convenience, but only
c4_evaluate.py reads them, and only for runs chosen without looking at test.

history.json, per epoch (see c2_plot_curves.py for why each is defined so):
  opt_loss / opt_loss_null    online optimisation loss (train-mode BN, aug,
                              batch-local Cox risk sets / weighted BCE) and the
                              same loss for a constant score
  eval_loss_{train,val}       eval-mode, unaugmented, whole split: Cox with
                              whole-split risk sets, or UNweighted BCE
  eval_null_{train,val}       eval loss of a constant score (Cox: eta = 0;
                              BCE: logit at the split's prevalence)
  cindex_train                eval-mode, unaugmented train C (comparable to val)
  cindex_train_online         C of the online batch outputs (not comparable)
  cindex_val, auc_val
  val_eta_q, {train,val}_eta_{mean,sd}, bn_running_*, val_loss_{pos,neg},
  val_loss_top1pct_share      diagnostics (logit shift, BN, extreme patients)
  eval_loss_val_bnbatch, cindex_val_bnbatch, ...  val with BN batch statistics
                              (curves set only)

Usage:
    python src/c2_train.py --set grid --count            # number of array tasks
    python src/c2_train.py --set grid --list             # print the runs
    python src/c2_train.py --set grid --task 0           # what each array task runs
    python src/c2_train.py --set grid --task 0 --device cpu --smoke  # tiny CPU smoke test
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from c2_plot_curves import plot_curves
from cnn3d import CNN3D, cox_ph_loss, harrell_c, load_organ, to_input, CACHE_DIR, RUNS_DIR, PROJECT

ORGANS = ["lungs", "sternum"]
RUNS_PER_TASK = 3
CHOSEN_JSON = PROJECT / "data" / "c3_chosen_configs.json"
LEGACY_CKPT = PROJECT / "models" / "{organ}_cnn3d.pt"

# New-recipe defaults (the grid overrides lr / weight_decay / dropout).
NEW_DEFAULTS = dict(recipe="iso", loss="cox", optimizer="adamw", batch_size=32,
                    max_epochs=80, patience=12, select_metric="cindex",
                    width=16, augment=True, train_subset="all", permute=False)
GRID = dict(lr=[1e-4, 3e-4, 1e-3], weight_decay=[1e-4, 1e-2], dropout=[0.0, 0.3])
# The legacy stage3_cnn3d.py recipe, verbatim.
LEGACY = dict(recipe="legacy64", loss="bce", optimizer="adam", lr=1e-3, weight_decay=1e-4,
              dropout=0.0, batch_size=8, max_epochs=60, patience=None, select_metric="auc",
              width=16, augment=False, train_subset="all", permute=False)


def run_id(c: dict) -> str:
    if c["kind"] == "score_old":
        return f"{c['set']}__{c['organ']}__score_old"
    return (f"{c['set']}__{c['organ']}__{c['tag']}__{c['recipe']}_{c['loss']}"
            f"_lr{c['lr']:g}_wd{c['weight_decay']:g}_do{c['dropout']:g}"
            f"_aug{int(c['augment'])}_tr{c['train_subset']}_perm{int(c['permute'])}_s{c['seed']}")


def build_runs(set_name: str, chosen_json: Path = CHOSEN_JSON) -> list[dict]:
    runs = []
    for organ in ORGANS:
        if set_name == "grid":
            for lr, wd, do in itertools.product(GRID["lr"], GRID["weight_decay"], GRID["dropout"]):
                runs.append({**NEW_DEFAULTS, "organ": organ, "lr": lr, "weight_decay": wd,
                             "dropout": do, "tag": "grid", "seed": 0})
            for s in range(3):
                runs.append({**LEGACY, "organ": organ, "tag": "legacy_scale", "seed": s})
            runs.append({**LEGACY, "organ": organ, "tag": "legacy_subset", "seed": 0,
                         "train_subset": "subset_v1"})
            runs.append({"organ": organ, "tag": "score_old", "kind": "score_old", "seed": 0,
                         "recipe": "legacy64"})
        elif set_name in ("final", "curves"):
            chosen = json.loads(Path(chosen_json).read_text())[organ]
            best = {**NEW_DEFAULTS, **{k: chosen[k] for k in ("lr", "weight_decay", "dropout")},
                    "organ": organ}
        if set_name == "curves":  # the plotted seed of each figure's group, full length + diagnostics
            diag = {"full_length": True, "diag": True}
            runs += [{**best, **diag, "tag": "main", "seed": 1},
                     {**best, **diag, "tag": "bce", "loss": "bce", "seed": 1},
                     {**best, **diag, "tag": "null", "permute": True, "seed": 1},
                     {**LEGACY, **diag, "organ": organ, "tag": "legacy_scale", "seed": 0}]
        elif set_name == "final":
            runs += [{**best, "tag": "main", "seed": s} for s in range(1, 6)]
            runs += [{**best, "tag": "bce", "loss": "bce", "seed": s} for s in range(1, 6)]
            runs += [{**best, "tag": "noaug", "augment": False, "seed": s} for s in range(1, 4)]
            for sub in ("subset_v1", "0.25", "0.5"):
                runs += [{**best, "tag": "lc", "train_subset": sub, "seed": s} for s in range(1, 4)]
            runs.append({**best, "tag": "null", "permute": True, "seed": 1})
        elif set_name != "grid":
            raise ValueError(set_name)
    for r in runs:
        r.setdefault("kind", "train")
        r["set"] = set_name
        r["run_id"] = run_id(r)
    return runs


def chunk_runs(runs: list[dict]) -> list[list[dict]]:
    """Group runs into array tasks of <= RUNS_PER_TASK runs, never mixing organs."""
    tasks = []
    for organ in ORGANS:
        rs = [r for r in runs if r["organ"] == organ]
        tasks += [rs[i:i + RUNS_PER_TASK] for i in range(0, len(rs), RUNS_PER_TASK)]
    return tasks


# ── data ─────────────────────────────────────────────────────────────────────

class OrganData:
    """One organ + recipe, loaded once and moved to `device`."""

    def __init__(self, organ: str, recipe: str, device: torch.device, cache_dir: Path,
                 smoke: bool = False):
        idx, arr = load_organ(organ, recipe, cache_dir, mmap=True)
        if smoke:  # tiny balanced subset for a CPU smoke test
            keep = np.concatenate([np.where(idx["split"] == s)[0][:24] for s in ("train", "val", "test")])
            idx, arr = idx.iloc[keep].reset_index(drop=True), arr[keep]
        t0 = time.time()
        self.x = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        print(f"[{organ}/{recipe}] loaded {tuple(self.x.shape)} {self.x.dtype} "
              f"to {device} in {time.time()-t0:.0f}s", flush=True)
        self.idx = idx
        self.organ, self.recipe = organ, recipe
        self.time = torch.tensor(idx["time"].to_numpy(np.float32), device=device)
        self.event = torch.tensor(idx["event"].to_numpy(np.float32), device=device)
        self.split = idx["split"].to_numpy()

    def rows(self, split: str) -> np.ndarray:
        return np.where(self.split == split)[0]


def train_rows(data: OrganData, subset: str, seed: int) -> np.ndarray:
    tr = data.rows("train")
    if subset == "all":
        return tr
    if subset == "subset_v1":
        return tr[data.idx["in_subset_v1"].to_numpy()[tr] == 1]
    frac = float(subset)  # event-stratified random fraction of train
    rng = np.random.default_rng(1000 + seed)
    ev = data.idx["event"].to_numpy()[tr]
    keep = [rng.choice(tr[ev == e], size=int(round(frac * (ev == e).sum())), replace=False)
            for e in (0, 1)]
    return np.sort(np.concatenate(keep))


def augment_batch(x: torch.Tensor, gen: torch.Generator):
    """Random +-3 voxel shift per sample (torch.roll) + HU jitter params."""
    b = x.shape[0]
    shifts = torch.randint(-3, 4, (b, 3), generator=gen)
    x = torch.stack([torch.roll(x[i], tuple(shifts[i].tolist()), dims=(0, 1, 2)) for i in range(b)])
    jitter = torch.stack([torch.randn(b, generator=gen) * 10.0,          # HU offset, sd 10 HU
                          1.0 + torch.randn(b, generator=gen) * 0.02], 1)  # HU scale, sd 2%
    return x, jitter.to(x.device)


# ── training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, data: OrganData, rows: np.ndarray, device, bs: int = 32,
            bn_batch_stats: bool = False) -> np.ndarray:
    """eta for `rows`, eval mode, no augmentation. bn_batch_stats: diagnostic
    only -- a throwaway copy whose BatchNorm layers normalise with each batch's
    own statistics (as in training) instead of the running averages."""
    model.eval()
    if bn_batch_stats:
        model = copy.deepcopy(model)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.train()
    out = []
    for i in range(0, len(rows), bs):
        r = torch.as_tensor(rows[i:i + bs], device=data.x.device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(to_input(data.x[r], data.organ, data.recipe)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def metrics(data: OrganData, rows: np.ndarray, eta: np.ndarray) -> dict:
    t = data.idx["time"].to_numpy()[rows]
    e = data.idx["event"].to_numpy()[rows]
    return {"cindex": harrell_c(t, eta, e),
            "auc": float(roc_auc_score(e, eta)) if 0 < e.sum() < len(e) else float("nan")}


def write_predictions(model, data, device, run_dir: Path,
                      splits=("train", "val", "test")) -> pd.DataFrame:
    rows = np.where(np.isin(data.split, splits))[0]
    pred = data.idx.iloc[rows][["pid", "split", "time", "event"]].copy()
    pred["eta"] = predict(model, data, rows, device)
    pred.to_csv(run_dir / "pred.csv", index=False)
    return pred


def bce_eval(eta: torch.Tensor, e: torch.Tensor) -> dict:
    """Unweighted BCE on a whole split, its no-information baseline (the
    binary entropy of the split's prevalence) and where the loss comes from."""
    per = nn.functional.binary_cross_entropy_with_logits(eta, e, reduction="none")
    p = float(e.mean())
    top = torch.topk(per, max(1, len(per) // 100)).values.sum()
    return {"loss": float(per.mean()), "null": -sum(x * np.log(x) for x in (p, 1 - p) if x > 0),
            "pos": float((per * e).sum() / len(per)), "neg": float((per * (1 - e)).sum() / len(per)),
            "top1pct_share": float(top / per.sum())}


def eval_split(eta_np: np.ndarray, t: torch.Tensor, e: torch.Tensor, loss: str) -> dict:
    """Evaluation loss with the SAME definition for every split (whole-split
    Cox risk sets, or unweighted BCE) and its constant-score baseline."""
    eta = torch.as_tensor(eta_np, dtype=torch.float32)
    t, e = t.cpu().float(), e.cpu().float()
    if loss == "cox":
        return {"loss": float(cox_ph_loss(eta, t, e)), "null": float(cox_ph_loss(torch.zeros_like(eta), t, e))}
    return bce_eval(eta, e)


def bn_stats(model) -> tuple[list, list]:
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
    return ([float(m.running_mean.abs().mean()) for m in bns],
            [float(m.running_var.mean()) for m in bns])


def score_old(cfg: dict, data: OrganData, device, run_dir: Path) -> dict:
    model = CNN3D(in_ch=1, width=16, dropout=0.0).to(device)
    model.load_state_dict(torch.load(str(LEGACY_CKPT).format(organ=cfg["organ"]), map_location=device))
    pred = write_predictions(model, data, device, run_dir)
    res = {s: metrics(data, data.rows(s), pred["eta"].to_numpy()[data.rows(s)])
           for s in ("train", "val", "test")}
    return {"scored": res}


def train_one(cfg: dict, data: OrganData, device, run_dir: Path, smoke: bool) -> dict:
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    in_ch = 1 if cfg["recipe"] == "legacy64" else 2
    model = CNN3D(in_ch=in_ch, width=cfg["width"], dropout=cfg["dropout"]).to(device)
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    tr = train_rows(data, cfg["train_subset"], seed)
    va = data.rows("val")
    t_tr, e_tr = data.time[tr], data.event[tr]
    if cfg["permute"]:  # null control: break the image <-> outcome link in train only
        perm = torch.as_tensor(np.random.default_rng(seed).permutation(len(tr)), device=t_tr.device)
        t_tr, e_tr = t_tr[perm], e_tr[perm]
    n_pos = float(e_tr.sum())
    pos_weight = torch.tensor((len(tr) - n_pos) / max(n_pos, 1.0), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(eta, t, e):
        return cox_ph_loss(eta, t, e) if cfg["loss"] == "cox" else bce(eta.float(), e)

    print(f"  {cfg['run_id']}: train n={len(tr)} ({int(n_pos)} events), val n={len(va)}", flush=True)
    max_epochs = 2 if smoke else cfg["max_epochs"]
    hist: dict = {}
    best, best_epoch, best_state, since_best, stop_epoch = -np.inf, -1, None, 0, None
    bs = cfg["batch_size"]
    diag = cfg.get("diag", False)
    if diag:
        (run_dir / "epochs").mkdir(exist_ok=True)
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(tr), generator=gen).numpy()
        losses, nulls, etas, pos = [], [], [], []
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            if len(b) < 2:
                continue
            x = data.x[torch.as_tensor(tr[b], device=data.x.device)]
            jitter = None
            if cfg["augment"]:
                x, jitter = augment_batch(x, gen)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                eta = model(to_input(x, data.organ, data.recipe, jitter))
            bt = torch.as_tensor(b, device=t_tr.device)
            loss = loss_fn(eta, t_tr[bt], e_tr[bt])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            with torch.no_grad():
                nulls.append(loss_fn(torch.zeros_like(eta), t_tr[bt], e_tr[bt]).item())
            etas.append(eta.detach().float().cpu().numpy())
            pos.append(b)
        # online train C from the (augmented, train-mode BN, changing-weights)
        # batch outputs -- kept for reference, NOT comparable to val C
        pos = np.concatenate(pos)
        tr_c_online = harrell_c(t_tr.cpu().numpy()[pos], np.concatenate(etas), e_tr.cpu().numpy()[pos])
        # end-of-epoch evaluation: frozen model, eval mode, no augmentation,
        # same loss definition on train and val (train labels as trained on,
        # i.e. permuted for the null run)
        eta_tr = predict(model, data, tr, device)
        eta_va = predict(model, data, va, device)
        ev_tr = eval_split(eta_tr, t_tr, e_tr, cfg["loss"])
        ev_va = eval_split(eta_va, data.time[va], data.event[va], cfg["loss"])
        m = metrics(data, va, eta_va)
        bn_m, bn_v = bn_stats(model)
        rec = {"opt_loss": float(np.mean(losses)), "opt_loss_null": float(np.mean(nulls)),
               "eval_loss_train": ev_tr["loss"], "eval_null_train": ev_tr["null"],
               "eval_loss_val": ev_va["loss"], "eval_null_val": ev_va["null"],
               "cindex_train": harrell_c(t_tr.cpu().numpy(), eta_tr, e_tr.cpu().numpy()),
               "cindex_train_online": tr_c_online, "cindex_val": m["cindex"], "auc_val": m["auc"],
               "val_eta_q": np.percentile(eta_va, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist(),
               "val_eta_mean": float(eta_va.mean()), "val_eta_sd": float(eta_va.std()),
               "train_eta_mean": float(eta_tr.mean()), "train_eta_sd": float(eta_tr.std()),
               "bn_running_mean_absmean": bn_m, "bn_running_var_mean": bn_v}
        if cfg["loss"] == "bce":
            rec.update({"val_loss_pos": ev_va["pos"], "val_loss_neg": ev_va["neg"],
                        "val_loss_top1pct_share": ev_va["top1pct_share"]})
        if diag:
            eta_vb = predict(model, data, va, device, bn_batch_stats=True)
            ev_vb = eval_split(eta_vb, data.time[va], data.event[va], cfg["loss"])
            rec.update({"eval_loss_val_bnbatch": ev_vb["loss"], "cindex_val_bnbatch": metrics(data, va, eta_vb)["cindex"],
                        "val_eta_mean_bnbatch": float(eta_vb.mean()), "val_eta_sd_bnbatch": float(eta_vb.std())})
            torch.save(model.state_dict(), run_dir / "epochs" / f"ep{epoch:03d}.pt")
        for k, v in rec.items():
            hist.setdefault(k, []).append(v)
        if stop_epoch is None:  # selection only inside the early-stopping window
            score = m[cfg["select_metric"]]
            if score > best or best_state is None:  # NaN score (no val events, smoke) keeps epoch 0
                best, best_epoch, since_best = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
            if cfg["patience"] is not None and since_best >= cfg["patience"]:
                stop_epoch = epoch
        print(f"    ep {epoch:3d} opt {rec['opt_loss']:.4f} evalTr {ev_tr['loss'] - ev_tr['null']:+.4f} "
              f"evalVa {ev_va['loss'] - ev_va['null']:+.4f} trC {rec['cindex_train']:.3f} "
              f"vaC {m['cindex']:.3f} vaAUC {m['auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if stop_epoch is not None and not cfg.get("full_length", False):
            break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "model.pt")
    # the curves set is for figures only: never write its test predictions
    pred = write_predictions(model, data, device, run_dir,
                             ("train", "val") if cfg["set"] == "curves" else ("train", "val", "test"))
    va_eta = pred.loc[pred["split"] == "val", "eta"].to_numpy()
    json.dump(hist, open(run_dir / "history.json", "w"))
    res = {"best_epoch": best_epoch, "stop_epoch": stop_epoch, "epochs_run": len(hist["cindex_val"]),
           "val": metrics(data, va, va_eta), "train_n": int(len(tr)), "train_events": int(n_pos),
           "pos_weight": float(pos_weight), "seconds": round(time.time() - t0, 1)}
    plot_curves(hist, cfg, res, run_dir / "curves.png")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=["grid", "final", "curves"])
    p.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    p.add_argument("--count", action="store_true", help="print number of array tasks and exit")
    p.add_argument("--list", action="store_true", help="print task -> runs and exit")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--chosen", default=str(CHOSEN_JSON), help="c3 output (final set only)")
    p.add_argument("--smoke", action="store_true", help="24 pids/split, 2 epochs (CPU check)")
    args = p.parse_args()

    tasks = chunk_runs(build_runs(args.set, args.chosen))
    if args.count:
        print(len(tasks))
        return
    if args.list:
        for i, t in enumerate(tasks):
            for r in t:
                print(i, r["run_id"])
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.backends.cudnn.benchmark = True
    print(f"device {device}; task {args.task}/{len(tasks)}", flush=True)

    loaded: dict = {}
    for cfg in tasks[args.task]:
        run_dir = Path(args.runs_dir) / args.set / cfg["run_id"]
        if (run_dir / "result.json").exists():
            print(f"  skip (done): {cfg['run_id']}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        key = (cfg["organ"], cfg["recipe"])
        if key not in loaded:
            loaded.clear()  # free the previous organ/recipe before loading the next
            torch.cuda.empty_cache() if device.type == "cuda" else None
            loaded[key] = OrganData(*key, device, Path(args.cache_dir), args.smoke)
        json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)
        fn = score_old if cfg["kind"] == "score_old" else (
            lambda c, d, dev, rd: train_one(c, d, dev, rd, args.smoke))
        res = fn(cfg, loaded[key], device, run_dir)
        json.dump({**cfg, **res}, open(run_dir / "result.json", "w"), indent=1)
        print(f"  done {cfg['run_id']}: {json.dumps(res)[:300]}", flush=True)


if __name__ == "__main__":
    main()
