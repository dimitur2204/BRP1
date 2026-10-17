#!/usr/bin/env python3
"""S6 -- where in the lungs does a trained run look? Saliency maps on val patients.

Two methods, both applied to the saved model (nothing is retrained):
  occlusion  slide a cube (--patch voxels, default 16 = 40 mm) through the
             volume with --stride (default 8 = 20 mm). At each position the lung
             voxels inside the cube are replaced by the patient's median lung
             intensity (--fill median), i.e. "what if this region looked like
             the rest of this lung?". Saliency of a voxel = mean over the cubes
             covering it of (eta - eta_occluded): > 0 means the region pushed
             the risk up. The cube is NOT set to 0 ("outside lung") by default,
             because that would shrink the lung, and eta tracks lung volume, so
             every region would look important.
  Grad-CAM   gradients of eta w.r.t. the feature maps of model.features[--gradcam-layer]
             (default 4: 64 channels at 18x16x16, ~2 cm cells; the last layer is
             only 5x4x4), channel-weighted, ReLU, upsampled to the input grid.
Both maps are kept only inside the network's lung mask (input > 0).

Which patients (val split; test is refused):
  - every val event, and the same number of random val controls (--n-controls)
  - figure groups (--top-k each): highest-eta events, lowest-eta events
    ("missed"), highest-eta controls ("false alarms")

Lobe analysis (does the map point at the lobe where the cancer was found?):
  Lobe labels are rebuilt in the cache grid with s2's exact resampling of the
  TotalSegmentator mask (nearest neighbour; the 5 mm rim of the dilated mask
  gets the nearest lobe). Cancer lobe(s) from prsn: loclup/loclin -> LUL,
  locllow -> LLL, locrup -> RUL, locrmid -> RML, locrlow -> RLL. Events with
  only hilar/mediastinal/unknown locations are left out.
  Per patient and lobe: enrichment = (lobe's share of positive saliency) /
  (lobe's share of lung volume); 1 = no preference. For each event, its cancer
  lobe(s) are compared with the SAME lobe(s) in (a) val controls and (b) events
  whose cancer was elsewhere. This matters: cancers are mostly upper-lobe, and
  a model that looks at the upper lobes in everyone would beat a naive 1-in-5
  chance level without ever finding a tumour.

Outputs, runs/{run}/saliency_{split}/:
  patients.csv        pid, group, eta, time, event, cancer lobes, per-lobe
                      enrichment for both methods
  lobe_summary.csv    per method and follow-up stratum (all / dx <= 1 y / > 1 y):
                      median enrichment of the cancer lobe in events vs the
                      same lobe in controls and in other-lobe events, top-lobe
                      hit rate, Mann-Whitney p
  fig_{group}.png     CT + occlusion + Grad-CAM, slices through the occlusion peak
  mean_maps.png       mean occlusion map of events vs controls (lungs are centred
                      on their bounding box at a fixed mm scale, so this is a rough
                      population map)
  maps/{pid}.npz      occlusion + Grad-CAM (float16) for the figure patients

Usage:
    sbatch -p gpu-short -t 02:00:00 src/submit_s6_saliency.sh --run epb4_s0
Smoke test (CPU, synthetic data):
    env/.venv/bin/python src/s6_saliency.py --run X --device cpu --limit 2 --top-k 1 \\
        --cohort ... --cache ... --cache-index ... --seg-dir ... --prsn ... --runs-dir ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import mannwhitneyu

from model import LungCNN, to_input
from nlst import (CACHE_INDEX, CACHE_NPY, COHORT_CSV, GRID, LUNG_LABELS, PRSN_CSV, RUNS_DIR,
                  SEG_DIR, SPACING_MM)

# TotalSegmentator v2 "total" lung labels (checked against the anatomy in check_orientation)
LOBES = {10: "LUL", 11: "LLL", 12: "RUL", 13: "RML", 14: "RLL"}
CANCER_LOBE_COLS = {"loclup": 10, "loclin": 10, "locllow": 11, "locrup": 12, "locrmid": 13, "locrlow": 14}
METHODS = ("occ", "cam")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val",
                    help="test is deliberately not allowed (test discipline)")
    ap.add_argument("--patch", type=int, default=16, help="occlusion cube edge [voxels of 2.5 mm]")
    ap.add_argument("--stride", type=int, default=8, help="occlusion step [voxels]")
    ap.add_argument("--fill", choices=["median", "zero"], default="median",
                    help="occluded lung voxels: patient's median lung value, or 0 (= outside lung)")
    ap.add_argument("--gradcam-layer", type=int, default=4, help="index into model.features")
    ap.add_argument("--n-controls", type=int, default=-1,
                    help="random val controls for the lobe reference; -1 = as many as events")
    ap.add_argument("--top-k", type=int, default=4, help="patients per figure group")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of events (smoke test)")
    ap.add_argument("--pids", help="comma-separated pids: only these, figures only (no lobe stats)")
    ap.add_argument("--batch", type=int, default=16, help="occluded volumes per forward pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cohort", default=str(COHORT_CSV))
    ap.add_argument("--cache", default=str(CACHE_NPY))
    ap.add_argument("--cache-index", default=str(CACHE_INDEX))
    ap.add_argument("--seg-dir", default=str(SEG_DIR))
    ap.add_argument("--prsn", default=str(PRSN_CSV))
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    return ap.parse_args()


def load_model(path: Path, device) -> LungCNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LungCNN(dropout=ckpt["config"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded {path}: epoch {ckpt['epoch']}, val C {ckpt['val_c']:.4f}", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, x_u8: np.ndarray, device, batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(x_u8), batch):
        out.append(model(to_input(torch.from_numpy(np.ascontiguousarray(x_u8[i:i + batch])).to(device))).cpu())
    return torch.cat(out).numpy()


def starts(n: int, patch: int, stride: int) -> list[int]:
    s = list(range(0, n - patch + 1, stride))
    if s[-1] + patch < n:
        s.append(n - patch)  # cover the far edge too
    return s


@torch.no_grad()
def occlusion(model, x: torch.Tensor, patch: int, stride: int, fill: str, batch: int):
    """x: (X, Y, Z) float in [0, 1]. Returns eta and the (X, Y, Z) occlusion map."""
    mask = x > 0
    eta0 = model(x[None, None])[0]
    value = x[mask].median() if fill == "median" else x.new_tensor(0.0)
    occluded = torch.where(mask, value, x)  # the whole volume with every lung voxel replaced
    m_np = mask.cpu().numpy()
    pos = [(a, b, c) for a in starts(x.shape[0], patch, stride) for b in starts(x.shape[1], patch, stride)
           for c in starts(x.shape[2], patch, stride)
           if m_np[a:a + patch, b:b + patch, c:c + patch].any()]
    heat = torch.zeros_like(x)
    count = torch.zeros_like(x)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = x.expand(len(chunk), *x.shape).clone()
        for j, (a, b, c) in enumerate(chunk):
            sl = (slice(a, a + patch), slice(b, b + patch), slice(c, c + patch))
            xb[(j, *sl)] = occluded[sl]
        delta = eta0 - model(xb[:, None])
        for j, (a, b, c) in enumerate(chunk):
            sl = (slice(a, a + patch), slice(b, b + patch), slice(c, c + patch))
            heat[sl] += delta[j]
            count[sl] += 1
    out = torch.where(count > 0, heat / count.clamp(min=1), torch.zeros_like(heat)) * mask
    return float(eta0), out, len(pos)


def gradcam(model, x: torch.Tensor, layer: int) -> torch.Tensor:
    store = {}
    hook = model.features[layer].register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        with torch.enable_grad():
            eta = model(x[None, None])[0]
            a = store["a"]
            (g,) = torch.autograd.grad(eta, a)
    finally:
        hook.remove()
    w = g.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=tuple(x.shape), mode="trilinear", align_corners=False)[0, 0]
    return (cam * (x > 0)).detach()


def lobe_labels(pid: str, centre: np.ndarray, seg_dir: Path, net_mask: np.ndarray) -> np.ndarray:
    """TotalSegmentator lobes resampled exactly like s2 (nearest neighbour), in the
    cache grid. Voxels of the network mask without a lobe (the 5 mm dilation rim)
    get the nearest lobe's label; everything outside the network mask is 0."""
    img = nib.as_closest_canonical(nib.load(seg_dir / f"{pid}_yr0" / "seg.nii.gz"))
    seg = np.asarray(img.dataobj)
    seg = np.where(np.isin(seg, LUNG_LABELS), seg, 0).astype(np.uint8)
    vox = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    step = SPACING_MM / vox
    offset = centre - (np.array(GRID) - 1) / 2.0 * step
    lab = ndimage.affine_transform(seg, np.diag(step), offset=offset, output_shape=GRID,
                                   order=0, mode="constant", cval=0)
    if (lab > 0).any():
        idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = lab[tuple(idx)]
    lab[~net_mask] = 0
    return lab


def check_orientation(lab: np.ndarray) -> dict:
    """RAS grid: x grows to the patient's right, z grows upwards. Left lobes must
    have a smaller mean x than right lobes; upper lobes a larger mean z than lower."""
    com = {LOBES[k]: np.argwhere(lab == k).mean(0) for k in LOBES if (lab == k).any()}
    ok = {}
    if {"LUL", "LLL", "RUL", "RLL"} <= com.keys():
        ok["left_is_left"] = com["LUL"][0] < com["RUL"][0] and com["LLL"][0] < com["RLL"][0]
        ok["upper_is_up"] = com["LUL"][2] > com["LLL"][2] and com["RUL"][2] > com["RLL"][2]
    return ok


def enrichment(sal: np.ndarray, lab: np.ndarray) -> dict:
    pos = np.clip(sal, 0, None)
    total, n = pos.sum(), (lab > 0).sum()
    out = {}
    for k, name in LOBES.items():
        sel = lab == k
        vol_share = sel.sum() / n if n else np.nan
        out[name] = (pos[sel].sum() / total) / vol_share if total > 0 and vol_share > 0 else np.nan
    return out


def lobe_summary(pts: pd.DataFrame) -> pd.DataFrame:
    """Cancer-lobe enrichment in events vs the same lobe(s) in controls / other-lobe events."""
    rows = []
    ev = pts[(pts.event == 1) & (pts.cancer_lobes != "")]
    ct = pts[pts.event == 0]
    strata = {"all": ev, "dx <= 1 y": ev[ev.time <= 365], "dx > 1 y": ev[ev.time > 365]}
    for m in METHODS:
        for sname, e in strata.items():
            own, ref_ct, ref_ev, hit, chance = [], [], [], [], []
            for _, p in e.iterrows():
                lobes = p.cancer_lobes.split("+")
                cols = [f"{m}_{l}" for l in lobes]
                own.append(p[cols].mean())
                ref_ct.append(ct[cols].mean(axis=1).median())
                other = ev[~ev.cancer_lobes.apply(lambda s: bool(set(s.split("+")) & set(lobes)))]
                ref_ev.append(other[cols].mean(axis=1).median())
                dens = p[[f"{m}_{l}" for l in LOBES.values()]].astype(float)
                hit.append(dens.idxmax()[len(m) + 1:] in lobes if dens.notna().any() else np.nan)
                chance.append(len(lobes) / len(LOBES))
            own, ref_ct, ref_ev = (np.asarray(v, float) for v in (own, ref_ct, ref_ev))
            ok = ~np.isnan(own)
            p_ct = mannwhitneyu(own[ok], ref_ct[ok]).pvalue if ok.sum() > 1 else np.nan
            p_ev = (mannwhitneyu(own[ok & ~np.isnan(ref_ev)], ref_ev[ok & ~np.isnan(ref_ev)]).pvalue
                    if (ok & ~np.isnan(ref_ev)).sum() > 1 else np.nan)
            rows.append({
                "method": m, "stratum": sname, "n_events": int(ok.sum()),
                "enrich_cancer_lobe_median": np.nanmedian(own) if ok.any() else np.nan,
                "same_lobe_controls_median": np.nanmedian(ref_ct) if ok.any() else np.nan,
                "same_lobe_other_events_median": np.nanmedian(ref_ev) if ok.any() else np.nan,
                "share_events_above_control_ref": float(np.mean(own[ok] > ref_ct[ok])) if ok.any() else np.nan,
                "p_vs_controls": p_ct, "p_vs_other_events": p_ev,
                "top_lobe_hit_rate": float(np.nanmean(np.asarray(hit, float))) if hit else np.nan,
                "top_lobe_chance": float(np.mean(chance)) if chance else np.nan,
            })
    return pd.DataFrame(rows)


def overlay(ax, ct2d, sal2d, title, cmap="hot"):
    ax.imshow(ct2d.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    if sal2d is not None:
        s = np.ma.masked_where(sal2d.T <= 0, sal2d.T)
        ax.imshow(s, cmap=cmap, origin="lower", alpha=0.55, vmin=0, vmax=max(float(sal2d.max()), 1e-6))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def figure(group: str, items: list, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not items:
        return
    fig, axes = plt.subplots(len(items), 4, figsize=(12, 3.1 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        x, occ, cam = it["x"], it["occ"], it["cam"]
        px, py, pz = np.unravel_index(np.argmax(occ), occ.shape)
        lobes = it["cancer_lobes"] or "-"
        head = f"{it['pid']} eta={it['eta']:.2f} ev={it['event']} lobe={lobes}"
        overlay(axes[r, 0], x[:, :, pz], None, f"{head}\naxial z={pz}")
        overlay(axes[r, 1], x[:, :, pz], occ[:, :, pz], "occlusion (axial)")
        overlay(axes[r, 2], x[:, py, :], occ[:, py, :], f"occlusion (coronal y={py})")
        overlay(axes[r, 3], x[:, :, pz], cam[:, :, pz], "Grad-CAM (axial)", cmap="viridis")
    fig.suptitle(f"{group}: risk-increasing regions (patient's right = image right; up = cranial)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"fig_{group}.png", dpi=110)
    plt.close(fig)


def mean_maps(acc: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for g in ("events", "controls") if acc.get(g, {}).get("n", 0)]
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 2, figsize=(7, 3.3 * len(groups)), squeeze=False)
    vmax = max(float((acc[g]["sum"] / acc[g]["n"]).max()) for g in groups) or 1e-6
    for r, g in enumerate(groups):
        m = acc[g]["sum"] / acc[g]["n"]
        axes[r, 0].imshow(m.max(axis=2).T, cmap="hot", origin="lower", vmin=0, vmax=vmax)
        axes[r, 0].set_title(f"{g} (n={acc[g]['n']}): max over z (axial view)", fontsize=8)
        axes[r, 1].imshow(m.max(axis=1).T, cmap="hot", origin="lower", vmin=0, vmax=vmax)
        axes[r, 1].set_title("max over y (coronal view)", fontsize=8)
        for a in axes[r]:
            a.axis("off")
    fig.suptitle("mean occlusion saliency (positive part)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mean_maps.png", dpi=110)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    device = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    run = Path(a.runs_dir) / a.run
    out = run / f"saliency_{a.split}"
    (out / "maps").mkdir(parents=True, exist_ok=True)
    model = load_model(run / "model.pt", device)

    cohort = pd.read_csv(a.cohort, dtype={"pid": str})
    df = cohort[cohort.split == a.split].sort_values("row").reset_index(drop=True)
    prsn = pd.read_csv(a.prsn, dtype={"pid": str}, usecols=["pid", *CANCER_LOBE_COLS])
    lobes = prsn.set_index("pid")[list(CANCER_LOBE_COLS)].eq(1)
    df["cancer_lobes"] = df.pid.map(lambda p: "+".join(sorted(
        {LOBES[CANCER_LOBE_COLS[c]] for c in CANCER_LOBE_COLS if p in lobes.index and lobes.at[p, c]},
        key=list(LOBES.values()).index)))
    df.loc[df.event == 0, "cancer_lobes"] = ""
    centres = pd.read_csv(a.cache_index, dtype={"pid": str}).set_index("pid")[["centre_x", "centre_y", "centre_z"]]
    cache = np.load(a.cache, mmap_mode="r")

    t0 = time.time()
    x_all = np.asarray(cache[df["row"].to_numpy()])  # rows are sorted = sequential read
    df["eta"] = predict(model, x_all, device)
    print(f"{a.split}: {len(df)} patients, {int(df.event.sum())} events, eta in {time.time() - t0:.0f}s",
          flush=True)

    k = a.top_k
    ev, ct = df[df.event == 1], df[df.event == 0]
    groups = {"high_eta_events": ev.nlargest(k, "eta").index, "low_eta_events": ev.nsmallest(k, "eta").index,
              "high_eta_controls": ct.nlargest(k, "eta").index}
    if a.pids:
        sel = df.index[df.pid.isin(a.pids.split(","))]
        groups = {"selected": sel}
        todo = list(sel)
    else:
        ev_idx = list(ev.index[:a.limit] if a.limit else ev.index)
        n_ct = len(ev_idx) if a.n_controls < 0 else a.n_controls
        ct_idx = list(rng.choice(ct.index, size=min(n_ct, len(ct)), replace=False))
        todo = sorted(set(ev_idx) | set(ct_idx) | {i for g in groups.values() for i in g})
    fig_idx = {i for g in groups.values() for i in g}

    rows, items, orient = [], {}, []
    acc = {"events": {"sum": np.zeros(GRID, np.float64), "n": 0},
           "controls": {"sum": np.zeros(GRID, np.float64), "n": 0}}
    for n, i in enumerate(todo, 1):
        t1 = time.time()
        p = df.loc[i]
        x = to_input(torch.from_numpy(x_all[i:i + 1]).to(device))[0, 0]
        eta0, occ, n_pos = occlusion(model, x, a.patch, a.stride, a.fill, a.batch)
        cam = gradcam(model, x, a.gradcam_layer)
        occ_np, cam_np, x_np = occ.cpu().numpy(), cam.cpu().numpy(), x.cpu().numpy()
        row = {"pid": p.pid, "event": int(p.event), "time": int(p.time), "eta": float(p.eta),
               "eta_check": eta0, "cancer_lobes": p.cancer_lobes, "n_occlusions": n_pos}
        if not a.pids:
            lab = lobe_labels(p.pid, centres.loc[p.pid].to_numpy(float), Path(a.seg_dir), x_np > 0)
            orient.append(check_orientation(lab))
            for m, sal in (("occ", occ_np), ("cam", cam_np)):
                row.update({f"{m}_{name}": v for name, v in enrichment(sal, lab).items()})
            g = "events" if p.event == 1 else "controls"
            acc[g]["sum"] += np.clip(occ_np, 0, None)
            acc[g]["n"] += 1
        rows.append(row)
        if i in fig_idx:
            items[i] = {"pid": p.pid, "eta": float(p.eta), "event": int(p.event),
                        "cancer_lobes": p.cancer_lobes, "x": x_np, "occ": occ_np, "cam": cam_np}
            np.savez_compressed(out / "maps" / f"{p.pid}.npz", occlusion=occ_np.astype(np.float16),
                                gradcam=cam_np.astype(np.float16))
        print(f"[{n}/{len(todo)}] {p.pid} ev={int(p.event)} eta={p.eta:.3f} "
              f"occlusions={n_pos} {time.time() - t1:.1f}s", flush=True)

    pts = pd.DataFrame(rows)
    pts["group"] = ""
    for g, idx in groups.items():
        pts.loc[pts.pid.isin(df.loc[idx, "pid"]), "group"] = g
    pts.to_csv(out / "patients.csv", index=False)
    print(f"max |eta - eta_check| = {np.abs(pts.eta - pts.eta_check).max():.2e}")
    if orient:
        o = pd.DataFrame(orient)
        print("lobe orientation checks (share of patients passing):", o.mean().round(3).to_dict())
    if not a.pids:
        summ = lobe_summary(pts)
        summ.to_csv(out / "lobe_summary.csv", index=False)
        print(summ.round(3).to_string(index=False))
        mean_maps(acc, out)
    for g, idx in groups.items():
        figure(g, [items[i] for i in idx if i in items], out)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
