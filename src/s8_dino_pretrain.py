#!/usr/bin/env python3
"""S8 -- DINOv2 pretraining on the lung volumes (D2 smoke/throughput, D3 full run).

The repo's scripts/run.py with two additions: it resumes from
runs/dino/{run}/ckpt/last.ckpt when that exists (run.py cannot resume, and a
long run is several Slurm jobs), and it refuses to resume with a different
config, because the lr / teacher-momentum / temperature schedules are all
fixed by trainer#max_epochs x steps per epoch at the first launch.

Configs (merged left to right; later `a#b` keys override):
  ext/DINOv2-3D-Med/configs/train.yaml       repo defaults (lr, losses, heads)
  ext/DINOv2-3D-Med/configs/models/<m>.yaml  backbone (default primus = PRIMUS-M)
  configs/dino/nlst_2p5mm.yaml               our data, 1 GPU, bf16, CSV logging
Overrides: key=value in MONAI config syntax, e.g. trainer#max_epochs=100.

Outputs, runs/dino/{run}/:
  launch.json         config files + overrides of the first launch (checked on resume)
  metrics.csv         losses, lr, collapse monitors, throughput (CSVLogger)
  gram/step_N.png     Gram matrix of the student CLS tokens
  ckpt/last.ckpt      resume point; ckpt/dino_epoch=NNN.ckpt every 10 epochs

Usage:
    sbatch src/submit_s8_dino_pretrain.sh --run-name trial_2h/big trainer#max_epochs=100
    env/.venv-dino/bin/python src/s8_dino_pretrain.py --run-name x --print-config
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "ext" / "DINOv2-3D-Med"
sys.path[:0] = [str(REPO), str(PROJECT / "src")]
RUNS = PROJECT / "runs" / "dino"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="primus", help="ext/.../configs/models/<model>.yaml")
    ap.add_argument("--config", default=str(PROJECT / "configs" / "dino" / "nlst_2p5mm.yaml"))
    ap.add_argument("--print-config", action="store_true", help="resolve and print, no training")
    ap.add_argument("overrides", nargs="*", help="key=value, MONAI config syntax")
    a = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    from monai.bundle import ConfigParser
    from utils.imports import import_module_from_path

    files = [str(REPO / "configs" / "train.yaml"), str(REPO / "configs" / "models" / f"{a.model}.yaml"),
             a.config]
    overrides = {"run_name": a.run_name, "runs_dir": str(RUNS)}
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)

    run_dir = RUNS / a.run_name
    launch = {"config_files": files, "overrides": overrides}
    ckpt = run_dir / "ckpt" / "last.ckpt"
    if (run_dir / "launch.json").exists():
        first = json.loads((run_dir / "launch.json").read_text())
        if first != launch:
            raise SystemExit(f"{run_dir} was launched with\n{json.dumps(first, indent=1)}\n"
                             f"now\n{json.dumps(launch, indent=1)}\n-- same arguments to resume, "
                             "or a new --run-name")
        if ckpt.exists():
            print(f"resuming from {ckpt}", flush=True)
    elif not a.print_config:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=1))

    # Overrides go in BEFORE parse(): the repo configs copy values with % macros
    # (e.g. embed_dim: "%hidden_size"), which parse() resolves. run.py updates after
    # parse(), so an override of hidden_size or trainer#log_every_n_steps never reaches
    # the copies (checked 2026-10-19).
    parser = ConfigParser()
    parser.read_config(files)
    parser.update(overrides)
    parser.parse()
    import_module_from_path("project", str(REPO))
    if a.print_config:
        for key in ["trainer", "lightning_module", "data_module"]:
            print(key, json.dumps(parser.get(key), indent=1, default=str))
        return

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    trainer = parser.get_parsed_content("trainer")
    module = parser.get_parsed_content("lightning_module")
    data = parser.get_parsed_content("data_module")
    n_params = sum(p.numel() for p in module.model.student_backbone.parameters())
    print(f"backbone {n_params / 1e6:.1f} M parameters, {len(data.train_dataset)} scans, "
          f"max_epochs {trainer.max_epochs}", flush=True)
    trainer.fit(module, data, ckpt_path=str(ckpt) if ckpt.exists() else None)


if __name__ == "__main__":
    main()
