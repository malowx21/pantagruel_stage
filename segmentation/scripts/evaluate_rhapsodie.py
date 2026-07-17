"""
Evaluation of checkpoints trained on Common Voice, tested on
Rhapsodie -- NO retraining here.

Two peak detection threshold modes:
--threshold_mode frozen  : Reuses the threshold (peak_prominence) optimized
on Common Voice -- this is the "zero-shot" figure
to highlight as the main result.
--threshold_mode oracle  : Sweeps several prominence values on Rhapsodie
and keeps the best one per file -- gives an upper
bound, useful to determine if a performance gap
comes from a poor signal or a miscalibrated threshold.

To be run from the "segmentation" folder (after extract_embeddings_rhapsodie.py):

python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode frozen
python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode oracle
"""

import collections
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from scipy.signal import find_peaks

from src.common.config import load_config
from src.fine_tuning.head_segmentation import ProsodicSegmentationHead
from src.fine_tuning.segmentation_dataset import SegmentationData, collate_fn
from src.fine_tuning.metrics import evaluate, logits_to_boundary_times
from src.fine_tuning.metrics import (
    sweep_peak_prominence,
    purity_coverage,
    bootstrap_f1_ci,
)


def infer_embedding_dim(embeddings):
    return embeddings[0].shape[-1]


def get_seeds(config):
    if "seeds" in config["training"]:
        return config["training"]["seeds"]
    return [config["training"]["seed"]]


def get_checkpoint_paths(config, encoder_name):
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    base_name = f"{config['paths']['best_model_name']}_{encoder_name}"
    seed_paths = [(seed, checkpoints_dir / f"{base_name}_seed{seed}.pt")
                  for seed in get_seeds(config)]
    existing = [(s, p) for s, p in seed_paths if p.exists()]
    if existing:
        return existing
    return [(None, checkpoints_dir / f"{base_name}.pt")]


@torch.no_grad()
def run_on_rhapsodie(model, loader, device, eval_cfg, file_ids, threshold_mode):
    per_file = []
    prominences = eval_cfg.get("prominence_sweep", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4])

    model.eval()
    idx = 0
    for embeddings, labels, lengths, durations in loader:
        embeddings = embeddings.to(device)
        logits = model(embeddings, lengths).cpu().numpy()
        labels_np = labels.numpy()
        lengths_np = lengths.numpy()
        durations_np = durations.numpy()

        for i in range(embeddings.shape[0]):
            t = int(lengths_np[i])
            duration = float(durations_np[i])
            logit = logits[i, :t]
            label = labels_np[i, :t]

            ref_peaks, _ = find_peaks(label, height=0.5)
            ref_times = ((ref_peaks / max(t, 1)) * duration).tolist()

            if threshold_mode == "frozen":
                pred_times = logits_to_boundary_times(
                    logit, duration,
                    prominence=eval_cfg["peak_prominence"],
                    min_distance_sec=eval_cfg["peak_min_distance_sec"],
                )
                p, r, f1 = evaluate(pred_times, ref_times, tol=eval_cfg["tolerance_sec"])
                best_prom = eval_cfg["peak_prominence"]
            else:  # oracle
                sweep = sweep_peak_prominence(
                    logit, duration, ref_times, prominences,
                    min_distance_sec=eval_cfg["peak_min_distance_sec"],
                    tol=eval_cfg["tolerance_sec"],
                )
                best = max(sweep, key=lambda x: x["f1"])
                p, r, f1, best_prom = (
                    best["precision"], best["recall"], best["f1"], best["prominence"]
                )
                pred_times = logits_to_boundary_times(
                    logit, duration, prominence=best_prom,
                    min_distance_sec=eval_cfg["peak_min_distance_sec"],
                )

            purity, coverage = purity_coverage(pred_times, ref_times, duration)

            per_file.append({
                "file_id": file_ids[idx],
                "precision": p, "recall": r, "f1": f1,
                "purity": purity, "coverage": coverage,
                "prominence_used": best_prom,
            })
            idx += 1

    return per_file


def summarize(per_file, group_key=None):
    groups = collections.defaultdict(list)
    for row in per_file:
        key = row.get(group_key, "all") if group_key else "all"
        groups[key].append(row["f1"])

    for key, f1s in groups.items():
        mean, lo, hi = bootstrap_f1_ci(f1s)
        print(f"  {str(key):12s}  n={len(f1s):3d}  F1={mean:.4f}  IC95%=[{lo:.4f}, {hi:.4f}]")


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_finetuning.yaml")
    parser.add_argument("--threshold_mode", type=str, choices=["frozen", "oracle"], default="frozen")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(len(config["encoder"]["name"])):
        encoder_name = config["encoder"]["name"][i]

        cache_dir = Path(config["data"]["cache_dir"]) / "rhapsodie"
        rha_file = cache_dir / f"{encoder_name}_rhapsodie.pt"
        if not rha_file.exists():
            raise FileNotFoundError(
                "run python -m scripts.extract_embeddings_rhapsodie"
            )

        data = torch.load(rha_file, map_location="cpu")
        ds = SegmentationData(data["embeddings"], data["labels"], data["durations"])
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

        for seed, checkpoint_path in get_checkpoint_paths(config, encoder_name):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = ProsodicSegmentationHead(
                infer_embedding_dim(data["embeddings"]),
                config["model"]["conv_channels"],
                config["model"]["dropout"],
                config["model"]["lstm_hidden"],
                config["model"]["lstm_layers"],
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

            per_file = run_on_rhapsodie(
                model, loader, device, config["evaluation"],
                data["file_ids"], args.threshold_mode,
            )

            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"\nRHAPSODIE  ({args.threshold_mode}) - {encoder_name}{seed_label}")
            summarize(per_file)


if __name__ == "__main__":
    main()

