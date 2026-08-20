"""
Evaluate fine-tuned segmentation heads (audio + text) on the test split.
Mirrors evaluate.py but:
  - loads   *_test_text.pt  caches
  - uses    ProsodicSegmentationHeadText
  - expects checkpoints  best_segmentation_head_<enc>_seed<s>_text.pt
"""

import torch
import numpy as np

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from scipy.signal import find_peaks

from src.common.config import load_config
from src.fine_tuning.head_segmentation_text import ProsodicSegmentationHeadText
from src.fine_tuning.segmentation_dataset import (
    SegmentationDataText, SegmentationData,
    collate_fn_text, collate_fn,
)
from src.fine_tuning.metrics import evaluate, logits_to_boundary_times


def infer_embedding_dim(embeddings):
    if not embeddings:
        raise ValueError("Cannot infer input dimension from an empty embedding cache")
    return embeddings[0].shape[-1]


def get_seeds(config):
    if "seeds" in config["training"]:
        return config["training"]["seeds"]
    return [config["training"]["seed"]]


def get_checkpoint_paths(config, encoder_name):
    """Return [(seed, Path), ...] for all checkpoints that exist on disk."""
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    base_name = f"{config['paths']['best_model_name']}_{encoder_name}"

    seed_paths = [
        (seed, checkpoints_dir / f"{base_name}_seed{seed}_text.pt")
        for seed in get_seeds(config)
    ]
    existing = [(s, p) for s, p in seed_paths if p.exists()]
    if existing:
        return existing
    # fallback: single-seed layout (seed 42 saved without seed suffix)
    return [(None, checkpoints_dir / f"{base_name}_text.pt")]


def build_loader(data, batch_size, num_workers):
    has_text = "text_features" in data
    if has_text:
        ds = SegmentationDataText(
            data["embeddings"], data["labels"], data["durations"], data["text_features"]
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=collate_fn_text, shuffle=False), True
    else:
        ds = SegmentationData(data["embeddings"], data["labels"], data["durations"])
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=collate_fn, shuffle=False), False


def unpack_batch(batch):
    if len(batch) == 5:
        embeddings, labels, lengths, durations, text = batch
    else:
        embeddings, labels, lengths, durations = batch
        text = None
    return embeddings, labels, lengths, durations, text


@torch.no_grad()
def evaluate_checkpoint(model, test_loader, device, eval_config):
    precision_list, recall_list, f1_list = [], [], []

    model.eval()
    for batch in test_loader:
        embeddings, labels, lengths, durations, text = unpack_batch(batch)
        embeddings = embeddings.to(device)
        text = text.to(device) if text is not None else None

        logits = model(embeddings, lengths, text=text).cpu().numpy()
        labels_np  = labels.numpy()
        lengths_np = lengths.numpy()
        durations_np = (durations.numpy() if durations is not None
                        else lengths_np.astype(np.float32))

        for i in range(embeddings.shape[0]):
            t        = int(lengths_np[i])
            duration = float(durations_np[i])
            logit    = logits[i, :t]
            label    = labels_np[i, :t]

            pred = logits_to_boundary_times(
                logit, duration,
                prominence=eval_config["peak_prominence"],
                min_distance_sec=eval_config["peak_min_distance_sec"],
            )
            ref_peaks, _ = find_peaks(label, height=0.5)
            ref_time = (ref_peaks / max(t, 1)) * duration

            p, r, f = evaluate(pred, ref_time.tolist(), tol=eval_config["tolerance_sec"])
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f)

    return (float(np.mean(precision_list)),
            float(np.mean(recall_list)),
            float(np.mean(f1_list)))

# main

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_finetuning.yaml")
    args = parser.parse_args()

    config     = load_config(args.config)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir  = Path(config["data"]["cache_dir"])
    batch_size = config["training"]["batch_size"]
    num_workers= config["training"]["num_workers"]

    for encoder_name in config["encoder"]["name"]:

        test_file = cache_dir / f"{encoder_name}_test_text.pt"
        if not test_file.exists():
            raise FileNotFoundError(
                f"Cache file not found: {test_file}\n"
                "Run: python -m scripts.extract_embeddings_speech_text --split test"
            )

        data = torch.load(test_file, map_location="cpu")
        has_text = "text_features" in data
        print(f"\n[eval] encoder={encoder_name}  text_features={'yes' if has_text else 'no'}")

        test_loader, _ = build_loader(data, batch_size, num_workers)
        input_dim = infer_embedding_dim(data["embeddings"])

        metrics = []
        for seed, ckpt_path in get_checkpoint_paths(config, encoder_name):
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location=device)
            model = ProsodicSegmentationHeadText(
                input_dim,
                config["model"]["conv_channels"],
                config["model"]["dropout"],
                config["model"]["lstm_hidden"],
                config["model"]["lstm_layers"],
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

            precision, recall, f1 = evaluate_checkpoint(
                model, test_loader, device, config["evaluation"]
            )
            metrics.append((precision, recall, f1))

            seed_label = f" seed={seed}" if seed is not None else ""
            print(f"\n RESULTS ON THE TEST SET -- model={encoder_name}{seed_label}")
            print(f"   Precision : {precision:.4f}")
            print(f"   Recall    : {recall:.4f}")
            print(f"   F1        : {f1:.4f}")

        if len(metrics) > 1:
            arr   = np.asarray(metrics)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            print(f"\n SUMMARY (all seeds) -- model={encoder_name}")
            print(f"   Precision : {means[0]:.4f} +/- {stds[0]:.4f}")
            print(f"   Recall    : {means[1]:.4f} +/- {stds[1]:.4f}")
            print(f"   F1        : {means[2]:.4f} +/- {stds[2]:.4f}")


if __name__ == "__main__":
    main()
