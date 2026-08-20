"""
Analyze false positives / false negatives for audio+text segmentation heads.
Mirrors analyze_errors.py but uses ProsodicSegmentationHeadText and *_text.pt caches.

Usage examples:
  python -m scripts.analyze_errors_text --encoder Pantagruel-Speech-Text-B-1K-v0 --seed 42
  python -m scripts.analyze_errors_text --encoder Pantagruel-Speech-Text-B-1K   --seed 123
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from src.common.config import load_config
from src.fine_tuning.head_segmentation_text import ProsodicSegmentationHeadText
from src.fine_tuning.metrics import evaluate, logits_to_boundary_times, match_boundaries


def infer_embedding_dim(embeddings):
    if not embeddings:
        raise ValueError("Cannot infer input dimension from an empty embedding cache")
    return embeddings[0].shape[-1]


def resolve_checkpoint(config, encoder_name, seed):
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    base_name = f"{config['paths']['best_model_name']}_{encoder_name}"

    if seed is not None:
        p = checkpoints_dir / f"{base_name}_seed{seed}_text.pt"
        if p.exists():
            return p

    p = checkpoints_dir / f"{base_name}_text.pt"
    if p.exists():
        return p

    raise FileNotFoundError(
        f"No _text checkpoint found for {encoder_name} (seed={seed})."
    )


def build_model(config, input_dim, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProsodicSegmentationHeadText(
        input_dim,
        config["model"]["conv_channels"],
        config["model"]["dropout"],
        config["model"]["lstm_hidden"],
        config["model"]["lstm_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def json_list(values):
    return json.dumps([float(v) for v in values])


@torch.no_grad()
def analyze_examples(config, model, data, device):
    rows      = []
    tolerance = config["evaluation"]["tolerance_sec"]
    has_text  = "text_features" in data

    for idx, items in enumerate(zip(
        data["embeddings"],
        data["labels"],
        data["durations"],
        data["text_features"] if has_text else [None] * len(data["embeddings"]),
    )):
        embedding, label, duration, text_feat = items
        length   = int(embedding.shape[0])
        duration = float(duration)

        inputs  = embedding.float().unsqueeze(0).to(device)
        lengths = torch.tensor([length], dtype=torch.long)

        # forward with text when available
        if has_text and text_feat is not None:
            text_in = text_feat.float().unsqueeze(0).to(device)
        else:
            text_in = None

        logits       = model(inputs, lengths, text=text_in).squeeze(0).cpu().numpy()
        sample_logit = logits[:length]
        sample_label = label.float().numpy()[:length]

        pred_times = logits_to_boundary_times(
            sample_logit,
            duration=duration,
            prominence=config["evaluation"]["peak_prominence"],
            min_distance_sec=config["evaluation"]["peak_min_distance_sec"],
        )

        ref_peaks, _ = find_peaks(sample_label, height=0.5)
        ref_times    = ((ref_peaks / max(length, 1)) * duration).tolist()

        precision, recall, f1 = evaluate(pred_times, ref_times, tol=tolerance)
        tp, fp, fn            = match_boundaries(pred_times, ref_times, tol=tolerance)

        rows.append({
            "sample_index": idx,
            "duration":     duration,
            "precision":    precision,
            "recall":       recall,
            "f1":           f1,
            "n_pred":       len(pred_times),
            "n_ref":        len(ref_times),
            "n_tp":         len(tp),
            "n_fp":         len(fp),
            "n_fn":         len(fn),
            "pred_times":   json_list(pred_times),
            "ref_times":    json_list(ref_times),
            "true_positive_pairs": json.dumps([[float(p), float(r)] for p, r in tp]),
            "false_positives":     json_list(fp),
            "false_negatives":     json_list(fn),
        })

    # worst examples first
    rows.sort(key=lambda row: (row["f1"], -(row["n_fp"] + row["n_fn"])))
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index", "duration",
        "precision", "recall", "f1",
        "n_pred", "n_ref", "n_tp", "n_fp", "n_fn",
        "pred_times", "ref_times",
        "true_positive_pairs", "false_positives", "false_negatives",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="configs/config_finetuning.yaml")
    parser.add_argument("--encoder", type=str, required=True,
                        help="e.g. Pantagruel-Speech-Text-B-1K-v0")
    parser.add_argument("--split",  type=str, default="test")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_file = (Path(config["data"]["cache_dir"])
                  / f"{args.encoder}_{args.split}_text.pt")
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            "Run: python -m scripts.extract_embeddings_speech_text"
            f" --split {args.split}"
        )

    data            = torch.load(cache_file, map_location="cpu")
    checkpoint_path = resolve_checkpoint(config, args.encoder, args.seed)
    model           = build_model(
        config, infer_embedding_dim(data["embeddings"]), checkpoint_path, device
    )

    rows = analyze_examples(config, model, data, device)

    seed_tag    = f"_seed{args.seed}" if args.seed is not None else ""
    output_path = (
        Path(args.output) if args.output
        else Path("error_analysis") / f"{args.encoder}{seed_tag}_{args.split}_text_errors.csv"
    )
    write_csv(rows, output_path)

    f1_values = np.asarray([row["f1"] for row in rows], dtype=np.float32)
    print(f"[errors] wrote {len(rows)} examples -> {output_path}")
    print(f"[errors] mean_f1={float(f1_values.mean()):.4f}")
    print("[errors] worst 10 examples:")
    for row in rows[:10]:
        print(
            f"  idx={row['sample_index']:5d}  f1={row['f1']:.4f}"
            f"  fp={row['n_fp']}  fn={row['n_fn']}"
        )


if __name__ == "__main__":
    main()
