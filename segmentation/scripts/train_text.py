"""
Training segmentation head on the embedding
"""
import torch
import time
import torch.nn as nn
import numpy as np
import random

from argparse import ArgumentParser

from pathlib import Path

from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.fine_tuning.segmentation_dataset import (
    SegmentationData, SegmentationDataText, collate_fn, collate_fn_text,
)
from src.fine_tuning.metrics import evaluate, logits_to_boundary_times
from src.fine_tuning.head_segmentation_text import ProsodicSegmentationHeadText

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_seeds(config, cli_seeds):
    if cli_seeds:
        return cli_seeds
    if "seeds" in config["training"]:
        return config["training"]["seeds"]
    return [config["training"]["seed"]]


def checkpoint_path(config, encoder_name, seed, multiple_seeds):
    filename = f"{config['paths']['best_model_name']}_{encoder_name}"
    if multiple_seeds:
        filename = f"{filename}_seed{seed}"
    return Path(config["paths"]["checkpoints_dir"]) / f"{filename}_text.pt"


def load_split(cache_dir, encoder_name, split):
    path = cache_dir / f'{encoder_name}_{split}_text.pt'
    if not path.exists():
        raise FileNotFoundError('There is no file in cache')
    data = torch.load(path)
    return data


def build_dataset_and_collate(data):
    """
    Picks SegmentationData/collate_fn or SegmentationDataText/collate_fn_text
    depending on whether this cache was produced by an encoder that
    supports text (i.e. contains a "text_features" key -- see
    extract_embeddings.py).
    """
    has_text = "text_features" in data
    if has_text:
        ds = SegmentationDataText(
            data["embeddings"], data["labels"], data["durations"], data["text_features"]
        )
        return ds, collate_fn_text, True
    ds = SegmentationData(data["embeddings"], data["labels"], data["durations"])
    return ds, collate_fn, False


def unpack_batch(batch):
    """
    collate_fn returns a 4-tuple, collate_fn_text returns a 5-tuple
    (with the frame-aligned text as the extra element). Both loaders
    are handled uniformly downstream by always returning `text`, set
    to None when it is not part of the batch.
    """
    if len(batch) == 5:
        embeddings, labels, lengths, durations, text = batch
    else:
        embeddings, labels, lengths, durations = batch
        text = None
    return embeddings, labels, lengths, durations, text


def infer_embedding_dim(embeddings):
    if not embeddings:
        raise ValueError("Cannot infer input dimension from an empty embedding cache")
    return embeddings[0].shape[-1]


def make_loss(config):
    loss = config['training']['loss']
    if loss == "bce_with_logits" :
        pos_weight = torch.tensor(config['training']['pos_weight'])
        b_loss= nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss == 'mse':
        b_loss = nn.MSELoss(reduction ='none')
    else : 
        raise ValueError('Loss unknown')
    return b_loss

def mask_loss(b_loss, labels, logits,lengths):
    output = b_loss(logits, labels)
    mask = torch.arange(logits.shape[1],device=logits.device)[None,:] < lengths[:,None].to(logits.device)
    mask = mask.float()
    return (output*mask).sum() / mask.sum().clamp(min=1.0)
    
def run_epoch(model, loader, base_loss_fn, optimizer, device, grad_clip, train: bool):

    model.train() if  train else model.eval()

    total_loss, n_batches= 0 , 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            embeddings, labels, lengths, durations, text = unpack_batch(batch)
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            text = text.to(device) if text is not None else None

            logits = model(embeddings, lengths, text=text)
            loss = mask_loss(base_loss_fn, labels, logits, lengths)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_f1(model, loader, device, eval_cfg):
    """
    Calculata the F1 score
    """

    model.eval()
    f1_scores = []
    for batch in loader:
        embeddings, labels, lengths, durations, text = unpack_batch(batch)
        embeddings = embeddings.to(device)
        text = text.to(device) if text is not None else None

        logits = model(embeddings, lengths, text=text).cpu().numpy()
        labels_np = labels.numpy()
        lengths_np = lengths.numpy()
        durations_np = (durations.numpy() if durations is not None
                                    else lengths_np.astype(np.float32)  # fallback
        )

        for i in range(embeddings.shape[0]):
            t = int(lengths_np[i])
            duration = float(durations_np[i])
            sample_logits = logits[i, :t]
            sample_labels = labels_np[i, :t]

            pred_times = logits_to_boundary_times(
                sample_logits,
                duration=duration,
                prominence=eval_cfg["peak_prominence"],
                min_distance_sec=eval_cfg["peak_min_distance_sec"],
            )

            ref_peaks, _ = find_peaks(sample_labels, height=0.5)
            ref_times = (ref_peaks / max(t, 1)) * duration

            _, _, f1 = evaluate(
                pred_times, ref_times.tolist(), tol=eval_cfg["tolerance_sec"]
            )
            f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


 
def main():
 
    parser = ArgumentParser()
    parser.add_argument('--config', type=str ,default='configs/config_finetuning.yaml')
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
 
    config = load_config(args.config)
    seeds = get_seeds(config, args.seeds)
 
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cache_dir  = Path(config['data']['cache_dir'])
    multiple_seeds = len(seeds) > 1
    for seed in seeds:
        set_seed(seed)
        print(f"\n START TRAINING seed={seed}")
 
        for  i in range(len(config["encoder"]["name"])):
            encoder_name = config['encoder']['name'][i]
 
            train_data = load_split(cache_dir, encoder_name, config["data"]["split_train"])
            valid_data = load_split(cache_dir, encoder_name, config["data"]["split_valid"])
 
            train_ds, collate, has_text = build_dataset_and_collate(train_data)
            valid_ds, _, _ = build_dataset_and_collate(valid_data)
 
            print(f"[train] encoder={encoder_name} text_features={'yes' if has_text else 'no'}")
 
            generator = torch.Generator()
            generator.manual_seed(seed)
 
            train_loader = DataLoader(
                train_ds,
                batch_size=config["training"]["batch_size"],
                shuffle=True,
                collate_fn=collate,
                num_workers=config["training"]["num_workers"],
                generator=generator,
              )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                collate_fn=collate,
                num_workers=config["training"]["num_workers"],
             )
 
            input_dim = infer_embedding_dim(train_data["embeddings"])
 
 
            model = ProsodicSegmentationHeadText(
                input_dim,
                config["model"]["conv_channels"],
                config["model"]["dropout"],
                config["model"]["lstm_hidden"],
                config["model"]["lstm_layers"]).to(device)
 
            base_loss_fn = make_loss(config)
            if isinstance(base_loss_fn, nn.BCEWithLogitsLoss):
                base_loss_fn.pos_weight = base_loss_fn.pos_weight.to(device)
 
            # The cross-attention branch (text_norm, cross_attention,
            # attn_gate) is new relative to the rest of the architecture.
            # Even with a zero-initialized output projection and gate, it
            # can still destabilize training if it is allowed to move as
            # fast as the rest of the model -- this showed up as some
            # seeds getting stuck at valid_f1=0.0000 for many epochs.
            # Giving it a slower learning rate lets it "ramp up" gradually
            # once the audio-only pathway has already started converging,
            # rather than competing with it from epoch 1.
            attention_param_names = ("cross_attention", "text_norm", "attn_gate")
            attention_params = [
                p for n, p in model.named_parameters()
                if any(tag in n for tag in attention_param_names)
            ]
            other_params = [
                p for n, p in model.named_parameters()
                if not any(tag in n for tag in attention_param_names)
            ]
            attention_lr = config["training"].get(
                "attention_lr", config["training"]["learning_rate"] / 10
            )
 
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": config["training"]["learning_rate"]},
                    {"params": attention_params, "lr": attention_lr},
                ],
                weight_decay=config["training"]["weight_decay"],
             )
 
            checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            best_path = checkpoint_path(config, encoder_name, seed, multiple_seeds)
 
            best_val_loss = float("inf")
            patience = config["training"]["early_stopping_patience"]
            patience_counter = 0
 
            for epoch in range(1, config["training"]["num_epochs"] + 1):
                t0 = time.time()
 
                train_loss = run_epoch(
                model, train_loader, base_loss_fn, optimizer, device,
                config["training"]["grad_clip_norm"], train=True,
                )
                valid_loss = run_epoch(
                    model, valid_loader, base_loss_fn, optimizer, device,
                    config["training"]["grad_clip_norm"], train=False,
                   )
                valid_f1 = evaluate_f1(model, valid_loader, device, config["evaluation"])
 
                dt = time.time() - t0
                print(
                f"[seed {seed} model {encoder_name} epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} valid_f1={valid_f1:.4f} ({dt:.1f}s)"
                )
 
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    patience_counter = 0
                    torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "seed": seed,
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_f1": valid_f1,
                    },
                    best_path,
                    )
                    print(f" new best model saved   ({best_path})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"[train] early stopping in epoch {epoch}")
                        break
 
            print(f"TRAINING END  seed={seed}. Best valid_loss = {best_val_loss:.4f} for model {encoder_name}")

if __name__ == "__main__":
    main()
