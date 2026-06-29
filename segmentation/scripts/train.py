"""
Training segmentation head on the embedding 
"""
import torch 
import time 
import torch.nn as nn 
import numpy as np

from argparse import ArgumentParser

from pathlib import Path

from scipy.signal import find_peaks
from torch.utils.data import DataLoader 

from src.common.config import load_config
from src.fine_tuning.segmentation_dataset import SegmentationData, collate_fn
from src.fine_tuning.metrics import evaluate, logits_to_boundary_times
from src.fine_tuning.head_segmentation import ProsodicSegmentationHead

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
def load_split(cache_dir, encoder_name, split):
    path = cache_dir / f'{encoder_name}_{split}.pt'
    if not path.exists():
        raise FileNotFoundError('There is no file in cache')
    data = torch.load(path)
    return data['embeddings'], data['labels'], data['durations']


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
        for embeddings , labels , lengths, durations in loader :
            embeddings= embeddings.to(device)
            labels = labels.to(device)
            
            logits= model(embeddings,lengths)
            loss = mask_loss(base_loss_fn,labels,logits,lengths)
    
            if train :
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_f1(model,loader, device,eval_cfg):
    """
    Calculata the F1 score 
    """
    
    model.eval()
    f1_scores=[]
    for embeddings, labels , lengths, durations in loader:
        embeddings = embeddings.to(device)
        logits = model(embeddings, lengths).cpu().numpy()
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
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    cache_dir  = Path(config['data']['cache_dir'])
    for  i in range(len(config["encoder"]["name"])-1,len(config["encoder"]["name"])):
        encoder_name = config['encoder']['name'][i]
    
    
        train_emb, train_lab, train_dur = load_split(
            cache_dir, encoder_name, config["data"]["split_train"]
        )
        valid_emb, valid_lab, valid_dur = load_split(
            cache_dir, encoder_name, config["data"]["split_valid"]
         )
    
        train_ds = SegmentationData(train_emb, train_lab, train_dur)
        valid_ds = SegmentationData(valid_emb, valid_lab, valid_dur)
    
    
        train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
          )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config["training"]["num_workers"],
         )
    
        if  i<2 :
            model = ProsodicSegmentationHead(
            config["encoder"]["hidden_dim"],
            config["model"]["conv_channels"],
            config["model"]["dropout"],
            config["model"]["lstm_hidden"],
            config["model"]["lstm_layers"]).to(device)
        else : 
            model = ProsodicSegmentationHead(
            config["encoder"]["hidden_dim_large"],
            config["model"]["conv_channels"],
            config["model"]["dropout"],
            config["model"]["lstm_hidden"],
            config["model"]["lstm_layers"]).to(device)

        base_loss_fn = make_loss(config)
        if isinstance(base_loss_fn, nn.BCEWithLogitsLoss):
            base_loss_fn.pos_weight = base_loss_fn.pos_weight.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
         )

        checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoints_dir / f"{config['paths']['best_model_name']}_{encoder_name}.pt"

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
            f"[model {encoder_name}  epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} valid_f1={valid_f1:.4f} ({dt:.1f}s)"
            )

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0
                torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
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

        print(f"[train] END . Best  valid_loss = {best_val_loss:.4f} for model {encoder_name}")


if __name__ == "__main__":
    main()

