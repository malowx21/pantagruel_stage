"""
Evaluate the fine tuned model on the test split
"""

import torch
import numpy as np 


from torch.utils.data import DataLoader 
from argparse import ArgumentParser
from pathlib import Path
from scipy.signal import   find_peaks

from src.common.config import load_config
from src.fine_tuning.head_segmentation import ProsodicSegmentationHead
from src.fine_tuning.segmentation_dataset import SegmentationData, collate_fn
from src.fine_tuning.metrics import evaluate , logits_to_boundary_times


def infer_embedding_dim(embeddings):
	if not embeddings:
		raise ValueError("Cannot infer input dimension from an empty embedding cache")
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
	existing_seed_paths = [(seed, path) for seed, path in seed_paths if path.exists()]
	if existing_seed_paths:
		return existing_seed_paths
	return [(None, checkpoints_dir / f"{base_name}.pt")]


@torch.no_grad()
def evaluate_checkpoint(model, test_loader, device, eval_config):
	precision, recall, f1 = [], [], []

	model.eval()
	for embeddings, labels, lengths, durations in test_loader:
		embeddings = embeddings.to(device)
		logits = model(embeddings, lengths).cpu().numpy()
		labels_np = labels.numpy()
		lengths_np = lengths.numpy()
		durations_np = durations.numpy() if durations is not None else lengths_np.astype(np.float32)

		for i in range(embeddings.shape[0]):
			t = int(lengths_np[i])
			duration = float(durations_np[i])
			logit = logits[i, :t]
			label = labels_np[i, :t]

			pred = logits_to_boundary_times(
				logit,
				duration,
				prominence=eval_config["peak_prominence"],
				min_distance_sec=eval_config["peak_min_distance_sec"],
			)

			ref, _ = find_peaks(label, height=0.5)
			ref_time = (ref / max(t, 1)) * duration

			results = evaluate(pred, ref_time.tolist(), tol=eval_config["tolerance_sec"])
			precision.append(results[0])
			recall.append(results[1])
			f1.append(results[2])

	return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1))


def main():

	parser = ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config_finetuning.yaml")
#	parser.add_argument("--checkpoint",type=str, default=None)
	args= parser.parse_args()
	config = load_config(args.config)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	for i in range(len(config["encoder"]["name"])):
		encoder_name = config["encoder"]["name"][i]
		
		cache_dir = Path(config["data"]["cache_dir"])
		test_file = cache_dir / f"{encoder_name}_test.pt"
		
		if not test_file.exists():
			raise FileNotFoundError("File not found, please run python -m  scripts.extract_embeddings --split test")

		data = torch.load(test_file, map_location="cpu")

		ds = SegmentationData(data["embeddings"],data["labels"],data["durations"])
		test_loader = DataLoader(ds, batch_size= config["training"]["batch_size"],
						num_workers= config["training"]["num_workers"],
						collate_fn = collate_fn,shuffle= False)

		metrics = []
		for seed, checkpoint_path in get_checkpoint_paths(config, encoder_name):
			if not checkpoint_path.exists():
				raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

			checkpoint = torch.load(checkpoint_path, map_location=device)
			model = ProsodicSegmentationHead(
				infer_embedding_dim(data["embeddings"]),
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
			print(f" \n RESULTS OF THE EVALUATION ON THE TEST SET FOR MODEL {encoder_name}{seed_label}")
			print(f" Precision : {precision:.4f}")
			print(f" Recall : {recall:.4f}")
			print(f"F1_score:  {f1:.4f}")

		if len(metrics) > 1:
			metrics_np = np.asarray(metrics)
			means = metrics_np.mean(axis=0)
			stds = metrics_np.std(axis=0)
			print(f" \n SUMMARY ON THE TEST SET FOR MODEL {encoder_name}")
			print(f" Precision : {means[0]:.4f} +/- {stds[0]:.4f}")
			print(f" Recall : {means[1]:.4f} +/- {stds[1]:.4f}")
			print(f"F1_score:  {means[2]:.4f} +/- {stds[2]:.4f}")


if  __name__ == "__main__":

	main()
