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
from src.fine_tuning.head_segmentation import ProsodicSegmentatioHead
from src.fine_tuning.segmentation_dataset import SegmentationData, collate_fn
from src.fine_tuning.metrics import evaluate , logits_to_boundary_times


def main():

	parser = ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config_finetuning.yaml")
	parser.add_argument("--checkpoint",type=str, default=None)
	args= parser.parse_args()
	config = load_config(args.config)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint_path = Path(config["paths"]["checkpoints_dir"])/ config["paths"]["best_model_name"]
	checkpoint = torch.load(checkpoint_path)

	model = ProsodicSegmentationHead(
		input_dim= config["encoder"]["hidden_dim"],
		conv_channels= config["model"]["conv_channels"],
		lstm_hidden= config["model"]["lstm_hidden"],
		lstm_layers= config["model"]["lstm_layers"],
		dropout= config["model"]["dropout"],
		).to(device)

	model.load_state_dict(checkpoint["model_state_dict"]
	model.eval()

	cache_dir = Path(config["data"]["cache_dir"]
	test_file = cache_dir / f"{config["encoder"]["name"]}_test.pt"
	
	if not test_file.exists():
		raise FileNotFoundError("File not found, please run python -m  scripts.extract_embeddings --split test")

	data= torch.load(test_file)
	ds = SegmentationData(data["embeddings"],data["labels"],data["durations"])
	test_loader = DataLoader(ds, batch_size= config["training"]["batch_size"],
					num_workers= config["training"]["num_workers"],
					collate_fn = collate_fn,suffle= False)


	precision, recall , f1 = [], [], []

	with torch.no_grad():
		for embeddings, labels , lengths, durations in test_loader :
			embeddings = embeddings.to(device)
			logits= model(embeddings,lengths).cpu().numpy()
			labels_np = labels.numpy()
			lengths_np = lengths.numpy()
			durations_np = (durations.numpy if durations is not None else lengths_np.astype(np.float32))

		for i in range(embeddings.shape[0]):
			t= int(lengths_np[i])
			duration = float(durations[i])
			logit = logits[i,:t]
			label = labels[i, :t]

			pred = logits_to_boudary_times(logit,duration, prominence=config["evaluation"]["peak_prominence"],min_distance_sec = config["evaluation"]["peak_min_distance_sec"])

			ref,_ = find_peaks(label,prominence= config["evaluation"]["peak_prominence"], distance = config["evaluation"]["peak_min_distance_sec"])
			ref_time = (ref / max(t,1))*duration

			results = evaluate(pred, ref_time.tolist(), tol = config["evaluation"]["tolerance_sec"])


			precision.append(results[0])
			recall.append(results[1])
			f1.append(results[2])


	print(" \n RESULTS OF THE EVALUATION ON THE TEST SET")
	print(" Precision :", np.mean(precision))
	print(" Recall :" , np.mean(recall))
	print("F1_score: ", np.mean(f1))


if  __name__ == "__main__":

	main()
