"""
Embedding extraction : reading the annotations from TextGri>
used for common Voice dataset . The files in cache have sam>

Run in the directory "segmentation" :

python -m scripts.extract_embeddings_rhapsodie --config configs/config_finetuning.yaml
"""

import torch
import argparse
from pathlib import Path

from src.common.load_data import RhapsodieDataLoader
from src.common.load_audio import load_audio
from src.common.load_annotations_rhap import get_ground_truth_rhapsodie
from src.fine_tuning.generate_labels import boudaries_to_labels
from src.zero_shot.pantagruel_audio import PantagruelSpeechModel
from src.zero_shot.leBenchmark_audio import LeBenchmarkSpeechModel
from src.common.config import load_config


def load_encoder(model_id):
    if model_id.startswith("LeBenchmark/"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LeBenchmarkSpeechModel(model_id, device=device)
    return PantagruelSpeechModel(model_id)


def extract(config, model_id, name):
    rha_cfg = config["rhapsodie"]

    loader = RhapsodieDataLoader(
        audio_dir=rha_cfg["audio_dir"],
        annotations_dir=rha_cfg["annotations_dir"],
    )
    df = loader.load_data()
    if len(df) == 0:
        raise RuntimeError(
            "Aucune paire audio/TextGrid trouvée -- vérifier "
            "rhapsodie.audio_dir et rhapsodie.annotations_dir dans le config."
        )

    cache_dir = Path(config["data"]["cache_dir"]) / "rhapsodie"
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(model_id)

    embeddings, labels, durations, file_ids= [], [], [], []

    for _, row in df.iterrows():
        audio, sr = load_audio(row["path"])
        duration = len(audio) / sr

        embedding = encoder.encode(audio, sr)
        num_frames = embedding.shape[0]

        boundaries = get_ground_truth_rhapsodie(
            row["textgrid_path"],
            tier_name=rha_cfg["tier_name"],
            duration=duration,
            include_utterance_end=rha_cfg.get("include_utterance_end", True),
        )
        label = boudaries_to_labels(
            boundaries,
            num_frames,
            duration,
            sigma_frames=config["labels"]["gaussian_sigma_frames"],
        )

        embeddings.append(torch.from_numpy(embedding).float())
        labels.append(torch.from_numpy(label).float())
        durations.append(duration)
        file_ids.append(row["file_id"])

    out_path = cache_dir / f"{name}_rhapsodie.pt"
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "durations": durations,
            "file_ids": file_ids,
        },
        out_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_finetuning.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    for i in range(len(config["encoder"]["model_id"])):
        extract(
            config,
            model_id=config["encoder"]["model_id"][i],
            name=config["encoder"]["name"][i],
        )


if __name__ == "__main__":
    main()

