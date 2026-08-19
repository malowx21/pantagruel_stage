"""
Extracting embeddings from the frozen encoder

In the folder "segmentation" run :


python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split  test
"""

import torch
import numpy as np
import argparse
from pathlib import Path


from src.common.load_data import DataLoader
from src.common.load_audio import load_audio
from src.common.extract_features_v1 import extract_features
from src.zero_shot.pantagruel_audio import PantagruelSpeechModel
from src.zero_shot.leBenchmark_audio import LeBenchmarkSpeechModel
from src.zero_shot.pantagruel_audio_text import PantagruelSpeechTextAudioModel
from src.fine_tuning.generate_labels import generate_labels
from src.fine_tuning.text_alignment import align_text_to_frames

from src.common.config import load_config


def load_encoder(model_id):
    if model_id.startswith("LeBenchmark/"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LeBenchmarkSpeechModel(model_id, device=device)
    if model_id.startswith("PantagrueLLM/Speech_Text"):
        return PantagruelSpeechTextAudioModel(model_id)
    return PantagruelSpeechModel(model_id)


def load_word_alignments(config, split):
    """
    Loads the word-alignment cache produced by generate_word_alignment.py
    for a given split, if it exists.

    Returns:
        dict mapping row["path"] -> {"words": [...], "duration": ..., "sentence": ...},
        or None if the cache file does not exist.
    """
    align_path = Path(config["data"]["cache_dir"]) / "word_alignement" / f"{split}_word_alignment.pt"
    if not align_path.exists():
        return None
    return torch.load(align_path)


def extract(config, split, max_samples, model_id, name):

    data_dir = Path(config['data']['common_voice_root'])
    cache_dir = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir)
    df = loader.load_data(split=split)
    if max_samples is not None:
        df = df.head(max_samples)

    encoder = load_encoder(model_id)

    has_text = hasattr(encoder, "encode_text")

    word_alignments = None
    if has_text:
        word_alignments = load_word_alignments(config, split)
        if word_alignments is None:
            raise FileNotFoundError(
                f"Encoder '{name}' supports encode_text but no word alignment "
                f"cache was found for split='{split}'. Run "
                f"scripts.word_alignment for this split first."
            )

    embeddings, labels, durations = [], [], []
    text_features = [] if has_text else None
    n_missing_alignment = 0

    for index, row in df.iterrows():

        audio, sr = load_audio(row['path'])
        features = extract_features(audio=audio, sr=sr)
        duration = len(audio) / sr

        embedding = encoder.encode(audio, sr)
        num_frames = embedding.shape[0]
        label = generate_labels(features, duration, num_frames, config)

        embeddings.append(torch.from_numpy(embedding).float())
        labels.append(torch.from_numpy(label).float())
        durations.append(duration)

        if has_text:
            entry = word_alignments.get(row['path'])
            words = entry["words"] if entry is not None else []

            if not words:
                n_missing_alignment += 1

            word_list = [w for w, _, _ in words]
            dim = embedding.shape[-1]

            if word_list:
                word_vectors = encoder.encode_text(word_list)
            else:
                word_vectors = np.zeros((0, dim), dtype=np.float32)

            text = align_text_to_frames(
                words, word_vectors, num_frames, duration, dim=dim
            )
            text_features.append(torch.from_numpy(text).float())

    if has_text and n_missing_alignment > 0:
        print(
            f"[extract] warning: {n_missing_alignment}/{len(df)} clips had no "
            f"word alignment (empty words list) -- their text_features are all-zero."
        )

    out_dict = {"embeddings": embeddings, "labels": labels, "durations": durations}
    if has_text:
        out_dict["text_features"] = text_features

    out_path = cache_dir / f"{name}_{split}_text.pt"
    torch.save(out_dict, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config_finetuning.yaml')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'])
    args = parser.parse_args()
    config = load_config(args.config)

    split_key = {
        "train": "split_train",
        "valid": "split_valid",
        "test": "split_test",
    }
    max_key = {
        "train": "max_samples_train",
        "valid": "max_samples_valid",
        "test": "max_samples_valid",
    }

    split_name = config['data'][split_key[args.split]]
    max_samples = config['data'].get(max_key[args.split])

    for i in range(len(config["encoder"]["model_id"])):
        extract(
            config,
            split=split_name,
            max_samples=max_samples,
            model_id=config["encoder"]["model_id"][i],
            name=config["encoder"]["name"][i],
        )


if __name__ == "__main__":
    main()
