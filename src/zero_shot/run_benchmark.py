import numpy as np
from pathlib import Path 
from load_data import DataLoader
from load_audio import load_audio
from extract_features_v1 import extract_features
from segmentation_prosodic_v1 import ProsodicSegmenter


def evaluate(pred, ref, tol=0.2):

    tp = 0
    matched = set()

    for p in pred:
        for i, r in enumerate(ref):
            if i not in matched and abs(p - r) <= tol:
                tp += 1
                matched.add(i)
                break

    fp = len(pred) - tp
    fn = len(ref) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def get_ground_truth(features, duration):

    pauses = features["pauses"]

    # garder seulement pauses longues
    gt = [
        (start + end) / 2
        for start, end in pauses
        if (end - start) > 0.2
    ]

    # toujours inclure fin de phrase
    gt.append(duration)

    return gt

def main():
    path  = Path(__file__).parent.parent / 'data' / 'raw' / 'cv-corpus-25.0-2026-03-09'/ 'fr'
    loader = DataLoader(path)
    df = loader.load_data("train")

    model = ProsodicSegmenter()

    results = []

    for i, row in df.head(200).iterrows():

        audio, sr = load_audio(row["path"])

        features = extract_features(audio, sr)

        peaks, times, _, _ = model.detect_boundaries(
            features["rms"],
            features["f0"],
            features["pauses"],
            prominence=0.1  # IMPORTANT: plus permissif
        )

        duration = len(audio) / sr

    
        gt = []

        # 1. pauses longues = pseudo-boundaries
        for start, end in features["pauses"]:
            if (end - start) > 0.2:
                gt.append((start + end) / 2)

        # 2. toujours inclure fin de phrase
        gt.append(duration)

        # sécurité
        gt = sorted(gt)

        p, r, f1 = evaluate(times, gt, tol=0.25)

        results.append(f1)

    print("F1 moyen:", np.mean(results))

if __name__ == "__main__":
    main()
