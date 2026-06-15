import numpy as np
from pathlib import Path
from load_data import DataLoader
from load_audio import load_audio
from extract_features_v1 import extract_features
from segmentation_prosodic_v1 import ProsodicSegmenter
from pantagruel_audio import PantagruelSpeechModel 
from leBenchmark_audio import LeBenchmarkSpeechModel
from transformers import AutoModel, AutoProcessor
import torch
from scipy.signal import find_peaks


def evaluate(pred, ref, tol=0.25):

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

    gt = []

    for start, end in features["pauses"]:
        if (end - start) >= 0.25:
            gt.append((start + end) / 2)

    gt.append(duration)

    return sorted(gt)




# MAIN
def main():
    path = Path(__file__).parent.parent / 'data' / 'raw' / 'cv-corpus-25.0-2026-03-09' / 'fr'
    loader = DataLoader(path)
    df = loader.load_data("train").head(200)

    baseline = ProsodicSegmenter()

    models_pantagruel = {
        "Pantagruel-B-1K": "PantagrueLLM/speech-base-1K",
        "Pantagruel-B-14K": "PantagrueLLM/speech-base-14K",
        "Pantagruel-L-14K": "PantagrueLLM/speech-large-14K",
        "Pantagruel-L-114K": "PantagrueLLM/speech-large-114K"
    }

    models_lebenchmark  = {
        "LeBenchmark-w2v-B-1k":"LeBenchmark/wav2vec2-FR-1K-base",
        "LeBenchmark-w2v-L-7k":"LeBenchmark/wav2vec2-FR-7K-large"
    }

    pantagruel_models = {
        name: PantagruelSpeechModel(path)
        for name, path in models_pantagruel.items()
    }
    
    lebenchmark_models = { 
        name: LeBenchmarkSpeechModel(model_id)
        for name, model_id in models_lebenchmark.items()
    }

    results = {
        "baseline": [],
        "Pantagruel-B-1K": [],
        "Pantagruel-B-14K": [],
        "Pantagruel-L-14K": [],
        "Pantagruel-L-114K": [],
        "LeBenchmark-w2v-B-1k":[],
        "LeBenchmark-w2v-L-7k":[],
        "LeBenchmark-w2v-L-14k":[]
    }


    for i, row in df.iterrows():

        audio, sr = load_audio(row["path"])

        features = extract_features(audio, sr)

        duration = len(audio) / sr

        gt = get_ground_truth(features, duration)

        # BASELINE
        try:
            _, times, _, _ = baseline.detect_boundaries(
                features["rms"],
                features["f0"],
                features["pauses"],
                prominence=0.1
            )

            _, _, f1 = evaluate(times, gt)
            results["baseline"].append(f1)

        except:
            results["baseline"].append(0)

        
        # PANTAGRUEL MODELS
        for name, model in pantagruel_models.items():

            try:
                times = model.detect_boundaries(audio, sr)

                _, _, f1 = evaluate(times, gt)

                results[name].append(f1)

            except:
                results[name].append(0)

        for name , model in lebenchmark_models.items():
            try :
                times = model.detect_boundaries(audio,sr)
                _,_, f1 = evaluate(times,gt)
                results[name].append(f1)
            except:
                results[name].append(0)


 
    # FINAL RESULTS
    print("\nSEGMENTATION BENCHMARK RESULTS : \n")
    for k, v in results.items():
        print(f"{k:25s} : {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
