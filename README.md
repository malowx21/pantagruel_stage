# Fine-tuning a Prosodic Segmentation Head

This project trains a prosodic segmentation head (`ProsodicSegmentationHead`)
on top of frozen audio encoders, using Common Voice FR, to detect prosodic
boundaries. It extends the existing zero-shot work in `src/zero_shot/` by
turning pause and prosodic peak detection heuristics into supervised training
labels.

The current configuration compares several encoders:

- `Pantagruel-B-1K`
- `Pantagruel-B-14K`
- `Pantagruel-L-14K`
- `LeBenchmark-w2v-B-1k`
- `LeBenchmark-w2v-L-7k`

## Structure

```text
configs/
  config_finetuning.yaml          # hyperparameters, paths, and encoder list

src/
  common/
    config.py                     # YAML config loading
    load_data.py                  # Common Voice loading
    load_audio.py                 # audio loading
    extract_features_v1.py        # RMS, F0, and pause extraction

  fine_tuning/
    segmentation_dataset.py       # Dataset + collate_fn with dynamic padding
    head_segmentation.py          # Conv1D + BiLSTM segmentation head
    generate_labels.py            # continuous label generation with Gaussians
    metrics.py                    # precision, recall, F1, and peak extraction

  zero_shot/
    pantagruel_audio.py           # Pantagruel encoder wrapper
    leBenchmark_audio.py          # LeBenchmark encoder wrapper
    segmentation_prosodic_v1.py   # heuristic prosodic baseline
    run_pantagruel_benchmark.py   # zero-shot benchmark

scripts/
  extract_embeddings.py           # encodes a split and caches labels
  train.py                        # trains the head on cached embeddings
  evaluate.py                     # evaluates checkpoints on the test split

data/cache/                       # cached embeddings + labels per encoder/split
checkpoints/                      # saved best checkpoints
requirements.txt                  # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

If you use CUDA, it is often better to install `torch` with the official command
matching your CUDA version before installing the remaining dependencies.

Pantagruel models may require a Hugging Face token. If needed, add a `.env` file
at the project root:

```text
HF_TOKEN=your_huggingface_token
```

## Configuration

The main configuration file is:

```bash
configs/config_finetuning.yaml
```

It contains:

- the Common Voice FR root path (`data.common_voice_root`);
- encoder names and model identifiers;
- maximum split sizes;
- model hyperparameters;
- training parameters;
- evaluation thresholds.

By default, Common Voice files are expected here:

```text
data/raw/cv-corpus-25.0-2026-03-09/fr
```

## Full Pipeline

### 1. Extract Embeddings and Labels

This step must be run once per split. It loads Common Voice, encodes each audio
file with every configured encoder, generates continuous labels from detected
pauses, and saves the tensors in `data/cache/`.

```bash
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split test
```

Generated files follow this format:

```text
data/cache/<encoder_name>_<split>.pt
```

Example:

```text
data/cache/Pantagruel-B-1K_train.pt
data/cache/Pantagruel-B-1K_dev.pt
data/cache/Pantagruel-B-1K_test.pt
```

### 2. Train the Segmentation Head

```bash
python -m scripts.train --config configs/config_finetuning.yaml
```

The script loads cached embeddings, trains one segmentation head per encoder,
applies early stopping on the validation split, and saves the best checkpoint in
`checkpoints/`.

Checkpoints follow this format:

```text
checkpoints/best_segmentation_head_<encoder_name>.pt
```

### 3. Evaluate on the Test Split

```bash
python -m scripts.evaluate --config configs/config_finetuning.yaml
```

The script reloads each checkpoint, predicts prosodic boundaries on the test
split, and prints precision, recall, and F1-score.

## Notes

- Encoders are used as frozen representation extractors.
- Supervised labels are built from pauses detected in the audio, with Gaussians
  centered on boundary positions.
- Sequences have variable lengths; padding is handled by `collate_fn`, and the
  loss is masked during training.
- The `valid` split maps to `data.split_valid`, which is currently set to `dev`
  in the configuration.
