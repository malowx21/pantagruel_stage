# French Prosodic Segmentation with Audio and Text Representations

This repository contains the experimental pipeline developed to compare French
self-supervised representations on a prosodic boundary detection task. It
covers audio encoders from the **Pantagruel** and **LeBenchmark** families,
the **Pantagruel Speech-Text** multimodal encoders, and two text guidance
mechanisms:

- direct concatenation of aligned audio and text representations;
- cross-attention, with audio as the query and text as the key and value.

The encoders are frozen. Only a lightweight segmentation head is trained on
Common Voice FR. Cross-corpus generalization is then measured without
retraining on Rhapsodie, an expert-annotated French speech corpus.

> **Project status:** research prototype. Data paths and encoder lists must be
> adapted in the configuration before running an experiment.

## Objectives

The project aims to:

1. compare Pantagruel and LeBenchmark under a common downstream protocol;
2. measure the effect of Speech-Text pretraining on the audio branch;
3. determine whether explicitly providing text to the head improves F1;
4. check whether rankings obtained on Common Voice transfer to more
   spontaneous speech in Rhapsodie;
5. make conclusions more robust through multiple seeds and per-file error
   analysis.

## Architectures

### Audio only

```text
audio -> frozen encoder -> embeddings (T, D)
      -> Conv1D -> BiLSTM -> MLP -> frame-level boundary score
```

### Direct concatenation

```text
audio -> aligned audio embeddings --------+
                                           +-> concatenation -> Conv1D -> BiLSTM -> MLP
text -> word embeddings -> frames --------+
```

Text vectors are repeated over the time frames covered by each word, then
concatenated with the audio vectors. The head's input dimension therefore
changes from `D` to `2D`.

### Cross-attention

```text
Q = audio frames
K = aligned text frames
V = aligned text frames

audio + Attention(Q, K, V) -> Conv1D -> BiLSTM -> MLP
```

Cross-attention allows each audio frame to weight the relevant textual
information. Its output projection is initialized to zero so that training
starts close to the audio-only behavior.

In these diagrams, `T` denotes the number of time frames and `D` the dimension
of a representation. The final MLP produces one logit per frame. Depending on
the configuration, training uses binary cross-entropy with logits or MSE, with
padding positions masked.

## Compared models

- `Pantagruel-B-1K`
- `Pantagruel-B-14K`
- `Pantagruel-L-14K`
- `Pantagruel-Speech-Text-B-1K-v0`
- `Pantagruel-Speech-Text-B-1K`
- `LeBenchmark-w2v-B-1k`
- `LeBenchmark-w2v-L-7k`

The `encoder.name` and `encoder.model_id` configuration fields are parallel
lists: they must contain the same number of elements and remain in the same
order.

## Repository layout

```text
configs/
  config_finetuning.yaml              main configuration

scripts/
  extract_embeddings.py               caches audio and Common Voice pseudo-labels
  word_alignment.py                   word-audio alignment with WhisperX
  extract_embeddings_speech_text.py   audio + aligned text caches
  train.py                            audio-only training
  train_text_conc.py                 concatenation training
  train_text_cross.py                cross-attention training
  evaluate.py                         audio-only evaluation on Common Voice
  evaluate_text_conc.py               concatenation evaluation
  evaluate_text_cross.py              cross-attention evaluation
  analyze_errors.py                  FP/FN analysis of audio heads
  extract_embeddings_rhapsodie.py    creation of the Rhapsodie cache
  evaluate_rhapsodie.py               frozen/oracle transfer to Rhapsodie

src/
  common/                             loading, configuration, and annotations
  fine_tuning/                        datasets, heads, labels, and metrics
  zero_shot/                           Pantagruel and LeBenchmark wrappers
  old/                                 previous prototypes kept for reference

checkpoints/                           trained head weights
reports/                               experimental reports and results
requirements.txt                      direct Python dependencies
LICENSE                               code license
```

Data, embedding caches, and encoder weights are not distributed with this
repository.

## Requirements

- Python 3.10 or newer;
- `ffmpeg`, especially for loading MP3 files and WhisperX;
- enough disk space for Common Voice, Rhapsodie, and the caches;
- a CUDA GPU is strongly recommended for extraction and alignment, but the
  heads can also run on CPU.

For Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Installation

From the `segmentation` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a CUDA machine, it may be preferable to first install PyTorch and Torchaudio
using the command corresponding to the machine's CUDA version, then install
the dependencies from the requirements file.

Some Pantagruel models may require Hugging Face authentication. In that case,
create an untracked `.env` file at the repository root:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Never publish this token.

## Corpus preparation

### Common Voice FR

The directory specified by `data.common_voice_root` must contain the Common
Voice TSV files and the `clips/` directory. The provided configuration uses:

```text
data/raw/cv-corpus-25.0-2026-03-09/fr/
  train.tsv
  dev.tsv
  test.tsv
  clips/
```

By default, the project limits training and validation with
`max_samples_train` and `max_samples_valid`. Set these values to `null` to use
the complete splits.

### Rhapsodie

Set the following paths in `configs/config_finetuning.yaml`:

```yaml
rhapsodie:
  audio_dir: data/raw/Rhap-corpus/sound_mp3
  annotations_dir: data/raw/Rhap-corpus/proso_textgrids/TextGrids-fev2013
  tier_name: periode
  boundary_convention: pause_midpoint
```

The base names of the audio and TextGrid files must allow the loader to match
them.

## Experimental configuration

The main file is `configs/config_finetuning.yaml`. It controls:

- Hugging Face identifiers and encoder dimensions;
- corpus paths and limits;
- construction of Gaussian pseudo-labels;
- Conv1D and BiLSTM dimensions and dropout;
- the loss function, learning rates, and early stopping;
- seeds;
- temporal tolerance and peak detection parameters;
- Rhapsodie annotation conventions.

The currently provided configuration selects the two Speech-Text models. To
reproduce the audio benchmark, replace the active lists with the audio
identifiers in the commented lines of the same file.

## Audio-only pipeline on Common Voice

### 1. Extract embeddings and create pseudo-labels

```bash
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split test
```

Outputs are saved as:

```text
data/cache/<encoder_name>_<actual_split_name>.pt
```

The `valid` CLI split maps to the Common Voice `dev` split by default and thus
produces a file with the `_dev.pt` suffix.

### 2. Train the audio head

```bash
python -m scripts.train --config configs/config_finetuning.yaml
```

To specify the seeds from the command line:

```bash
python -m scripts.train --config configs/config_finetuning.yaml --seeds 42 123 2024
```

### 3. Evaluate

```bash
python -m scripts.evaluate --config configs/config_finetuning.yaml
```

The script displays precision, recall, and F1 for each checkpoint, followed by
their mean and standard deviation when multiple seeds are available.

### 4. Analyze errors

```bash
python -m scripts.analyze_errors \
  --config configs/config_finetuning.yaml \
  --encoder Pantagruel-B-1K \
  --split test \
  --seed 42
```

The generated CSV contains predicted and reference boundaries, true positives,
false positives, and false negatives for each excerpt.

## Pipeline with text guidance

This pipeline must be run with a Pantagruel Speech-Text encoder that provides
the `encode_text` method.

### 1. Align words with audio

```bash
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split train
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split valid
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split test
```

WhisperX aligns the known Common Voice transcription with the signal. Caches
are saved in `data/cache/word_alignement/`. After any change to the alignment
logic, these files must be regenerated before recreating the text embeddings.

### 2. Extract audio and text representations

```bash
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split test
```

Each `<encoder_name>_<split>_text.pt` cache contains audio embeddings,
pseudo-labels, durations, and `text_features` with shape `(T, D)`.

### 3. Train and evaluate concatenation

```bash
python -m scripts.train_text_conc --config configs/config_finetuning.yaml --seeds 42 123 2024
python -m scripts.evaluate_text_conc --config configs/config_finetuning.yaml
```

### 4. Train and evaluate cross-attention

```bash
python -m scripts.train_text_cross --config configs/config_finetuning.yaml --seeds 42 123 2024
python -m scripts.evaluate_text_cross --config configs/config_finetuning.yaml
```

The two methods use distinct checkpoint names: `_text_conc.pt` and
`_text_cross.pt`. They can therefore be trained and compared without
overwriting the weights of the other architecture.

## Transfer to Rhapsodie

The following commands evaluate **audio-only** heads, including those
associated with a Speech-Text pretrained encoder. Explicit text guidance on
Rhapsodie is not implemented in this script.

```bash
python -m scripts.extract_embeddings_rhapsodie --config configs/config_finetuning.yaml
python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode frozen
python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode oracle
```

- `frozen` keeps the threshold selected on Common Voice: this is the actual
  transfer without recalibration;
- `oracle` sweeps several thresholds on Rhapsodie and provides an exploratory
  upper bound, not a strict zero-shot result.

## Main results

Full results are detailed in `reports/`. Across three seeds:

- the best Common Voice audio F1 is achieved by `LeBenchmark-w2v-L-7k`:
  `0.3773 +/- 0.0025`;
- the best Pantagruel result on Common Voice is achieved by
  `Pantagruel-Speech-Text-B-1K` used with audio only: `0.3562 +/- 0.0006`;
- the best frozen transfer to Rhapsodie is achieved by `Pantagruel-B-1K`:
  `0.4426 +/- 0.0154`;
- neither concatenation nor cross-attention improves F1 over audio-only use
  in the experiments conducted.

These values depend on the protocol, pseudo-labels, peak calibration, and cache
versions used. They do not constitute a general ranking of encoders.

## Reproducibility and considerations

- Use the same seeds, splits, sample limits, and caches for every comparison.
- Do not compare a regenerated text cache with baselines built from an older
  alignment without rerunning the affected conditions.
- Common Voice provides pause-derived pseudo-labels here, not expert prosodic
  annotations.
- Oracle F1 must not be presented as blind transfer performance.
- Hugging Face models loaded with `trust_remote_code=True` execute third-party
  code: verify the source and, for an archivable experiment, pin a model
  revision.
- The `src/old/` directories and some text analysis scripts are historical
  artifacts; the main entry points are those listed in the pipelines above.

## References

- P.-H. Le et al., [*Pantagruel: Unified Self-Supervised Encoders for French
  Text and Speech*](https://aclanthology.org/2026.lrec-1.799/), LREC 2026.
- S. Evain et al., [*LeBenchmark: A Reproducible Framework for Assessing
  Self-Supervised Representation Learning from
  Speech*](https://arxiv.org/abs/2104.11462), Interspeech 2021.
- A. Lacheret et al., [*Rhapsodie: a Prosodic-Syntactic Treebank for Spoken
  French*](https://aclanthology.org/L14-1329/), LREC 2014.
- M. Bain et al., [*WhisperX: Time-Accurate Speech Transcription of Long-Form
  Audio*](https://arxiv.org/abs/2303.00747), Interspeech 2023.

## License

The code specific to this repository is distributed under the MIT License; see
[`LICENSE`](LICENSE). This license does not automatically cover Common Voice,
Rhapsodie, downloaded checkpoints, Hugging Face models, or other third-party
resources. Each resource retains its own terms of use and redistribution.
