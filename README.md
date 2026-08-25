# Segmentation prosodique du français par représentations audio et textuelles

Ce dépôt contient le pipeline expérimental développé pour comparer des
représentations auto-supervisées françaises sur une tâche de détection de
frontières prosodiques. Il couvre les encodeurs audio des familles
**Pantagruel** et **LeBenchmark**, les encodeurs multimodaux
**Pantagruel Speech-Text**, ainsi que deux mécanismes de guidage textuel :

- concaténation directe des représentations audio et texte alignées ;
- cross-attention, avec l'audio comme requête et le texte comme clé et valeur.

Les encodeurs sont gelés. Seule une tête de segmentation légère est entraînée
sur Common Voice FR. La généralisation inter-corpus est ensuite mesurée sans
réentraînement sur Rhapsodie, corpus de parole française annoté par des
experts.

> **Statut du projet :** prototype de recherche. Les chemins de données et les
> listes d'encodeurs doivent être adaptés dans la configuration avant de lancer
> une expérience.

## Objectifs

Le projet vise à :

1. comparer Pantagruel et LeBenchmark dans un protocole aval commun ;
2. mesurer l'effet du pré-entraînement Speech-Text sur la branche audio ;
3. déterminer si un texte explicitement fourni à la tête améliore le F1 ;
4. vérifier si les classements obtenus sur Common Voice se transfèrent à de la
   parole plus spontanée dans Rhapsodie ;
5. rendre les conclusions plus robustes grâce à plusieurs seeds et à une
   analyse des erreurs par fichier.

## Architectures

### Audio seul

```text
audio -> encodeur gelé -> embeddings (T, D)
      -> Conv1D -> BiLSTM -> MLP -> score de frontière par trame
```

### Concaténation directe

```text
audio -> embeddings audio alignés ----+
                                        +-> concaténation -> Conv1D -> BiLSTM -> MLP
texte -> embeddings de mots -> trames -+
```

Les vecteurs textuels sont répétés sur les trames temporelles couvertes par
chaque mot, puis concaténés aux vecteurs audio. La dimension d'entrée de la
tête passe ainsi de `D` à `2D`.

### Cross-attention

```text
Q = trames audio
K = trames textuelles alignées
V = trames textuelles alignées

audio + Attention(Q, K, V) -> Conv1D -> BiLSTM -> MLP
```

La cross-attention permet à chaque trame audio de pondérer l'information
textuelle pertinente. Sa projection de sortie est initialisée à zéro afin que
l'entraînement commence près du comportement audio seul.

Dans ces schémas, `T` désigne le nombre de trames temporelles et `D` la
dimension d'une représentation. Le MLP final produit un logit par trame. Selon
la configuration, l'apprentissage utilise une BCE avec logits ou une MSE, avec
masquage des positions de padding.

## Modèles comparés

- `Pantagruel-B-1K`
- `Pantagruel-B-14K`
- `Pantagruel-L-14K`
- `Pantagruel-Speech-Text-B-1K-v0`
- `Pantagruel-Speech-Text-B-1K`
- `LeBenchmark-w2v-B-1k`
- `LeBenchmark-w2v-L-7k`

Les champs `encoder.name` et `encoder.model_id` de la configuration sont deux
listes parallèles : ils doivent contenir le même nombre d'éléments et rester
dans le même ordre.

## Organisation du dépôt

```text
configs/
  config_finetuning.yaml              configuration principale

scripts/
  extract_embeddings.py               caches audio et pseudo-labels Common Voice
  word_alignment.py                   alignement mot--audio avec WhisperX
  extract_embeddings_speech_text.py   caches audio + texte aligné
  train.py                             entraînement audio seul
  train_text_conc.py                   entraînement par concaténation
  train_text_cross.py                  entraînement par cross-attention
  evaluate.py                          évaluation audio seul sur Common Voice
  evaluate_text_conc.py                évaluation de la concaténation
  evaluate_text_cross.py               évaluation de la cross-attention
  analyze_errors.py                    analyse FP/FN des têtes audio
  extract_embeddings_rhapsodie.py      création du cache Rhapsodie
  evaluate_rhapsodie.py                transfert frozen/oracle vers Rhapsodie

src/
  common/                              chargement, configuration et annotations
  fine_tuning/                         jeux de données, têtes, labels et métriques
  zero_shot/                           wrappers Pantagruel et LeBenchmark
  old/                                 prototypes antérieurs conservés pour référence

checkpoints/                           poids des têtes entraînées
reports/                               rapports et résultats expérimentaux
requirements.txt                      dépendances Python directes
LICENSE                               licence du code
```

Les données, caches d'embeddings et poids des encodeurs ne sont pas distribués
avec ce dépôt.

## Prérequis

- Python 3.10 ou plus récent ;
- `ffmpeg`, notamment pour le chargement des fichiers MP3 et WhisperX ;
- suffisamment d'espace disque pour Common Voice, Rhapsodie et les caches ;
- un GPU CUDA est vivement recommandé pour l'extraction et l'alignement, mais
  les têtes peuvent aussi être exécutées sur CPU.

Pour Ubuntu ou Debian :

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Installation

Depuis le dossier `segmentation` :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour une machine CUDA, il peut être préférable d'installer d'abord PyTorch et
Torchaudio avec la commande correspondant à la version CUDA de la machine,
puis d'exécuter l'installation du fichier de dépendances.

Certains modèles Pantagruel peuvent nécessiter une authentification Hugging
Face. Créer dans ce cas un fichier `.env` non versionné à la racine :

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Ne jamais publier ce jeton.

## Préparation des corpus

### Common Voice FR

Le dossier indiqué par `data.common_voice_root` doit contenir les fichiers TSV
de Common Voice et le dossier `clips/`. La configuration fournie utilise :

```text
data/raw/cv-corpus-25.0-2026-03-09/fr/
  train.tsv
  dev.tsv
  test.tsv
  clips/
```

Le projet limite par défaut l'entraînement et la validation avec
`max_samples_train` et `max_samples_valid`. Mettre ces valeurs à `null` pour
parcourir l'intégralité des splits.

### Rhapsodie

Renseigner les chemins suivants dans `configs/config_finetuning.yaml` :

```yaml
rhapsodie:
  audio_dir: data/raw/Rhap-corpus/sound_mp3
  annotations_dir: data/raw/Rhap-corpus/proso_textgrids/TextGrids-fev2013
  tier_name: periode
  boundary_convention: pause_midpoint
```

Les noms de base des fichiers audio et TextGrid doivent permettre leur
appariement par le chargeur.

## Configuration expérimentale

Le fichier principal est `configs/config_finetuning.yaml`. Il contrôle :

- les identifiants Hugging Face et la dimension des encodeurs ;
- les chemins et limites des corpus ;
- la construction des pseudo-labels gaussiens ;
- les dimensions Conv1D, BiLSTM et le dropout ;
- la fonction de perte, les learning rates et l'early stopping ;
- les seeds ;
- la tolérance temporelle et les paramètres de détection des pics ;
- les conventions d'annotation Rhapsodie.

La configuration actuellement fournie sélectionne les deux modèles Speech-Text.
Pour reproduire le benchmark audio, remplacer les listes actives par les
identifiants audio présents dans les lignes commentées du même fichier.

## Pipeline audio seul sur Common Voice

### 1. Extraire les embeddings et créer les pseudo-labels

```bash
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split test
```

Les sorties sont enregistrées sous la forme :

```text
data/cache/<nom_encodeur>_<nom_split_reel>.pt
```

Le split CLI `valid` correspond par défaut au split Common Voice `dev` et
produit donc un fichier suffixé par `_dev.pt`.

### 2. Entraîner la tête audio

```bash
python -m scripts.train --config configs/config_finetuning.yaml
```

Pour imposer les initialisations depuis la ligne de commande :

```bash
python -m scripts.train --config configs/config_finetuning.yaml --seeds 42 123 2024
```

### 3. Évaluer

```bash
python -m scripts.evaluate --config configs/config_finetuning.yaml
```

Le script affiche précision, rappel et F1 pour chaque checkpoint, puis leur
moyenne et leur écart-type lorsque plusieurs seeds sont disponibles.

### 4. Analyser les erreurs

```bash
python -m scripts.analyze_errors \
  --config configs/config_finetuning.yaml \
  --encoder Pantagruel-B-1K \
  --split test \
  --seed 42
```

Le CSV produit contient les frontières prédites et de référence, les vrais
positifs, faux positifs et faux négatifs de chaque extrait.

## Pipeline avec guidage textuel

Ce pipeline doit être exécuté avec un encodeur Pantagruel Speech-Text qui
dispose de la méthode `encode_text`.

### 1. Aligner les mots sur l'audio

```bash
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split train
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split valid
python -m scripts.word_alignment --config configs/config_finetuning.yaml --split test
```

WhisperX aligne la transcription Common Voice connue sur le signal. Les caches
sont enregistrés dans `data/cache/word_alignement/`. Après toute modification
de la logique d'alignement, ces fichiers doivent être régénérés avant de
recréer les embeddings textuels.

### 2. Extraire les représentations audio et textuelles

```bash
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings_speech_text --config configs/config_finetuning.yaml --split test
```

Chaque cache `<nom_encodeur>_<split>_text.pt` contient les embeddings audio,
les pseudo-labels, les durées et `text_features`, de forme `(T, D)`.

### 3. Entraîner et évaluer la concaténation

```bash
python -m scripts.train_text_conc --config configs/config_finetuning.yaml --seeds 42 123 2024
python -m scripts.evaluate_text_conc --config configs/config_finetuning.yaml
```

### 4. Entraîner et évaluer la cross-attention

```bash
python -m scripts.train_text_cross --config configs/config_finetuning.yaml --seeds 42 123 2024
python -m scripts.evaluate_text_cross --config configs/config_finetuning.yaml
```

Les deux méthodes utilisent des noms de checkpoints distincts :
`_text_conc.pt` et `_text_cross.pt`. Elles peuvent ainsi être entraînées et
comparées sans écraser les poids de l'autre architecture.

## Transfert vers Rhapsodie

Les commandes suivantes évaluent les têtes **audio seul**, y compris celles
associées à un encodeur pré-entraîné Speech-Text. Le guidage textuel explicite
sur Rhapsodie n'est pas implémenté dans ce script.

```bash
python -m scripts.extract_embeddings_rhapsodie --config configs/config_finetuning.yaml
python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode frozen
python -m scripts.evaluate_rhapsodie --config configs/config_finetuning.yaml --threshold_mode oracle
```

- `frozen` conserve la proéminence choisie sur Common Voice : c'est le vrai
  transfert sans recalibration ;
- `oracle` balaie plusieurs proéminences sur Rhapsodie et fournit une borne
  supérieure exploratoire, pas un résultat zero-shot strict.

## Résultats principaux

Les résultats complets sont détaillés dans `reports/`. Sur trois seeds :

- le meilleur F1 Common Voice audio est obtenu par
  `LeBenchmark-w2v-L-7k` : `0,3773 ± 0,0025` ;
- le meilleur Pantagruel sur Common Voice est
  `Pantagruel-Speech-Text-B-1K` utilisé avec l'audio seul :
  `0,3562 ± 0,0006` ;
- le meilleur transfert frozen vers Rhapsodie est obtenu par
  `Pantagruel-B-1K` : `0,4426 ± 0,0154` ;
- ni la concaténation ni la cross-attention n'améliorent le F1 par rapport à
  l'utilisation audio seule dans les expériences réalisées.

Ces valeurs dépendent du protocole, des pseudo-labels, de la calibration des
pics et des versions de caches utilisées. Elles ne constituent pas un
classement général des encodeurs.

## Reproductibilité et points d'attention

- Utiliser les mêmes seeds, splits, limites d'échantillons et caches pour toute
  comparaison.
- Ne pas comparer un cache textuel régénéré avec des baselines construites à
  partir d'un ancien alignement sans relancer les conditions concernées.
- Common Voice fournit ici des pseudo-labels dérivés des pauses, et non une
  annotation prosodique experte.
- Le F1 oracle ne doit pas être présenté comme une performance de transfert
  aveugle.
- Les modèles Hugging Face chargés avec `trust_remote_code=True` exécutent du
  code tiers : vérifier la source et, pour une expérience archivable, figer une
  révision du modèle.
- Les répertoires `src/old/` et certains scripts d'analyse textuelle sont des
  artefacts historiques ; les entrées principales sont celles listées dans les
  pipelines ci-dessus.

## Références

- P.-H. Le et al., [*Pantagruel: Unified Self-Supervised Encoders for French
  Text and Speech*](https://aclanthology.org/2026.lrec-1.799/), LREC 2026.
- S. Evain et al., [*LeBenchmark: A Reproducible Framework for Assessing
  Self-Supervised Representation Learning from
  Speech*](https://arxiv.org/abs/2104.11462), Interspeech 2021.
- A. Lacheret et al., [*Rhapsodie: a Prosodic-Syntactic Treebank for Spoken
  French*](https://aclanthology.org/L14-1329/), LREC 2014.
- M. Bain et al., [*WhisperX: Time-Accurate Speech Transcription of Long-Form
  Audio*](https://arxiv.org/abs/2303.00747), Interspeech 2023.

## Licence

Le code propre à ce dépôt est distribué sous licence MIT ; voir
[`LICENSE`](LICENSE). Cette licence ne couvre pas automatiquement Common Voice,
Rhapsodie, les checkpoints téléchargés, les modèles Hugging Face ni les autres
ressources tierces. Chaque ressource conserve ses propres conditions
d'utilisation et de redistribution.
