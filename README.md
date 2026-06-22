# Fine-tuning d'une tête de segmentation prosodique (encodeur gelé)

Cette semaine,l'objectif etait d'entraîner une tête de segmentation (`ProsodicSegmentationHead`)
au-dessus d'un encodeur audio gelé (Pantagruel-B-1K), sur Common Voice FR,
pour détecter les frontières prosodiques. Il prolonge le travail zero-shot
existant (`src/zero_shot/`) en transformant les heuristiques de détection
de pics en labels d'entraînement supervisé.

## Structure

```
configs/
  config_finetuning.yaml      # tous les hyperparamètres et chemins
src/
  common/
    config.py               # chargement de la config YAML
    load_data.py             #  chargement Common Voice
    load_audio.py            #  chargement audio
    extract_features_v1.py   #  RMS, F0, pauses
  fine_tuning/
    segmentaiton_dataset.py               # Dataset + collate_fn (padding dynamique)
    head_segmentation.py     # tête de segmentation (corrigée, gère le masque)
    generate_labels.py       # pauses donnent  labels continus (gaussiennes)
    metrics.py                # F1 avec tolérance + extraction de pics
  zero_shot/
    pantagruel_audio.py      #  wrapper encodeur Pantagruel
    segmentation_prosodic_v1.py  #  baseline heuristique
    run_pantagruel_benchmark.py  #  benchmark
scripts/
  extract_embeddings.py     # encode tout un split + génère les labels -> cache .pt
  train.py                   # entraîne la tête sur le cache
  
data/cache/                  # embeddings + labels mis en cache (.pt par split)
checkpoints/                 # meilleurs modèles sauvegardés
```

## Pipeline complet

1. **Extraction des embeddings et labels** (une fois par split, mis en cache) :

```bash
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split train
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split valid
python -m scripts.extract_embeddings --config configs/config_finetuning.yaml --split test
```

   Ceci charge Common Voice, encode chaque audio avec Pantagruel-B-1K (gelé),
   genere les labels continus (gaussiennes centrees sur les pauses detectees),
   et sauvegarde tout dans `data/cache/Pantagruel-B-1K_<split>.pt`.

2. **Entraînement de la tête de segmentation** :

```bash
python -m scripts.train.py --config configs/config_finetuning.yaml
```

   Charge le cache, entraîne avec early stopping, sauvegarde le meilleur
   checkpoint dans `checkpoints/best_segmentation_head.pt`.



