import json
import torch
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from collections import defaultdict

load_dotenv()

DATA_DIR    = Path(__file__).parent.parent.parent / "data/syllogisme_data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data/results_syllogisme"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "CamemBERT-B-Wk":      "camembert/camembert-base-wikipedia-4gb",
    "CamemBERT-B-OSC":     "camembert/camembert-base-oscar-4gb",
    "Pantagruel-B-Wk-MLM": "PantagrueLLM/text-base-wiki-mlm",
}

LABEL_TO_TEXT = {
    "True":  "correct",
    "False": "incorrect",
}
LABELS = list(LABEL_TO_TEXT.keys())


def build_prompt(premises_fr, conclusion_fr, mask_token):
    """
    Prompt structure avec [MASK] a la place du label.
    """
    premises_oneline = premises_fr.replace("\n", " ")
    return (
        f"Premisses : {premises_oneline} "
        f"Conclusion : {conclusion_fr} "
        f"La conclusion decoule-t-elle logiquement des premisses ? "
        f"Reponse : {mask_token}."
    )


def compute_label_score(model, tokenizer, premises_fr, conclusion_fr, label_text, device):
    """
    Place [MASK] dans le prompt et mesure P(label | contexte).
    Une seule passe forward, pas de biais de longueur.
    """
    prompt = build_prompt(premises_fr, conclusion_fr, tokenizer.mask_token)

    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encoding["input_ids"][0]

    # Position du [MASK]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        print(f"[WARN] [MASK] introuvable dans le prompt")
        return float("-inf")
    mask_pos = mask_positions[0].item()

    # Token du label
    label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    if len(label_ids) != 1:
        print(f"[WARN] multi-token label: '{label_text}' -> {label_ids}")
        return float("-inf")
    label_id = label_ids[0]

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
    )
        logits = outputs.logits

    # Gestion des différentes formes de logits
        if logits.ndim == 3:
            token_logits = logits[0, mask_pos]
        elif logits.ndim == 2:
            token_logits = logits[mask_pos]
        else:
            return float("-inf")

    # Vérification que le vocabulaire est valide
        if token_logits.shape[0] == 0 or label_id >= token_logits.shape[0]:
            return float("-inf")

        log_probs = torch.log_softmax(token_logits, dim=-1)

    return log_probs[label_id].item()

def evaluate_model(model_name, model_key):
    data_path = DATA_DIR / "FOLIO_fr.json"

    if not data_path.exists():
        print(f"Fichier introuvable : {data_path}")
        return {}

    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)

    # Filtre binaire : on garde uniquement True et False
    examples = [ex for ex in examples if ex["label"] in ["True", "False"]]

    print(f"\nModel : {model_name}")
    print(f"Exemples apres filtre binaire : {len(examples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
    config    = AutoConfig.from_pretrained(model_key, trust_remote_code=True)

    if not hasattr(config, "is_decoder"):
        config.is_decoder = False

    model = AutoModelForMaskedLM.from_pretrained(
        model_key,
        config=config,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Verification tokenisation des labels
    print("Verification tokenisation des labels :")
    for label, text in LABEL_TO_TEXT.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        status = "OK" if len(ids) == 1 else f"WARN {len(ids)} tokens"
        print(f"  '{text}' -> {ids} [{status}]")

    results      = []
    label_counts = defaultdict(int)
    correct      = 0

    for i, ex in enumerate(tqdm(examples, desc=f"{model_name}")):
        premises_fr   = ex["premises_fr"]
        conclusion_fr = ex["conclusion_fr"]
        true_label    = ex["label"]

        scores = {}
        for label in LABELS:
            scores[label] = compute_label_score(
                model, tokenizer,
                premises_fr, conclusion_fr,
                LABEL_TO_TEXT[label],
                device,
            )

        predicted_label = max(scores, key=scores.get)
        is_correct      = (predicted_label == true_label)

        if is_correct:
            correct += 1
        label_counts[true_label] += 1

        # Debug sur les 3 premiers exemples
        if i < 3:
            print(f"\n[DEBUG ex {i}]")
            print(f"  Premisses  : {premises_fr[:80]}...")
            print(f"  Conclusion : {conclusion_fr[:80]}")
            for label in LABELS:
                marker = ">" if label == true_label else " "
                print(f"  {marker} [{label:5}] {LABEL_TO_TEXT[label]:10} -> {scores[label]:.4f}")
            print(f"  Predit : {predicted_label} | Vrai : {true_label} {'OK' if is_correct else 'ERREUR'}")

        results.append({
            "story_id":        ex["story_id"],
            "example_id":      ex["example_id"],
            "true_label":      true_label,
            "predicted_label": predicted_label,
            "scores":          scores,
            "correct":         is_correct,
        })

    accuracy = correct / len(examples) if examples else 0.0
    majority = max(label_counts.values()) / len(examples)

    print(f"\n── Resultats globaux ──────────────────")
    print(f"  Accuracy      : {correct}/{len(examples)} = {accuracy:.1%}")
    print(f"  Baseline      : {majority:.1%} (majority class)")
    print(f"  Distribution  : {dict(label_counts)}")

    return {
        "model":        model_name,
        "accuracy":     accuracy,
        "n_examples":   len(examples),
        "n_correct":    correct,
        "label_counts": dict(label_counts),
        "details":      results,
    }


if __name__ == "__main__":
    all_results = {}

    for model_name, model_key in MODELS.items():
        res = evaluate_model(model_name, model_key)

        if res:
            all_results[model_name] = res

        out_path = RESULTS_DIR / f"results_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nSauvegarde : {out_path}")

    # Resume comparatif
    print("\n" + "="*55)
    print("RESUME COMPARATIF")
    print("="*55)
    print(f"{'Modele':<25} {'Accuracy':>10} {'N':>6} {'Baseline':>10}")
    print("-"*55)
    for model_name, res in all_results.items():
        majority = max(res["label_counts"].values()) / res["n_examples"]
        print(
            f"{model_name:<25} "
            f"{res['accuracy']:>9.1%} "
            f"{res['n_examples']:>6} "
            f"{majority:>9.1%}"
        )
