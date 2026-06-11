import json
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from collections import defaultdict


load_dotenv()


DATA_DIR = Path(__file__).parent.parent.parent / "data/syllogisme_data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "data/results_syllogisme"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Models to evaluate (maybe i'll add some models to compare with in the benchmark)
MODELS = {"CamemBERT-B-Wk": "camembert/camembert-base-wikipedia-4gb","CamemBERT-B-OSC":"camembert/camembert-base-oscar-4gb",
          "Pantagruel-B-Camtok-Wk" : "PantagrueLLM/text-base-camtok-wiki",
          #"Pantagruel-B-Wk-MLM" : "PantagrueLLM/text-base-wiki-mlm" ,
          #"Pantagruel-B-Crs-MLM":  "PantagrueLLM/text-base-croissant-mlm-old"
}

LABEL_TO_TEXT = {"True": "Vrai","False": "Non","Uncertain": "possible"}
LABELS = list(LABEL_TO_TEXT.keys())


def compute_pll(model, tokenizer,premises_fr,conclusion_fr, label_text, device) :
    """
    Calculate the Pseudo log likehood of a text for an MLM model
    """

    full_text = (f"{premises_fr.replace(chr(10), ' ')} "f"donc, {conclusion_fr.lower()} "f"C'est {label_text}.")
    model.eval()
    encoding = tokenizer(full_text,return_tensors="pt",truncation=True,max_length=512).to(device)
    input_ids = encoding['input_ids'][0]
    label_token_ids = tokenizer(label_text,add_special_tokens=False)["input_ids"]
    total_log_prob=0

    with torch.no_grad():
        for tok_id in label_token_ids:
            positions = (input_ids==tok_id).nonzero(as_tuple=True)[0]
            if len(positions) ==0:
                continue
            pos = positions[-1].item()

            masked_ids = input_ids.clone()
            # MLM
            masked_ids[pos] = tokenizer.mask_token_id
            output = model(input_ids = masked_ids.unsqueeze(0),attention_mask=encoding["attention_mask"])
            logits = output.logits
            if logits.ndim == 3:
                token_logits = logits[0, pos]

            elif logits.ndim == 2:
                token_logits = logits[pos]

            elif logits.ndim == 1:
                token_logits = logits

            else:
                raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

            log_prob = torch.log_softmax(token_logits,dim=-1)
            total_log_prob+= log_prob[tok_id].item()


            if len(label_token_ids) == 0:
                return 0.0
    return total_log_prob / len(label_token_ids)

def build_full_text(premises_fr, conclusion_fr, label_text):
    """
    Build the input sentence that we'll enter to the model
    """
    premises_oneline = premises_fr.replace("\n", " ")
    return (f"{premises_oneline} "f"donc, {conclusion_fr.lower()} "f"C'est {label_text}.")

def compute_label_score(model, tokenizer, premises_fr, conclusion_fr, label_text, device):
    """
    Met [MASK] à la place du label et mesure directement
    P(label | contexte) en une seule passe.
    """
    full_text = (
        f"{premises_fr.replace(chr(10), ' ')} "
        f"donc, {conclusion_fr.lower()} "
        f"C'est {tokenizer.mask_token}."
    )

    model.eval()
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    input_ids = encoding["input_ids"][0]

    # Position du [MASK]
    mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        return 0.0
    mask_pos = mask_pos[0].item()

    # Token du label
    label_token_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
    if len(label_token_ids) == 0:
        return 0.0
    label_tok = label_token_ids[0]  # on prend le premier token

    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        logits    = output.logits[0, mask_pos]
        if logits.shape[0] == 0 or label_tok >= logits.shape[0]:
            return float('-inf')   # score invalide
        log_probs = torch.log_softmax(logits, dim=-1)
        
    topk = torch.topk(log_probs, 20)

    print("\nTOP TOKENS:")
    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
        print(tokenizer.decode([idx]), val)
    return log_probs[label_tok].item()

def evaluate_model(model_name, model_key) :
    data_path = DATA_DIR / f"FOLIO_validation_fr.json"
    with open(data_path, encoding="utf-8") as f:
        examples = json.load(f)
    print(f" Model : {model_name} ")
    examples= examples[:3 ]
    device = "cuda" if torch.cuda.is_available() else "cpu" # TODO : need to test it with gpu
    tokenizer = AutoTokenizer.from_pretrained(model_key)
    config = AutoConfig.from_pretrained(model_key,trust_remote_code=True)

    if not hasattr(config, "is_decoder"):
        config.is_decoder = False

    if model_name[0]=="C":
        model = AutoModelForMaskedLM.from_pretrained(model_key).to(device)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_key,config=config,trust_remote_code=True).to(device)
    results = []
    label_counts = defaultdict(int)
    correct= 0

    for i, ex in enumerate(examples):
        premises_fr = ex["premises_fr"]
        conclusion_fr = ex["conclusion_fr"]
        true_label = ex["label"]

        # Compute the pseudo log likehood
        pll_scores = {}
        for label in LABELS:
            text = build_full_text(premises_fr,conclusion_fr,LABEL_TO_TEXT[label])
            pll_scores[label] = compute_label_score(model, tokenizer, premises_fr,conclusion_fr,LABEL_TO_TEXT[label], device)

        if i < 3:
            print(f"\n[DEBUG ex {i}]")
            # Affiche les scores
            for label in LABELS:
                print(f"  [{label:10}] score={pll_scores[label]:.4f}")
            # Affiche les tokens du label
            for label in LABELS:
                label_text = LABEL_TO_TEXT[label]
                label_token_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
                print(f"  [TOKEN] '{label_text}' → ids={label_token_ids} → tokens={[tokenizer.decode([t]) for t in label_token_ids]}")
            print(f" predit : {max(pll_scores, key=pll_scores.get)} | vrai : {true_label}")
        # Prediction
        predicted_label = max(pll_scores,key=pll_scores.get)
        # Check if our prediction is correct or not
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        label_counts[true_label] += 1

        results.append({
            "story_id": ex["story_id"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "pll_scores": pll_scores,
            "correct":  is_correct,
        })

    accuracy = correct / len(examples) if examples else 0
    print(f"\n Global results")
    print(f"  Accuracy  : {correct}/{len(examples)} = {accuracy:.1%}")
    print(f"  Distribution : {dict(label_counts)}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "n_examples":len(examples),
        "n_correct": correct,
        "label_counts": dict(label_counts),
        "details":results,
    }


# Main
if __name__ == "__main__":
    all_results = {}
    for model_name, model_key in MODELS.items():
        model_results = {}
        res = evaluate_model(model_name, model_key)
        if res :
            model_results['validation'] = res

        all_results[model_name] = model_results
        # Saving results 
        out_path = RESULTS_DIR / f"results_{model_name.replace(' ', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        print(f"\nResultats sauvegardes : {out_path}")


