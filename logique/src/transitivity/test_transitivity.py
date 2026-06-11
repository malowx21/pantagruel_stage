import json
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

name = "PantagrueLLM/text-base-wiki-mlm"
config = AutoConfig.from_pretrained(name, trust_remote_code=True)

if not hasattr(config, "is_decoder"):
    config.is_decoder = False
if getattr(config, "vocab_size", None) is None:
    config.vocab_size = None

tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

if getattr(config, "vocab_size", None) is None and getattr(tokenizer, "vocab_size", None) is not None:
    config.vocab_size = tokenizer.vocab_size

model = AutoModelForMaskedLM.from_pretrained(name,config=config,trust_remote_code=True)

def pseudo_log_likelihood(text):
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask", None)
    total_score = 0
    seq_len = input_ids.size(1)
    for i in range(1, seq_len - 1):
        if attention_mask is not None and attention_mask[0, i] == 0:
            continue
        masked_input = input_ids.clone()
        true_token = masked_input[0, i].item()
        masked_input[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(input_ids=masked_input)
        logits = outputs.logits
        if logits.ndim == 3:
            token_logits = logits[0, i]
        elif logits.ndim == 2:
            token_logits = logits[i]
        else:
            raise RuntimeError(f"Logits tensor a une forme inattendue: {logits.shape}")
        log_probs = torch.log_softmax(token_logits, dim=-1)
        score_token = log_probs[true_token]
        total_score += float(score_token)

    return total_score


def predict(example):   
    premise = example["premise"]
    scores = []
    for candidate in example["choices"]:
        sentence = (premise+ "\n\n"+ f"La personne la plus grande est {candidate}.")
        score = pseudo_log_likelihood(sentence)
        scores.append(score)
    best_idx = scores.index(max(scores))
    return example["choices"][best_idx]


def evaluate(dataset_path):
    correct = 0
    total = 0
    with open(dataset_path, "r") as f:
        for line in f:
            example = json.loads(line)
            pred = predict(example)
            gold = example["answer"]
            print(f"ID={example['id']} / pred={pred} / gold={gold}")
            if pred == gold:
                correct += 1
            total += 1
    accuracy = correct / total
    print()
    print(f"Accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    evaluate("./data/data_transitivity.jsonl")