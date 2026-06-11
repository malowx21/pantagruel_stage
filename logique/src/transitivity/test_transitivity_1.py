import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


name = "PantagrueLLM/text-base-wiki-mlm"
config = AutoConfig.from_pretrained(name, trust_remote_code=True)

if not hasattr(config, "is_decoder"):
    config.is_decoder = False

tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True
)
model.eval()

def pseudo_log_likelihood(text):
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask", None)
    # ensure model has a mask token
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer has no mask token id. Use an MLM-capable tokenizer/model.")

    # move tensors to model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    total_score = 0.0
    seq_len = input_ids.size(1)

    for i in range(1, seq_len - 1):
        if attention_mask is not None and attention_mask[0, i] == 0:
            continue

        masked_input = input_ids.clone()
        true_token = int(masked_input[0, i].item())

        masked_input[0, i] = mask_id

        with torch.no_grad():
            outputs = model(input_ids=masked_input, attention_mask=attention_mask)

        logits = outputs.logits
        # logits shape expected: [batch, seq_len, vocab_size]
        token_logits = logits[0, i]

        log_probs = torch.log_softmax(token_logits, dim=-1)

        total_score += float(log_probs[true_token])

    return total_score



# Dataset
dataset = load_dataset("facebook/xnli", "fr")
print("Dataset splits:", list(dataset.keys()))

# choose a validation split robustly
for candidate in ("validation_matched", "validation", "validation_unmatched"):
    if candidate in dataset:
        split_name = candidate
        break
else:
    # fallback to first split available
    split_name = list(dataset.keys())[0]

# on prend le français
split_ds = dataset[split_name]
print("Columns:", split_ds.column_names)
if "language" in split_ds.column_names:
    data = split_ds.filter(lambda x: x["language"] == "fr")
    print(f"Using split: {split_name}; {len(data)} examples after language filter")
else:
    data = split_ds
    print(f"Using split: {split_name}; no 'language' column — {len(data)} examples (no language filter applied)")

label_map = {0: "entailment",1: "neutral",2: "contradiction"}

def predict(example):

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    scores = {}
    for label_id, label_text in label_map.items():
        text = premise + " " + hypothesis + " " + label_text
        score = pseudo_log_likelihood(text)
        scores[label_id] = score

    return max(scores, key=scores.get)

#
# EVALUATION
def evaluate(data):

    correct = 0
    total = 0

    for ex in data:
        pred = predict(ex)
        gold = ex["label"]
        print(f"pred={pred} | gold={gold}")

        if pred == gold:
            correct += 1
        total += 1
    acc = correct / total
    print("\n===================")
    print(f"Accuracy = {acc:.4f}")

# main 
if __name__ == "__main__":
    evaluate(data)