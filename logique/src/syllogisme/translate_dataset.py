"""
Translation FOLIO english to french with NLLB
NLLB = No Language Left Behind
Model : facebook/nllb-200-distilled-600M 
"""
 
import json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 
DATA_DIR   = Path(__file__).parent.parent.parent  / 'data/syllogisme_data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "facebook/nllb-200-distilled-600M"
 
# Codes de langue NLLB 
SRC_LANG = "eng_Latn" 
TGT_LANG = "fra_Latn"
 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Translation
def translate_text(text) :
    inputs = tokenizer(text,return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG), 
        max_length= 512
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




def translate_folio_example(example) :
    """
    Traduce an example of  a Folio dictionary. 
    Traduce the premiese, conclusion but does not traduce the label (True/ False/ Unknown)
    """
    translated = example.copy()
    premises_lines = example["premises"].strip().split("\n")
    translated_premises = []
    for line in premises_lines:
        line = line.strip()
        if line:
            translated_premises.append(translate_text(line))
    translated["premises_fr"] = "\n".join(translated_premises)

    # Traduction de la conclusion
    translated["conclusion_fr"] = translate_text(example["conclusion"])
    # On garde l'original pour référence
    translated["premises_en"]   = example["premises"]
    translated["conclusion_en"] = example["conclusion"]

    return translated


# Pipeline 
def translate_split(split_name):
    input_path  = DATA_DIR / f"FOLIO_{split_name}.json"
    output_path = DATA_DIR / f"FOLIO_{split_name}_fr.json"

    with open(input_path, encoding="utf-8") as f:
        examples = json.load(f)

    translated_examples = []
    for i, ex in enumerate(examples):
        # TODO need to access to GPU so that  I can translate all the sentences in the FOLIO dataset
        if i< 100:
            print(f"Traduction exemple {i+1}/{len(examples)}")
            translated = translate_folio_example(ex)
            translated_examples.append(translated)
        else : 
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated_examples, f, ensure_ascii=False, indent=2)


# Main
if __name__ == "__main__":
    for split in ["train", "validation"]:
        translate_split(split)

