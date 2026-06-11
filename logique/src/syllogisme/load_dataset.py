"""
Loading FOLIO dataset from HuggingFace
Saving data in  JSON.
"""
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import json, os
from pathlib import Path

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data/syllogisme_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
login(token=HF_TOKEN)
ds = load_dataset("yale-nlp/FOLIO", token=HF_TOKEN)

for split in ["train", "validation"]:
    if split in ds:
        data = [ex for ex in ds[split]]
        out_path = OUTPUT_DIR / f"FOLIO_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
