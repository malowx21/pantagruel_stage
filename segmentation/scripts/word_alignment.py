import torch
import whisperx

from pathlib import Path
from argparse import ArgumentParser

from src.common.load_data import DataLoader
from src.common.load_audio import load_audio
from src.common.config import load_config

def load_align_model(lang, device):
    model, metadata = whisperx.load_align_model(language_code = lang , device= device)
    return model , metadata

def build_utterance_segment(sentence, duration):
    return[{"start": 0, "end":duration,"text":sentence}]

def align(segment,model,metadata, audio , device):
    result = whisperx.align(segment,model,metadata,audio,device)
    
    #result["segments"] list of dict
    words =[]
    for s in result["segments"] : 
        for w in s.get("words", []):
            if "start" not in w or "end" not in w :
                continue 
            words.append((w["word"],float(w["start"]),float(w['end'])))
    return words 

def process_split(config, split ,max_samples,  device):
    data_dir = Path(config["data"]["common_voice_root"])
    cache_dir = Path(config["data"]["cache_dir"]) / "word_alignement"
    cache_dir.mkdir(parents=True,exist_ok=True)
    
    loader = DataLoader(data_dir)
    df = loader.load_data(split= split) 
    if max_samples is not None : 
        df = df.head(max_samples)
        model , metadata= load_align_model("fr",device)
    alignements= {}
    for i , row in df.iterrows():
        audio, sr = load_audio(row["path"])
        duration = len(audio)/ sr
        sentence = row["sentence"]
        try :
            segments = build_utterance_segment(sentence=sentence,duration=duration)
            words = align(segments,model,metadata,audio, device)
        except Exception as e :
            print(f"align failed on {row['path']}: {e}")
            words=[]
        alignements[row["path"]]={"words":words,"duration":duration,"sentence":sentence}
    out_path = cache_dir / f"{split}_word_alignemnt.pt"
    torch.save(alignements,out_path)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config",type=str, default='configs/config_finetuning.yaml')
    parser.add_argument("--split",type=str,choices=["train","valid","test"],required=True)
    args = parser.parse_args()
        
    config =  load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    splits = {"train":"split_train", "valid":"split_valid", "test":"split_test" }
    max_samples = {"train": "max_samples_train","valid":"max_samples_valid","test":"max_samples_valid"}
    
    split = config["data"][splits[args.split]]
    max_sample = config["data"].get(max_samples[args.split])
    
    process_split(config,split, max_sample,device)
    
if __name__=="__main__":
    main()   
        
    
