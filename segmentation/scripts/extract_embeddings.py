"""
Extracting embeddings from the frozen encoder 
"""

import torch 
import argparse
from pathlib import Path 


from src.common.load_data import DataLoader
from src.common.load_audio import load_audio
from src.common.extract_features_v1 import extract_features
from src.zero_shot.pantagruel_audio import PantagruelSpeechModel
from src.fine_tuning.generate_labels import generate_labels

from src.common.config import load_config



def extract(config , split, max_samples):
        
    data_dir = Path(config['data']['common_voice_root'])
    cache_dir  = Path(config['data']['cache_dir'])
    cache_dir.mkdir(parents=True,exist_ok=True)
    
    loader = DataLoader(data_dir)
    df = loader.load_data(split= split)
    if max_samples is not None :
        df = df.head(max_samples)
        
    encoder = PantagruelSpeechModel(config['encoder']['model_id'])
    
    embeddings, labels , durations = [],[],[]
    
    for index , row  in df.itterows():
    
        audio , sr = load_audio(row['path'])
        features = extract_features(audio= audio, sr=sr)
        duration = len(audio)/ sr 
        
        embedding  = encoder.encode(audio, sr)
        num_frames = embedding[0]
        label = generate_labels(features,duration,num_frames,config)
        
        embeddings.append(torch.from_numpy(embedding))
        labels.append(torch.from_numpy(label))
        durations.append(torch.from_numpy(duration))
    
    out_path = cache_dir / f"{config['encoder']['name']}_{split}.pt"
    torch.save({"embeddings": embeddings,"labels": labels,"durations": durations},out_path  )
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config__finetuning.yaml')
    parser.add_argument('--split', type=str, choices=['train','valid','test'])
    args = parser.parse_args()
    config = load_config(args.config)
    
    split_key = {
        "train": "split_train",
        "valid": "split_valid",
        "test": "split_test",
    }
    max_key = {
        "train": "max_samples_train",
        "valid": "max_samples_valid",
        "test": "max_samples_valid",
    }
    
    split_name = config['data'][split_key[args.split]]
    max_samples = config['data'].get(max_key[args.split])
    
    extract(config, split=split_name,max_samples=max_samples)
    
if __name__ =="__main__":
    main()