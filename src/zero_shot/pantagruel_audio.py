import os 
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from scipy.signal import  find_peaks
from  dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

class PantagruelSpeechModel:

    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token = HF_TOKEN 
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            token = HF_TOKEN
        )
        self.model.eval()

    def encode(self, audio, sr):
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.squeeze(0).cpu().numpy()

    def detect_boundaries(self, audio, sr):
        emb = self.encode(audio, sr)
        energy = np.linalg.norm(emb, axis=1)
        energy = np.convolve(
            energy,
            np.ones(5) / 5,
            mode="same"
        )

        diff = np.abs(np.diff(energy, prepend=energy[0]))

        peaks, _ = find_peaks(diff, prominence=0.1, distance=5)

        times = peaks * (len(audio) / sr) / len(diff)

        return times
