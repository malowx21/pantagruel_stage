import os
import torch
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


class PantagruelSpeechTextAudioModel:

    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self.model.eval()

    def encode(self, audio, sr):
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                mode="AUDIO",
                return_dict=True,
            )
        return outputs.audio_output.last_hidden_state.squeeze(0).cpu().numpy()