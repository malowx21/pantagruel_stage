import os
import torch
import numpy as np
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


class PantagruelSpeechTextAudioModel:

    def __init__(self, model_name, normalize=False):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self.model.eval()
        self.normalize = normalize

    def encode(self, audio, sr):
        if sr != 16000:
            raise ValueError(
                f"Le modèle attend du 16kHz, reçu sr={sr}. "
                f"Rééchantillonner l'audio avant l'appel à .encode()."
            )

        wav = np.asarray(audio, dtype=np.float32)
        if self.normalize:
            mean, std = wav.mean(), wav.std()
            if std > 1e-8:
                wav = (wav - mean) / std

        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, num_samples)

        with torch.no_grad():
            outputs = self.model(
                input_values=wav_tensor,
                input_ids=None,
                padding_mask=None,
                mode="AUDIO",
                mask=False,
                return_dict=True,
            )

        return outputs.audio_output.last_hidden_state.squeeze(0).cpu().numpy()


