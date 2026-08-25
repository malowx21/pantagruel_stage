import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from scipy.signal import  find_peaks

class PantagruelAudioSegmenter:

    def __init__(self, model_name):

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model.eval()

    def extract_embeddings(self, audio, sr):

        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # embeddings temporels
        emb = outputs.last_hidden_state.squeeze(0)

        return emb.numpy()

    def detect_boundaries(self, audio, sr, prominence=0.5):

        emb = self.extract_embeddings(audio, sr)

        # convert embeddings  energy signal
        energy = np.linalg.norm(emb, axis=1)

        # smoothing
        energy = np.convolve(
            energy,
            np.ones(5)/5,
            mode="same"
        )

        # boundary = sudden changes
        diff = np.abs(np.diff(energy, prepend=energy[0]))

        peaks, _ = find_peaks(diff, prominence=prominence)

        # conversion temps
        times = peaks * (len(audio)/sr) / len(diff)

        return peaks, times, diff
