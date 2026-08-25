import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer 
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,token = HF_TOKEN)

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

    def encode_text(self, words):
        """
        Encode a list of words using the TEXT mode of the Pantagruel
        Speech-Text encoder, then pool the (possibly sub-word) token
        embeddings back to one vector per word.
 
        Using `is_split_into_words=True` with the exact word list coming
        from the forced-alignment step (rather than re-tokenizing the raw
        sentence) guarantees that word i here corresponds exactly to
        word i in the (word, start_sec, end_sec) alignment, so the two
        can be zipped together downstream without any risk of mismatch
        (e.g. due to punctuation handling differing between the two
        tokenizations).
 
        Args:
            words: list of word strings, in order -- typically the first
                element of each (word, start, end) tuple produced by
                generate_word_alignment.py
 
        Returns:
            np.ndarray of shape (num_words, hidden_dim) -- one embedding
            per word, in the same representation space as the audio
            embeddings returned by .encode()
        """
        hidden_dim = self.model.config.hidden_size
 
        if len(words) == 0:
            return np.zeros((0, hidden_dim), dtype=np.float32)
 
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
 
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding.get("attention_mask"),
                padding_mask=None,
                mode="TEXT",
                mask=False,
                return_dict=True,
            )
 
        # (T_tokens, D) -- one row per sub-word token, not per word yet
        token_embeddings = outputs.text_output.last_hidden_state.squeeze(0)
 
        # Maps each token position back to the index of the original word
        # it belongs to (None for special tokens such as [CLS]/[SEP]).
        word_ids = encoding.word_ids(batch_index=0)
 
        pooled = []
        for word_idx in range(len(words)):
            token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if not token_positions:
                # The tokenizer dropped this word entirely -- rare, but we
                # fall back to a zero vector rather than letting the
                # index mapping between words and embeddings shift.
                pooled.append(torch.zeros(hidden_dim))
                continue
            pooled.append(token_embeddings[token_positions].mean(dim=0))
 
        return torch.stack(pooled).cpu().numpy()
