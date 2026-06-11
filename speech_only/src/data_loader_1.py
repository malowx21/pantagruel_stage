from torch.utils.data import IterableDataset

class VoxPopuliDataset(IterableDataset):

    def __init__(self, hf_dataset, sr=16000):
        self.dataset = hf_dataset
        self.sr = sr

    def decode_audio(self, path):
        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-i", path,
            "-ar", str(self.sr),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1"
        ]

        audio_bytes = subprocess.check_output(cmd)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        return audio

    def __iter__(self):
        for sample in self.dataset:

            audio_path = sample.get("audio", {}).get("path", None)
            if audio_path is None:
                continue

            audio = self.decode_audio(audio_path)

            yield {
                "audio": audio,
                "sampling_rate": self.sr,
                "text": sample.get("sentence", ""),
                "id": sample.get("id", "")
            }
