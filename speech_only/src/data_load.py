
from datasets import load_dataset
import subprocess
import numpy as np

ds = load_dataset("facebook/voxpopuli", "fr", streaming=True)

def decode_audio(path, sr=16000):

    """
    Decode audio using ffmpeg (no torchcodec needed)
    Returns: numpy array float32

    """

    cmd = [

        "ffmpeg",
        "-loglevel", "quiet",
        "-i", path,
        "-ar", str(sr),
        "-ac", "1",
        "-f", "f32le",
        "pipe:1"

    ]

    audio_bytes = subprocess.check_output(cmd)
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    return audio

