import pandas as pd
import librosa
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'cv-corpus-25.0-2026-03-09' / 'fr'
df = pd.read_csv(DATA_DIR / 'train.tsv', sep='\t')
df['path'] = df['path'].apply(lambda x: str(DATA_DIR / 'clips' / x))

audio_path = df['path'].iloc[0]
sentence = df['sentence'].iloc[0]

y, sr = librosa.load(audio_path, sr=None)

print(f"Phrase: {sentence}")
print(f"Duree : {len(y)/sr:.2f} secondes")
print(f"Sample rate : {sr} Hz")
print(f"Nombre de samples : {len(y)}")
