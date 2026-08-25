from pathlib import Path
from src.common.load_data import DataLoader 
from src.common.load_audio import load_audio 
import librosa 
import parselmouth

DATA = Path(__file__).parent.parent / 'data'/ 'raw'/ 'cv-corpus-25.0-2026-03-09'/ 'fr'

loader = DataLoader(DATA)

df = loader.load_data()

#print(f"Nombre d'exemples : {len(df)}")

sample = df.iloc[0]

sp,sr = load_audio(sample)


audio = parselmouth.Sound(sp, sampling_frequency=sr)
print(type(audio))
