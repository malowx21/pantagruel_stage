import librosa


def load_audio(path,sr=16000):
	sp, sr = librosa.load(path,sr= sr)
	return sp,sr

