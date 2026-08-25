import librosa


def load_audio(path,sr=16000):
	"""
	Loading audio from the data

	Args:
		path : Path of the audio files 
		sr : Sampling rate . Defaults to 16000.

	Returns:
		sp : Audio time series
		sr : Sampling rate of sp 
	"""
	sp, sr = librosa.load(path,sr= sr)
	return sp,sr

