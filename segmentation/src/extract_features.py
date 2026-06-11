import numpy as np
import librosa
import parselmouth


def extract_rms(audio, frame_length=2048, hop_length=512):
	rms =  librosa.feature.rms(y=audio,frame_length=frame_length, hop_length= hop_length)[0]
	return rms

def extract_f0(audio,sr=16000,time_step=0.01,pitch_floor=75,pitch_ceiling=600):
	sound = parselmouth.Sound(audio,sampling_frequency=sr) 
	pitch = sound.to_pitch(time_step=time_step,pitch_floor=pitch_floor,pitch_ceiling=pitch_ceiling)
	f0 =  pitch.selected_array['frequency']
	return f0

def extract_pauses(audio, sr, top_db=30):
	non_pause_interval =  librosa.effects.split(audio,top_db= top_db)
	pause_interval =[]
	if len(non_pause_interval) < 2:
		return pause_interval
	for i  in range(len(non_pause_interval)-1):
		pause_interval.append((non_pause_interval[i][1]/sr,non_pause_interval[i+1][0]/sr))
	return pause_interval

def extract_features(audio,sr):
	rms = extract_rms(audio)
	f0 = extract_f0(audio)
	pause  =  extract_pauses(audio,sr)

	return {'rms': rms, 'f0': f0, 'pause': pause}


