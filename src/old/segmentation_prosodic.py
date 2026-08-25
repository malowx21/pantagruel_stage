import numpy as np
from scipy.signal import find_peaks


class ProsodicSegmenter:
	"""
	Prosodic boundary detection using continous scoring 
	features : RMS energy variation / F0 / Pause duration 

	"""
	def __init__(self, sr= 16000,hop_length=512):
		self.sr = sr
		self.hop_length= hop_length
				
		#weights
		self.w_rms = 0.4
		self.w_f0 = 0.3
		self.w_pause= 0.3


	def normalize(self,x):
		x = np.array(x)
		return   (x - np.mean(x))/np.std(x)


	def rms_feature(self,rms):
		rms = self.normalize(rms)
		rms = np.convolve(rms,np.array([0.2,0.2,0.2,0.2,0.2]),mode ='same')
		drms = np.diff(rms ,prepend = rms[0])
		return -drms  # ici je retourne le moins pour avoir un pic lors de la chute d'energie

	def f0_feature(self,f0):
		f0  = np.array(f0)
		f0 = np.where(f0==0,np.nan,f0)
		idx = np.arange(len(f0))
		valid = ~np.isnan(f0)

		f0_interp= np.interp(idx,idx[valid],f0[valid])
		f0_norm = self.normalize(f0_interp)
		df0 = np.diff(f0_norm, prepend=f0_norm[0])
		return np.abs(df0)

	def pause_feature(self, pauses, length):
		pause_signal = np.zeros(length)
		for (start, end) in pauses:
			start_idx = int(start * self.sr / self.hop_length)
			end_idx = int(end * self.sr / self.hop_length)
			duration = end - start
			strength = np.clip(duration, 0, 1)
			pause_signal[start_idx:end_idx] = strength
		return pause_signal

	def compute_score(self, rms, f0, pauses):

        	L = len(rms)

        	rms_feat = self.normalize(self.rms_feature(rms))
        	f0_feat = self.normalize(self.f0_feature(f0))
        	pause_feat = self.pause_feature(pauses, L)

        	min_len = min(len(rms_feat), len(f0_feat), len(pause_feat))

        	score = (
            	self.w_rms * rms_feat[:min_len] +
            	self.w_f0 * f0_feat[:min_len] +
            	self.w_pause * pause_feat[:min_len]
        	)

        	return score

  
    	# boundary detection
	def detect_boundaries(self, rms, f0, pauses, prominence=0.5):

        	score = self.compute_score(rms, f0, pauses)

        	# smoothing (important pour prosodie)
        	score_smooth = np.convolve(
            			score,
            			np.ones(5) / 5,
            			mode="same"
        			)

        	# detection robuste des pics
        	peaks, _ = find_peaks(score_smooth, prominence=prominence)

        	# conversion en timestamps (secondes)
        	times = peaks * self.hop_length / self.sr

        	return peaks, times, score_smooth	
