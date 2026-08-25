import os 
import torch
import numpy as np
from transformers import AutoModel , AutoProcessor ,AutoFeatureExtractor
from scipy.signal import find_peaks
from dotenv import load_dotenv



class LeBenchmarkSpeechModel:

	def __init__(self, model_name,device="cuda"):
		self.device= device
		self.model = AutoModel.from_pretrained(model_name).to(device)
#		self.processor= AutoProcessor.from_pretrained(model_name)
		self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
		self.model.eval()

	def encode(self, audio, sr):
		inputs = self.feature_extractor(audio, sampling_rate= sr, return_tensors ='pt')
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		with torch.no_grad():
			outputs = self.model(**inputs)
		return outputs.last_hidden_state.squeeze(0).cpu().numpy()

	def detect_boundaries(self,audio,sr):
		emb = self.encode(audio, sr)
		energy = np.linalg.norm(emb,axis=1)
		energy = np.convolve(energy, np.array([0.2,0.2,0.2,0.2,0.2]),mode="same")
		diff = np.diff(energy , prepend = energy[0])
		diff = np.abs(diff)
		peaks,_  = find_peaks(diff, prominence = 0.1, distance =5)
		times = peaks*(len(audio)/sr)/len(diff)
		return times



