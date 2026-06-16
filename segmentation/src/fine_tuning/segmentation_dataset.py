import torch 
from torch.utils.data import Dataset 

class SegmentationDateset(Dataset):
	def __init__(self, embeddings, labels):
		"""
		embeddings , labels lists of tensors 
		"""
		self.embeddings= embeddings
		self.labels= labels

	def __len__(self):
		return len(self.embeddings)
		
	def __getitem(self,idx):

		emb = self.embeddings[idx].float()
		lab = self.labels[idx].float()
		return emb, lab
		
