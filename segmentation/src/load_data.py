import os
import pandas as pd 


class  DataLoader:
	def __init__(self , base_path):
		self.base_path = base_path 
		self.clips_path = os.path.join(base_path, 'clips')

	def  load_data(self, split ="train"):
		tsv = os.path.join(self.base_path, f'{split}.tsv')
		df = pd.read_csv(tsv,sep='\t')
		df['path']= df['path'].apply(lambda x : os.path.join(self.clips_path,x))
		return df 


