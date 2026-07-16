import os
import pandas as pd
from pathlib import Path

class  DataLoader:
	def __init__(self , base_path):
		self.base_path = base_path 
		self.clips_path = os.path.join(base_path, 'clips')

	def  load_data(self, split ="train"):
		tsv = os.path.join(self.base_path, f'{split}.tsv')
		df = pd.read_csv(tsv,sep='\t')
		df['path']= df['path'].apply(lambda x : os.path.join(self.clips_path,x))
		return df 


class RhapsodieDataLoader:

	def __init__(self,audio_path , annotations_path): 
		self.audio_path = Path(audio_path)
		self.annotations_path = Path(annotations_path)

	def load_data(self) :
		row = []
		for  file_path in sorted(self.audio_path.glob("*.mp3")):
			file_id = file_path.stem
			textgrid_path = self.annotations_path  / f"{file_id}.TextGrid"
		
			row.append({"path": str(file_path),
			    "textgrid_path" : str(textgrid_path),
			    "file_id" : file_id})

		df = pd.DataFrame(row)

		return df 



