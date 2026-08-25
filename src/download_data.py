from dotenv import load_dotenv
from datacollective import download_dataset
 
load_dotenv()
path  = download_dataset("cmn5zugst00w3nv07upovf2bg")

print(path)
