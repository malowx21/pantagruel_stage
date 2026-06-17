import torch 
from torch.utils.data import Dataset

class SegmentationData(Dataset):
    """
    Dataset for prosodic segmentation
    
    embeddings : list of tensors (for each sample )
    labels : list of tensors (continuous score for each frame)
    durations : list of floats (real duration of audio )
    """
    
    def __init__(self,embeddings,labels, durations):
        
        if len(labels)!= len(embeddings):
            raise ValueError("Boths lists should have same size")
        self.embeddings = embeddings
        self.labels = labels
        self.durations = durations 
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        
        embedding = self.embeddings[index].float()
        label = self.labels[index].float()
        duration = self.duration[index]
        return embedding , label, duration 
    

def collate_fn(batch):
    
    """
    managing the padding dynamically 
    """
    embeddings, labels, durations  = zip(*batch)
    
    l = torch.tensor([emb.shape[0] for emb in embeddings])
    max_l = int(l.max().item())
    dim = embeddings[0].shape[1]
    
    embeddings_pad = torch.zeros(len(batch),max_l,dim,dtype=torch.float32)
    labels_pad = torch.zeros(len(batch), max_l, dtype=torch.float32)
    
    for j , (emb,lab) in enumerate(zip(embeddings,labels)) :
        sh= emb.shape[0]
        embeddings_pad[j,:sh,:]= emb
        labels_pad[j,:sh]=lab
        
    if durations[0] is None:
        durations_tensor = None
    else:
        durations_tensor = torch.tensor(durations, dtype=torch.float32)
        
    return embeddings_pad, labels_pad, l, durations_tensor