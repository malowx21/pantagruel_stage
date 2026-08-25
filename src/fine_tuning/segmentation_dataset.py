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
        duration = self.durations[index]
        return embedding , label, duration 


class SegmentationDataText(SegmentationData):
    """
    Same as SegmentationData, but also carries the frame-aligned text
    stream produced by src.fine_tuning.text_alignment.align_text_to_frames.
 
    Only used for encoders that support encode_text (currently the
    Pantagruel Speech-Text variants). For any other encoder, keep using
    the base SegmentationData + collate_fn, unchanged.
 
    text_features : list of tensors (T_i, D), same T_i as the
        corresponding embeddings[i], and same D (see extract_embeddings.py)
    """
 
    def __init__(self, embeddings, labels, durations, text_features):
        super().__init__(embeddings, labels, durations)
        if len(text_features) != len(embeddings):
            raise ValueError("text_features must have the same size as embeddings")
        self.text_features = text_features
 
    def __getitem__(self, index):
        embedding, label, duration = super().__getitem__(index)
        text = self.text_features[index].float()
        return embedding, label, duration, text

def collate_fn(batch):
    
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


def collate_fn_text(batch):
    """
    Same batching logic as collate_fn, plus padding for the text stream.
 
    Reuses collate_fn itself (rather than duplicating the padding logic)
    by stripping the text element out, delegating to collate_fn for the
    audio/label/duration part, then padding text_features the same way.
    """
    base_batch = [(emb, lab, dur) for emb, lab, dur, _ in batch]
    embeddings_pad, labels_pad, l, durations_tensor = collate_fn(base_batch)
 
    texts = [txt for _, _, _, txt in batch]
    max_l = int(l.max().item())
    dim = texts[0].shape[1]
 
    text_pad = torch.zeros(len(batch), max_l, dim, dtype=torch.float32)
    for j, txt in enumerate(texts):
        sh = txt.shape[0]
        text_pad[j, :sh, :] = txt
 
    return embeddings_pad, labels_pad, l, durations_tensor, text_pad
