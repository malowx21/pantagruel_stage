import numpy as np 

def get_ground_truth(features, duration, min_pause_duration = 0.25, include_utterance_end=True):
    """
    Extracting boundaries  from pauses detected in the feature extraction

    Args:
        feateures : dictionnary of all features, for instance ``pauses``
        duration : total duration of the audio 
        include_utterance_end : add a boundary at the end 
    Returns:
        gt (list) : list of the discrete points of pauses 
    """
    gt = []
    for s, e in features["pauses"]:
        if (e - s)>= min_pause_duration:
            gt.append((s+e)/2)
        
    if include_utterance_end : 
        gt.append(duration)
        
    return sorted(gt)


def boudaries_to_labels(boundaries_sec, num_frames,duration, sigma_frames):
    """
    Converts the list of boundaries to a continuous score for each frame

    Args:
        boundaries_sec : boundaaries in second
        num_frames : total number of frames 
        duration : the total duration  of the audio 
        sigma_frames : standard deviation  

    Returns:
        Returns an np.ndarray of continuous labels after applying a normal distribution 
    """
    
    if num_frames ==0 : 
        return np.zeros(0,dtype = np.float32)
    
    labels = np.zeros(num_frames,dtype=np.float32)
    frame_idx = np.arange(num_frames)
    
    if duration >  0 :
        frames_per_sec = num_frames / duration 
    else :
        frames_per_sec = 0
    
    
    for b in boundaries_sec:
        gauss =np.exp(-0.5*((frame_idx- (b* frames_per_sec)) / sigma_frames )**2 )
        labels = np.maximum(labels, gauss)
        
    return np.clip(labels,0.0,1.0)


def generate_labels(features,duration, num_frames, config):  
    
    boundaries= get_ground_truth(features, duration,min_pause_duration=config["labels"]["min_pause_duration"],include_utterance_end=config["labels"]["include_utterance_end"])
    labels = boudaries_to_labels(boundaries,num_frames,duration,sigma_frames=config['labels']['gaussian_sigma_frames'] )
    return labels 
