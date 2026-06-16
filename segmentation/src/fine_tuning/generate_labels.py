import numpy as np 

def get_ground_truth(features, duration, min_pause_duration = 0.25, include_utterance_end=True):
    """
    Extracting boundaries (in seconds) from pauses detected in the feature extraction

    Args:
        duration : total duration of the audio 
        include_utterance_end : add a boundary at the end 
    Returns:
        _type_: _description_
    """
    gt = []
    for s, e in features["pauses"]:
        if (e - s)>= min_pause_duration:
            gt.append((s+e)/2)
        
    if include_utterance_end : 
        gt.append(duration)
        
    return sorted(gt)


def boudaries_to_labels(boundaries_sec, num_frames, duration, sigma_frames):
    """
    Converts the list of boundaries to a continuous score for each frame
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
        gauss =(1/(np.sqrt(2*np.pi)*sigma_frames)) * np.exp(-0.5*((frame_idx- (b* frames_per_sec)) / sigma_frames )**2 )
        labels = np.maximum(labels, gauss)
        
    return labels


def generate_labels(features,duration, num_frames):  #TODO add config file in common where i put all the configurations (for example min_pause_duration )
    """
    Pipeline 
    """
    
    boundaries= get_ground_truth(features, duration,min_pause_duration=0.25,include_utterance_end=True)
    labels = boudaries_to_labels(boundaries, duration, num_frames, 1.5)
    return labels 