import numpy as np 
from scipy.signal import find_peaks




def evaluate(pred, ref, tol = 0.25):
    """_summary_

    Args:
        pred (_type_): _description_
        ref (_type_): _description_
        tol (float, optional): _description_. Defaults to 0.25.

    Returns:
        _type_: _description_
    """
    tp= 0
    matched = set()

    for p in pred : 
        for i, r in enumerate(ref):
            if i not in matched and abs(p-r) <= tol:
                matched.add(i)
                tp+=1
                break
    fp = len(pred)-tp
    fn = len(ref)-tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def logits_to_boundary_times(logits,duration,
    prominence= 0.1,
    min_distance_sec= 0.2,
    ) :
    """_summary_

    Args:
        logits (_type_): _description_
        duration (_type_): _description_
        prominence (float, optional): _description_. Defaults to 0.1.
        min_distance_sec (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    if len(logits) == 0:
        return []

    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    num_frames = len(probs)
    frames_per_sec = num_frames / duration if duration > 0 else 1.0
    distance_frames = max(1, int(min_distance_sec * frames_per_sec))

    peaks, _ = find_peaks(probs, prominence=prominence, distance=distance_frames) #peaks include the frame indexes of the peaks 
    times = peaks / frames_per_sec 
    return times.tolist()


