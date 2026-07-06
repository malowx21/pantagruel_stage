import numpy as np 
from scipy.signal import find_peaks



def match_boundaries(pred, ref, tol=0.25):
    """
    Match predicted boundaries to reference boundaries within a tolerance.
    """
    tp = []
    fp = []
    matched_ref = set()

    for p in pred:
        match_idx = None
        for i, r in enumerate(ref):
            if i not in matched_ref and abs(p - r) <= tol:
                match_idx = i
                break

        if match_idx is None:
            fp.append(p)
        else:
            matched_ref.add(match_idx)
            tp.append((p, ref[match_idx]))

    fn = [r for i, r in enumerate(ref) if i not in matched_ref]
    return tp, fp, fn



def evaluate(pred, ref, tol = 0.25):
    """
    Evaluate the different metrics by comparing the difference to a certain tolerance (tol)

    Args:
        pred : Predictions
        ref : Real values 
        tol : Tolerance of comparison . Defaults to 0.25.

    Returns:
        Metrics precsion, recall and F1 score 
    """
    tp_matches, fp_times, fn_times = match_boundaries(pred, ref, tol=tol)
    tp = len(tp_matches)
    fp = len(fp_times)
    fn = len(fn_times)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def logits_to_boundary_times(logits,duration,
    prominence= 0.1,
    min_distance_sec= 0.2,
    ) :
    """
    Transforms the logits, results of the neural network to time boundaries so we can compare them to the real ones 

    Args:
        logits : logits 
        duration : The total duration of the audio
        prominence : defaults to 0.1.
        min_distance_sec : defaults to 0.2.

    Returns:
        time stampes of the boundaries in second 
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

