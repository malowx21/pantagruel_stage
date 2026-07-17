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


def sweep_peak_prominence(
    logits, duration, ref_times, prominences, min_distance_sec=0.2, tol=0.25
):
    from src.fine_tuning.metrics import evaluate, logits_to_boundary_times

    results = []
    for prom in prominences:
        pred_times = logits_to_boundary_times(
            logits, duration=duration, prominence=prom,
            min_distance_sec=min_distance_sec,
        )
        p, r, f1 = evaluate(pred_times, ref_times, tol=tol)
        results.append({"prominence": prom, "precision": p, "recall": r, "f1": f1})
    return results


def _boundaries_to_segments(boundaries, duration):
    edges = sorted(set([0.0] + [b for b in boundaries if 0.0 < b < duration] + [duration]))
    return list(zip(edges[:-1], edges[1:]))


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def purity_coverage(pred_boundaries, ref_boundaries, duration):
    
    S = _boundaries_to_segments(pred_boundaries, duration)
    R = _boundaries_to_segments(ref_boundaries, duration)

    if not S or not R:
        return 0.0, 0.0

    purity_num = sum(max(_overlap(s, r) for r in R) for s in S)
    purity_den = sum(s[1] - s[0] for s in S)
    coverage_num = sum(max(_overlap(s, r) for s in S) for r in R)
    coverage_den = sum(r[1] - r[0] for r in R)

    purity = purity_num / purity_den if purity_den > 0 else 0.0
    coverage = coverage_num / coverage_den if coverage_den > 0 else 0.0
    return purity, coverage

def bootstrap_f1_ci(per_file_f1, n_boot=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(per_file_f1)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(values.mean()), float(lower), float(upper)
