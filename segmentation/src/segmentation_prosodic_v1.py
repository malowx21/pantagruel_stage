import numpy as np
from scipy.signal import find_peaks


class ProsodicSegmenter:
    """
    Prosodic boundary detection using:

    - RMS energy drops
    - F0 variation
    - Pause duration

    Produces a continuous boundary score.
    """

    def __init__(self,sr=16000,hop_length=512,w_rms=0.4,w_f0=0.3,w_pause=0.3):
        self.sr = sr
        self.hop_length = hop_length

        self.w_rms = w_rms
        self.w_f0 = w_f0
        self.w_pause = w_pause

    def normalize(self, x):
        x = np.asarray(x)
        if len(x) == 0:
            return x
        std = np.std(x)
        if std < 1e-8:
            return np.zeros_like(x)

        return (x - np.mean(x)) / std


    # RMS FEATURE
    def rms_feature(self, rms):

        rms = self.normalize(rms)

        rms = np.convolve(rms,np.ones(5) / 5,mode="same")

        drms = np.diff(rms,prepend=rms[0])

        # Energy drop = candidate boundary
        return np.maximum(-drms, 0)


    # F0 FEATURE
    def f0_feature(self, f0):
        f0 = np.asarray(f0)
        f0 = np.where(f0 <= 0, np.nan, f0)
        valid = ~np.isnan(f0)

        if np.sum(valid) < 2:
            return np.zeros(len(f0))

        idx = np.arange(len(f0))

        f0_interp = np.interp(idx,idx[valid],f0[valid])
        f0_norm = self.normalize(f0_interp)

        df0 = np.diff(f0_norm,prepend=f0_norm[0])

        return np.abs(df0)

    # PAUSE FEATURE
    def pause_feature(self, pauses, length):
        pause_signal = np.zeros(length)

        for start, end in pauses:
            start_idx = int(
                start * self.sr / self.hop_length
            )

            end_idx = int(
                end * self.sr / self.hop_length
            )

            start_idx = max(0, start_idx)
            end_idx = min(length, end_idx)

            duration = end - start

            strength = min(duration, 1.0)

            pause_signal[start_idx:end_idx] += strength

        return pause_signal


    # SCRE
    def compute_score(self,rms,f0,pauses):
        L = len(rms)

        rms_feat = self.normalize(self.rms_feature(rms))
        f0_feat = self.normalize(self.f0_feature(f0))
        pause_feat = self.pause_feature(pauses,L)

        score = (
            self.w_rms * rms_feat +
            self.w_f0 * f0_feat +
            self.w_pause * pause_feat
        )

        return score


    # BOUNDARY DETECTION
    def detect_boundaries(self,rms,
        f0,pauses,
        prominence=0.1,
        min_distance_sec=0.2):

        score = self.compute_score(rms,f0,pauses)
        score_smooth = np.convolve(score,np.ones(5) / 5,mode="same")

        distance_frames = int(min_distance_sec *self.sr /self.hop_length)

        peaks, properties = find_peaks(
            score_smooth,
            prominence=prominence,
            distance=distance_frames)

        times = (peaks * self.hop_length /self.sr)

        return peaks, times, score_smooth, properties
