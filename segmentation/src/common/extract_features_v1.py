import numpy as np
import librosa
import parselmouth


DEFAULT_SR = 16000
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512


def extract_rms(audio,frame_length=DEFAULT_FRAME_LENGTH,hop_length=DEFAULT_HOP_LENGTH):
    """
    Compute the Root Mean Square energy for each audio frame.

    Args:
        audio : Audio time series.
        frame_length : Number of samples per analysis frame.
        hop_length : Number of samples between successive frames.

    Returns:
        np.ndarray: RMS energy computed for each frame.
    """
    rms = librosa.feature.rms(y=audio,frame_length=frame_length,hop_length=hop_length)[0]
    return rms


def extract_f0(audio,sr=DEFAULT_SR,hop_length=DEFAULT_HOP_LENGTH,pitch_floor=75,pitch_ceiling=600):
    """
    Extract the fundamental frequency F0 using Praat.

    The analysis time step is set to match the hop length used for
    RMS extraction, ensuring temporal alignment between both features.

    Args:
        audio : Audio time series.
        sr : Sampling rate of the audio signal.
        hop_length : Number of samples between successive analysis frames.
        pitch_floor : Minimum detectable pitch in Hz.
        pitch_ceiling : Maximum detectable pitch in Hz.

    Returns:
        np.ndarray: Estimated fundamental frequency for each analysis frame.
        Unvoiced frames are returned as 0 .
    """
    sound = parselmouth.Sound(audio,sampling_frequency=sr)
    time_step = hop_length / sr
    pitch = sound.to_pitch(time_step=time_step,pitch_floor=pitch_floor,pitch_ceiling=pitch_ceiling)
    f0 = pitch.selected_array["frequency"]

    return f0


def extract_pauses(audio,sr,hop_length=DEFAULT_HOP_LENGTH,top_db=30):
    """
    Detect pauses from silent regions in an audio signal.

    Args:
        audio: Audio time series.
        sr: Sampling rate of the audio signal.
        hop_length: Number of samples between successive analysis frames.
        top_db : Threshold (in dB) below the reference level used to distinguish silence from speech.

    Returns:
        list[tuple[float, float]]: List of pause intervals expressed as (start_time, end_time) in seconds.
        Returns an empty list if fewer than two speech segments are detected.
    """

    speech_intervals = librosa.effects.split(audio,top_db=top_db)
    pauses = []

    if len(speech_intervals) < 2:
        return pauses

    for i in range(len(speech_intervals) - 1):
        pause_start = speech_intervals[i][1] / sr
        pause_end = speech_intervals[i + 1][0] / sr

        pauses.append((pause_start, pause_end))

    return pauses


def align_features(rms, f0):
    """
    Align RMS and F0 features to the same number of frames using linear interpolation.

    Args:
        rms: Root-Mean-Square energy array.
        f0: Fundamental frequency  array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the original RMS array 
        and the resampled, aligned F0 array.
    """
    target_length = len(rms)

    if len(f0) == target_length:
        return rms, f0

    old_idx = np.arange(len(f0))
    new_idx = np.linspace(0,len(f0) - 1,target_length)

    f0_aligned = np.interp(new_idx,old_idx,f0)

    return rms, f0_aligned


def extract_features(audio,sr=DEFAULT_SR,frame_length=DEFAULT_FRAME_LENGTH,hop_length=DEFAULT_HOP_LENGTH,top_db=30):
    """
    Main feature extraction pipeline to compute RMS, F0, and pause intervals from audio.

    Args:
        audio: Audio time series.
        sr: Sampling rate of the audio signal.
        frame_length: Number of samples per frame for windowing.
        hop_length: Number of samples between successive analysis frames.
        top_db: Threshold (in dB) below the reference level used to distinguish silence from speech.

    Returns:
        dict: A dictionary containing the extracted features and metadata:
            - "rms" (np.ndarray): Aligned Root-Mean-Square energy.
            - "f0" (np.ndarray): Aligned Fundamental frequency tracking.
            - "pauses" (list[tuple]): Detected pause intervals in seconds.
            - "n_frames" (int): Total number of aligned frames.
            - "hop_length" (int): Hop length used for extraction.
            - "sr" (int): Sampling rate of the processed audio.
    """
    rms = extract_rms(audio,frame_length=frame_length,hop_length=hop_length)

    f0 = extract_f0(audio,sr=sr,hop_length=hop_length)

    rms, f0 = align_features(rms,f0)

    pauses = extract_pauses(audio,sr=sr,hop_length=hop_length,top_db=top_db)

    return {
        "rms": rms,
        "f0": f0,
        "pauses": pauses,
        "n_frames": len(rms),
        "hop_length": hop_length,
        "sr": sr
    }
