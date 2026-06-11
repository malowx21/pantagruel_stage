import numpy as np
import librosa
import parselmouth


DEFAULT_SR = 16000
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512


def extract_rms(audio,frame_length=DEFAULT_FRAME_LENGTH,hop_length=DEFAULT_HOP_LENGTH):
    """
    RMS energy per frame.
    """
    rms = librosa.feature.rms(y=audio,frame_length=frame_length,hop_length=hop_length)[0]
    return rms


def extract_f0(audio,sr=DEFAULT_SR,hop_length=DEFAULT_HOP_LENGTH,pitch_floor=75,pitch_ceiling=600):
    """
    F0 extraction with Praat (Parselmouth).

    The time step is aligned with the hop_length
    used for RMS extraction.
    """
    sound = parselmouth.Sound(audio,sampling_frequency=sr)
    time_step = hop_length / sr
    pitch = sound.to_pitch(time_step=time_step,pitch_floor=pitch_floor,pitch_ceiling=pitch_ceiling)
    f0 = pitch.selected_array["frequency"]

    return f0


def extract_pauses(audio,sr,hop_length=DEFAULT_HOP_LENGTH,top_db=30):
    """
    Detect pauses from silent regions.
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
    Align RMS and F0 to the same number of frames.
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
    Main feature extraction function.
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
