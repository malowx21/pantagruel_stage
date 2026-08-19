"""
Aligns word-level text embeddings onto the audio frame grid.
"""

import numpy as np


def align_text_to_frames(words, word_vectors, num_frames, duration, dim):
    """
    Repeats each word's embedding over every audio frame it covers.

    Args:
        words: list of (word, start_sec, end_sec) tuples, in order --
            typically the "words" list produced by word_alignment.py
            for a given clip.
        word_vectors: np.ndarray of shape (len(words), dim), typically the
            output of PantagruelSpeechTextAudioModel.encode_text(word_list)
        num_frames: number of audio frames for this clip
        duration: total duration of the audio in seconds.
        dim: embedding dimension.
    Returns:
        np.ndarray of shape (num_frames, dim).
    """
    if num_frames == 0:
        return np.zeros((0, dim), dtype=np.float32)

    text_frames = np.zeros((num_frames, dim), dtype=np.float32)

    if len(words) == 0 or duration <= 0:
        return text_frames

    frames_per_sec = num_frames / duration

    for (word, start, end), vec in zip(words, word_vectors):
        start_frame = int(start * frames_per_sec)
        end_frame = int(end * frames_per_sec)

        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames))

        if end_frame <= start_frame:
            end_frame = start_frame + 1

        text_frames[start_frame:end_frame] = vec

    return text_frames
