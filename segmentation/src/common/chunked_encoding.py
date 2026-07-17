import numpy as np


def encode_long_audio(encoder, audio, sr, chunk_sec=30.0, step_sec=15.0):
 
    total_duration = len(audio) / sr

    if total_duration <= chunk_sec:
        return encoder.encode(audio, sr)

    overlap_sec = chunk_sec - step_sec

    chunk_starts = []
    t = 0.0
    while t < total_duration:
        chunk_starts.append(t)
        t += step_sec

    chunk_embeddings = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_sec, total_duration)
        start_sample = int(t_start * sr)
        end_sample = int(t_end * sr)
        chunk_audio = audio[start_sample:end_sample]
        if len(chunk_audio) == 0:
            continue

        emb = encoder.encode(chunk_audio, sr)  # (T_chunk, D)
        n_frames = emb.shape[0]
        actual_duration = t_end - t_start
        frames_per_sec = n_frames / actual_duration if actual_duration > 0 else 0

        is_last = t_end >= total_duration - 1e-6
        keep_local_start = 0.0 if i == 0 else overlap_sec / 2
        keep_local_end = actual_duration if is_last else (actual_duration - overlap_sec / 2)

        keep_frame_start = int(round(keep_local_start * frames_per_sec))
        keep_frame_end = int(round(keep_local_end * frames_per_sec))
        keep_frame_end = max(keep_frame_end, keep_frame_start)

        chunk_embeddings.append(emb[keep_frame_start:keep_frame_end])

    if not chunk_embeddings:
        raise RuntimeError("Aucun chunk encodé -- vérifier la durée du fichier audio.")

    return np.concatenate(chunk_embeddings, axis=0)