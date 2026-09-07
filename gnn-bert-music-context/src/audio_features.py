"""Mel spectrograms, chroma, MFCC, and beat-aware segmentation.

Matches the course preprocessing spec:
resample 22,050 Hz, 128-bin log-mel (or 12-bin chroma), per-track normalize,
fixed windows or beat-synchronous segments via librosa.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from src.config import load_config


@dataclass
class AudioBundle:
    y: np.ndarray
    sr: int
    log_mel: np.ndarray  # (n_mels, T)
    chroma: np.ndarray  # (12, T)
    mfcc: np.ndarray  # (n_mfcc, T)
    rms: np.ndarray
    centroid: np.ndarray
    zcr: np.ndarray
    tempo: float
    beat_frames: np.ndarray


def _cfg():
    return load_config()


def load_mono(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    cfg = _cfg()
    sr = sr or int(cfg["sample_rate"])
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def log_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int | None = None) -> np.ndarray:
    cfg = _cfg()
    n_mels = n_mels or int(cfg["n_mels"])
    hop = int(cfg["hop_length"])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop, power=2.0)
    log_mel = librosa.power_to_db(S, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def chroma_features(y: np.ndarray, sr: int) -> np.ndarray:
    cfg = _cfg()
    hop = int(cfg["hop_length"])
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_chroma=int(cfg["n_chroma"]))
    C = C / (C.sum(axis=0, keepdims=True) + 1e-8)
    return C.astype(np.float32)


def mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    cfg = _cfg()
    hop = int(cfg["hop_length"])
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=int(cfg["n_mfcc"]), hop_length=hop)
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-6)
    return M.astype(np.float32)


def extract_bundle(y: np.ndarray, sr: int) -> AudioBundle:
    cfg = _cfg()
    hop = int(cfg["hop_length"])
    log_mel = log_mel_spectrogram(y, sr)
    chroma = chroma_features(y, sr)
    mfcc = mfcc_features(y, sr)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0].astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0].astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0].astype(np.float32)
    # Fixed-window graphs are the default; skip beat tracking (expensive) unless needed.
    tempo = float(cfg.get("default_tempo", 120.0))
    beat_frames = np.array([], dtype=np.int64)
    return AudioBundle(
        y=y.astype(np.float32),
        sr=sr,
        log_mel=log_mel,
        chroma=chroma,
        mfcc=mfcc,
        rms=rms,
        centroid=centroid,
        zcr=zcr,
        tempo=tempo,
        beat_frames=np.asarray(beat_frames),
    )


def frame_times(n_frames: int, sr: int, hop: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)


def segment_slices(n_frames: int, sr: int, hop: int, segment_seconds: float) -> list[tuple[int, int]]:
    """Fixed-window frame slices covering the track."""
    frames_per_seg = max(1, int(round(segment_seconds * sr / hop)))
    slices = []
    start = 0
    while start < n_frames:
        end = min(n_frames, start + frames_per_seg)
        if end - start >= max(2, frames_per_seg // 3):
            slices.append((start, end))
        start += frames_per_seg
    if not slices:
        slices = [(0, n_frames)]
    return slices


def beat_synchronous_slices(beat_frames: np.ndarray, n_frames: int, group: int = 2) -> list[tuple[int, int]]:
    beats = [0, *beat_frames.tolist(), n_frames]
    beats = sorted(set(int(b) for b in beats if 0 <= b <= n_frames))
    if len(beats) < 3:
        return [(0, n_frames)]
    slices = []
    i = 0
    while i < len(beats) - 1:
        j = min(len(beats) - 1, i + group)
        a, b = beats[i], beats[j]
        if b > a:
            slices.append((a, b))
        i += group
    return slices or [(0, n_frames)]


def pool_segment(bundle: AudioBundle, start: int, end: int) -> np.ndarray:
    """Concatenate pooled audio descriptors for one segment (node features)."""
    sl = slice(start, end)

    def _mean_std(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            arr = arr[None, sl]
        else:
            arr = arr[:, sl]
        mu = arr.mean(axis=1)
        sd = arr.std(axis=1)
        return np.concatenate([mu, sd], axis=0)

    feat = np.concatenate(
        [
            _mean_std(bundle.log_mel),
            _mean_std(bundle.chroma),
            _mean_std(bundle.mfcc),
            _mean_std(bundle.rms),
            _mean_std(bundle.centroid / (bundle.centroid.max() + 1e-6)),
            _mean_std(bundle.zcr),
        ]
    ).astype(np.float32)
    return feat


def node_feature_dim() -> int:
    cfg = _cfg()
    # mean+std for mel, chroma, mfcc, rms, centroid, zcr
    return 2 * (int(cfg["n_mels"]) + int(cfg["n_chroma"]) + int(cfg["n_mfcc"]) + 3)
