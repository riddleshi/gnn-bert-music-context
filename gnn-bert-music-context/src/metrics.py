"""Classification, regression, and retrieval metrics from §6 of the spec."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, r2_score


def bce_logits_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def tag_metrics(y_true: np.ndarray, logits: np.ndarray, thresh: float = 0.5) -> dict[str, float]:
    y_prob = bce_logits_to_prob(logits)
    y_hat = (y_prob >= thresh).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    macro = f1_score(y_true_i, y_hat, average="macro", zero_division=0)
    micro = f1_score(y_true_i, y_hat, average="micro", zero_division=0)
    try:
        present = y_true_i.sum(axis=0) > 0
        if present.any():
            auc_pr = float(average_precision_score(y_true_i[:, present], y_prob[:, present], average="macro"))
        else:
            auc_pr = 0.0
    except ValueError:
        auc_pr = 0.0
    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "auc_pr": float(auc_pr),
    }


def genre_accuracy(y_true: np.ndarray, logits: np.ndarray) -> float:
    pred = logits.argmax(axis=1)
    return float((pred == y_true).mean())


def emotion_metrics(v_true: np.ndarray, v_pred: np.ndarray, a_true: np.ndarray, a_pred: np.ndarray) -> dict[str, float]:
    mae_v = float(np.mean(np.abs(v_true - v_pred)))
    mae_a = float(np.mean(np.abs(a_true - a_pred)))
    return {
        "mae_valence": mae_v,
        "mae_arousal": mae_a,
        "mae_emotion": 0.5 * (mae_v + mae_a),
        "r2_valence": float(r2_score(v_true, v_pred)) if len(v_true) > 1 else 0.0,
        "r2_arousal": float(r2_score(a_true, a_pred)) if len(a_true) > 1 else 0.0,
    }


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    prior = y_train.mean(axis=0)
    # Use train frequency as predicted probability; threshold 0.5 for F1.
    return np.broadcast_to(prior, (n_test, y_train.shape[1])).copy()


def random_baseline(n_test: int, n_tags: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_test, n_tags)).astype(np.float32)
