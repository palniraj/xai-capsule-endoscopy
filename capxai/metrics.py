"""Classification and calibration metrics, and temperature scaling."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(y_true, y_pred, labels) -> dict:
    """Macro scores are averaged over `labels` (classes present in the test split)."""
    kw = dict(labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, **kw),
        "macro_recall": recall_score(y_true, y_pred, **kw),
        "macro_f1": f1_score(y_true, y_pred, **kw),
        "micro_f1": f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def calibration_metrics(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
    onehot = np.eye(probs.shape[1])[y_true]
    nll = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, None)).mean()
    brier = ((probs - onehot) ** 2).sum(axis=1).mean()
    return {"ece": ece, "mce": mce, "nll": nll, "brier": brier,
            "mean_confidence": conf.mean()}


def reliability_bins(probs, y_true, n_bins: int = 15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((lo, hi, int(m.sum()),
                     correct[m].mean() if m.any() else np.nan,
                     conf[m].mean() if m.any() else np.nan))
    return rows


def cluster_bootstrap_ci(values, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean and percentile CI, resampling whole groups (videos) with replacement.

    Frames from one video are strongly correlated, so resampling frames would give
    intervals that are too narrow.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uniq = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in uniq])
    counts = np.array([(groups == g).sum() for g in uniq])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def cluster_bootstrap_metrics(y_true, y_pred, groups, metrics=("macro_f1", "mcc"), n_boot: int = 500,
                              seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CIs for classification metrics, resampling whole videos.

    Macro scores in each resample are averaged over the classes present in it.
    Returns {"<metric>_ci_low": .., "<metric>_ci_high": ..}.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        yt, yp = y_true[idx], y_pred[idx]
        res = classification_metrics(yt, yp, sorted(np.unique(yt)))
        for m in metrics:
            boot[m].append(res[m])
    out = {}
    for m in metrics:
        lo, hi = np.percentile(boot[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"{m}_ci_low"], out[f"{m}_ci_high"] = float(lo), float(hi)
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): minimise validation NLL over T.

    NLL is a 1-D function of log T, so a bounded scalar search (T in [0.05, 20]) is
    used instead of LBFGS, which can diverge to T -> 0 on small or degenerate inputs.
    """
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)

    def nll(log_t):
        return float(torch.nn.functional.cross_entropy(lg / np.exp(log_t), yt))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                          options={"xatol": 1e-4})
    return float(np.exp(res.x))
