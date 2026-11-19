"""Attribution maps and the metrics that compare them with clinician bounding boxes.

All maps are returned as float arrays of shape (B, H, W) at model-input resolution,
where larger values mean "more important for the target class".
"""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Saliency
from scipy.stats import spearmanr

from .config import IMAGENET_MEAN, IMAGENET_STD
from .models import gradcam_layer

METHODS = ("gradcam", "integrated_gradients", "saliency", "random", "center")
BASELINE_METHODS = ("random", "center")  # model-independent reference maps


def _gradcam(model, arch, x, target):
    lgc = LayerGradCam(model, gradcam_layer(model, arch))
    a = lgc.attribute(x, target=target, relu_attributions=True)
    a = LayerAttribution.interpolate(a, x.shape[-2:], interpolate_mode="bilinear")
    return a[:, 0]


def _integrated_gradients(model, x, target, n_steps=32, internal_batch_size=32):
    ig = IntegratedGradients(model)
    # Black image in pixel space, expressed in the normalised input space.
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    baseline = ((0 - mean) / std).expand_as(x)
    a = ig.attribute(x, baselines=baseline, target=target, n_steps=n_steps,
                     internal_batch_size=internal_batch_size)
    return a.abs().sum(dim=1)


def _saliency(model, x, target):
    a = Saliency(model).attribute(x, target=target, abs=True)
    return a.sum(dim=1)


def _center(shape, sigma_frac=0.25):
    b, h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    s = sigma_frac * min(h, w)
    g = np.exp(-(((yy - h / 2) ** 2) + ((xx - w / 2) ** 2)) / (2 * s * s))
    return np.repeat(g[None], b, axis=0)


def attribution_maps(model, arch, x, target, method, rng=None, ig_steps=32) -> np.ndarray:
    """x: (B,3,H,W) normalised input on the model's device. target: (B,) long tensor."""
    b, _, h, w = x.shape
    if method == "random":
        rng = rng or np.random.default_rng(0)
        return rng.random((b, h, w))
    if method == "center":
        return _center((b, h, w))
    x = x.clone().requires_grad_(True)
    if method == "gradcam":
        a = _gradcam(model, arch, x, target)
    elif method == "integrated_gradients":
        a = _integrated_gradients(model, x, target, n_steps=ig_steps)
    elif method == "saliency":
        a = _saliency(model, x, target)
    else:
        raise ValueError(method)
    return a.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Localisation metrics against a boolean box mask (H, W)
# ---------------------------------------------------------------------------
def localisation_scores(amap: np.ndarray, mask: np.ndarray) -> dict:
    """Scores for one map.

    pointing_hit   1 if the single most important pixel lies in the box (Zhang et al., 2018).
    energy_in_box  share of positive attribution mass inside the box (Wang et al., 2020).
    top_area_prec  of the k most important pixels, k = box area, the share inside the box.
    box_area_frac  box area / image area; a uniform map scores this on the two last metrics.
    degenerate     1 if the map has no positive mass (e.g. Grad-CAM all zeros).
    """
    pos = np.clip(amap, 0, None)
    total = pos.sum()
    area = mask.mean()
    if total <= 0 or not np.isfinite(total):
        return {"pointing_hit": 0.0, "energy_in_box": np.nan, "top_area_prec": np.nan,
                "box_area_frac": area, "degenerate": 1.0}
    flat = amap.ravel()
    k = int(mask.sum())
    top = np.argpartition(-flat, k - 1)[:k]
    return {
        "pointing_hit": float(mask.ravel()[int(flat.argmax())]),
        "energy_in_box": float(pos[mask].sum() / total),
        "top_area_prec": float(mask.ravel()[top].mean()),
        "box_area_frac": float(area),
        "degenerate": 0.0,
    }


def map_similarity(a: np.ndarray, b: np.ndarray, pool: int = 4) -> float:
    """Spearman rank correlation of two maps after average-pooling by `pool`.

    Used by the model-parameter randomisation test (Adebayo et al., 2018).
    """
    def down(m):
        h, w = m.shape
        m = m[: h - h % pool, : w - w % pool]
        return m.reshape(h // pool, pool, w // pool, pool).mean(axis=(1, 3)).ravel()
    da, db = down(np.abs(a)), down(np.abs(b))
    if da.std() == 0 or db.std() == 0:
        return np.nan
    return float(spearmanr(da, db).statistic)
