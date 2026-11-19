"""Classification and calibration results for trained runs, with temperature scaling.

Usage:
    python scripts/evaluate.py runs/resnet50_fold0 runs/resnet50_fold1

Scores are reported on two test subsets: "all" (the official protocol) and
"unseen_videos", which drops test frames from the 7 videos that also contribute
frames to the training split. 95% CIs for macro-F1 and MCC resample whole videos.

Writes results/classification.csv, results/calibration.csv, results/per_class_<run>.csv,
and figures/confusion_<run>.png, figures/reliability_<run>.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402
from capxai.data import load_index, shared_video_ids  # noqa: E402
from capxai.metrics import (calibration_metrics, classification_metrics,  # noqa: E402
                            cluster_bootstrap_metrics, fit_temperature, reliability_bins, softmax)


def plot_confusion(y, pred, labels, path, title):
    cm = confusion_matrix(y, pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    names = [C.CLASSES[i] for i in labels]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (row-normalised colour, counts in cells)")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(probs_before, probs_after, y, T, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for probs, name in ((probs_before, "uncalibrated"), (probs_after, f"temperature T={T:.2f}")):
        rows = [r for r in reliability_bins(probs, y) if r[2] > 0]
        ax.plot([r[4] for r in rows], [r[3] for r in rows], "o-", label=name)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--n-boot", type=int, default=500, help="video-level bootstrap resamples for CIs")
    args = ap.parse_args()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIGURES_DIR.mkdir(exist_ok=True)

    index = load_index()
    shared = shared_video_ids(index)
    video_of = index.drop_duplicates("filename").set_index("filename")["video_id"]

    cls_rows, cal_rows = [], []
    for run in args.runs:
        name = run.name
        val, test = np.load(run / "logits_val.npz", allow_pickle=True), np.load(run / "logits_test.npz", allow_pickle=True)
        T = fit_temperature(val["logits"], val["y"])
        y = test["y"]
        videos = video_of.loc[test["filename"]].to_numpy()
        p0, p1 = softmax(test["logits"]), softmax(test["logits"], T)
        pred = p0.argmax(1)  # temperature scaling does not change the argmax

        # "unseen_videos" drops test frames from videos that also have frames in the training split.
        for subset, keep in (("all", np.ones(len(y), bool)), ("unseen_videos", ~np.isin(videos, list(shared)))):
            ys, ps, vs = y[keep], pred[keep], videos[keep]
            labels = sorted(np.unique(ys))  # macro scores over classes present in the subset
            cls_rows.append({"run": name, "subset": subset, "n_test": int(keep.sum()),
                             "n_videos": len(np.unique(vs)), "n_classes_in_test": len(labels),
                             **classification_metrics(ys, ps, labels),
                             **cluster_bootstrap_metrics(ys, ps, vs, n_boot=args.n_boot)})
            for tag, p in (("uncalibrated", p0), ("temperature_scaled", p1)):
                cal_rows.append({"run": name, "subset": subset, "calibration": tag,
                                 "temperature": 1.0 if tag == "uncalibrated" else T,
                                 **calibration_metrics(p[keep], ys)})

        labels = sorted(np.unique(y))
        pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        pd.DataFrame({"class": [C.CLASSES[i] for i in labels], "precision": pr, "recall": rc,
                      "f1": f1, "support": sup}).to_csv(C.RESULTS_DIR / f"per_class_{name}.csv", index=False)
        plot_confusion(y, pred, labels, C.FIGURES_DIR / f"confusion_{name}.png", f"{name}: test split")
        plot_reliability(p0, p1, y, T, C.FIGURES_DIR / f"reliability_{name}.png", f"{name}: reliability")

    cls = pd.DataFrame(cls_rows)
    cal = pd.DataFrame(cal_rows)
    if len(args.runs) > 1:  # mean over runs (the two directions of the 2-fold protocol)
        point = [c for c in cls.columns if c not in ("run", "subset") and "_ci_" not in c]
        mean = cls.groupby("subset", sort=False)[point].mean().reset_index().assign(run="mean")
        cls = pd.concat([cls, mean], ignore_index=True)
    cls.to_csv(C.RESULTS_DIR / "classification.csv", index=False)
    cal.to_csv(C.RESULTS_DIR / "calibration.csv", index=False)
    pd.set_option("display.width", 160)
    print(cls.round(4).to_string(index=False))
    print(cal.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
