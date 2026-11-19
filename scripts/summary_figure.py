"""One-figure summary: an example frame, localisation versus reference maps, and the sanity check.

Usage:
    python scripts/summary_figure.py --run runs/resnet50_fold0 --runs runs/resnet50_fold0 runs/resnet50_fold1

The example frame is chosen by a fixed rule before the randomised maps are computed:
a correctly classified test frame of --label from a video not shared with the training
split, with a box covering 5-25% of the image (median box size among those). The rule does
not look at any explanation score, so the example favours no method. Panel (a) shows the
trained model only; the evidence about methods is in panels (b) and (c), which aggregate
all frames.

Writes figures/summary.png (and .pdf).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402
from capxai.data import (CapsuleDataset, box_in_model_coords, denormalize, eval_transform,  # noqa: E402
                         load_index, shared_video_ids)
from capxai.explain import attribution_maps  # noqa: E402
from capxai.models import load_checkpoint  # noqa: E402

NAMES = {"random": "Random map", "center": "Centre map", "gradcam": "Grad-CAM",
         "integrated_gradients": "Integrated Gradients", "saliency": "Saliency"}
COLORS = {"gradcam": "#1f77b4", "integrated_gradients": "#ff7f0e", "saliency": "#2ca02c",
          "random": "#7f7f7f", "center": "#bcbd22"}


def pick_example(run: Path, label: str) -> pd.Series:
    per = pd.read_csv(run / "localisation_per_image.csv")
    ig = per[(per.method == "random") & (per.correct == 1) & (per.label == label)
             & per.box_area_frac.between(0.05, 0.25)]  # one row per frame; no method score used
    if ig.empty:
        raise SystemExit(f"no candidate frame for {label}")
    ig = ig[~ig.video_id.isin(shared_video_ids(load_index()))]  # patient not seen in training
    if ig.empty:
        raise SystemExit(f"no candidate frame for {label} from an unseen video")
    ig = ig.sort_values(["box_area_frac", "filename"]).reset_index(drop=True)
    fname = ig.iloc[len(ig) // 2]["filename"]
    frames = pd.read_csv(run / "test_frames.csv")
    return frames[frames.filename == fname].iloc[0]


def norm(a):
    a = np.clip(a, 0, None)
    return a / (a.max() + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run used for the example frame")
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="runs for the summary panels")
    ap.add_argument("--label", default="Angiectasia")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--out", type=Path, default=C.FIGURES_DIR / "summary")
    args = ap.parse_args()
    torch.manual_seed(C.SEED)

    # ---- example maps: trained model and fully randomised copy
    model, ckpt = load_checkpoint(args.run / "best.pt", "cpu")
    arch = ckpt["arch"]
    row = pick_example(args.run, args.label)
    ds = CapsuleDataset(pd.DataFrame([row]), eval_transform())
    x, y, _ = ds[0]
    xb, tgt = x.unsqueeze(0), torch.tensor([y])
    maps = {m: attribution_maps(model, arch, xb, tgt, m, ig_steps=args.ig_steps)[0]
            for m in ("gradcam", "integrated_gradients", "saliency")}
    img = denormalize(x)
    x0, y0, x1, y1 = box_in_model_coords(row)

    # ---- figure
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})
    fig = plt.figure(figsize=(6.7, 3.75))
    top = fig.add_gridspec(1, 4, wspace=0.06, left=0.13, right=0.87, top=0.92, bottom=0.555)
    bot = fig.add_gridspec(1, 2, wspace=0.42, width_ratios=[1, 1.05], left=0.175, right=0.985,
                           top=0.44, bottom=0.13)
    panels = [(None, f"{row['label']}, unseen video\n(green: clinician box)"),
              ("gradcam", "Grad-CAM"), ("integrated_gradients", "Integrated Gradients"),
              ("saliency", "Saliency")]
    for c, (m, title) in enumerate(panels):
        ax = fig.add_subplot(top[0, c])
        ax.imshow(img)
        if m:
            ax.imshow(norm(maps[m]), cmap="jet", alpha=0.5)
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.2))
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=7, pad=2)
        if c == 0:
            ax.text(-0.3, 1.17, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left")

    # (b) pointing hit with video-bootstrap CI, both folds
    axb = fig.add_subplot(bot[0, 0])
    order = ["random", "center", "gradcam", "saliency", "integrated_gradients"]
    for k, run in enumerate(args.runs):
        loc = pd.read_csv(C.RESULTS_DIR / f"localisation_{run.name}.csv").query("subset == 'all'").set_index("method")
        for i, m in enumerate(order):
            r = loc.loc[m]
            yy = i + (0.17 if k == 0 else -0.17)
            axb.errorbar(r.pointing_hit, yy, xerr=[[r.pointing_hit - r.pointing_hit_ci_low],
                                                   [r.pointing_hit_ci_high - r.pointing_hit]],
                         fmt="o" if k == 0 else "s", ms=3.5, color=COLORS[m], capsize=1.5, lw=0.9,
                         mfc=COLORS[m] if k == 0 else "white")
    axb.set_yticks(range(len(order)), [NAMES[m] for m in order])
    axb.set_xlim(0, 1)
    axb.set_xlabel("Pointing hit rate (95% CI over videos)")
    axb.plot([], [], "ko", ms=3.5, label="fold 0")
    axb.plot([], [], "ks", ms=3.5, mfc="white", label="fold 1")
    axb.legend(loc="lower right", frameon=False, handletextpad=0.2)
    axb.grid(axis="x", lw=0.3, alpha=0.5)
    axb.set_title("Localisation on all boxed lesion frames", fontsize=7, pad=3)
    axb.text(-0.5, 1.06, "b", transform=axb.transAxes, fontsize=9, fontweight="bold")

    # (c) sanity check
    axc = fig.add_subplot(bot[0, 1])
    for k, run in enumerate(args.runs):
        san = pd.read_csv(C.RESULTS_DIR / f"sanity_{run.name}.csv")
        for m in ("gradcam", "integrated_gradients", "saliency"):
            s = san[san.method == m].sort_values("stage_index")
            axc.plot(s.stage_index, s.spearman_mean, "-o" if k == 0 else "--s", ms=2.5, lw=1,
                     color=COLORS[m], mfc=COLORS[m] if k == 0 else "white",
                     label=NAMES[m] if k == 0 else None)
        stages = s.randomised_down_to.tolist()
    axc.axhline(0, color="k", lw=0.5)
    axc.set_xticks(range(len(stages)), ["trained"] + [f"→{t}" for t in stages[1:]], rotation=35, ha="right")
    axc.set_ylabel("Spearman vs trained map")
    axc.set_ylim(-0.5, 1.25)
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=3, columnspacing=0.6,
               handlelength=1.2, handletextpad=0.3, fontsize=6)
    axc.grid(axis="y", lw=0.3, alpha=0.5)
    axc.set_title("Cascading randomisation (solid: fold 0, dashed: fold 1)", fontsize=7, pad=3)
    axc.text(-0.22, 1.06, "c", transform=axc.transAxes, fontsize=9, fontweight="bold")

    args.out.parent.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    print(f"example: {row['filename']} ({row['label']}), wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
