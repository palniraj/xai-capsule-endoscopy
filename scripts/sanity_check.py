"""Model-parameter randomisation test (Adebayo et al., NeurIPS 2018).

Layers are re-initialised one at a time from the classifier head downwards (cascading).
If a method's maps barely change while the model is being destroyed, the maps reflect
the image (edges, colour) rather than what the model learned, and should not be trusted.
Rank correlation near 1 after randomisation = failed sanity check.

Usage:
    python scripts/sanity_check.py --run runs/resnet50_fold0 --n-images 100

Writes results/sanity_<run>.csv and figures/sanity_<run>.png, figures/sanity_examples_<run>.png.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402
from capxai.data import CapsuleDataset, box_mask, denormalize, eval_transform  # noqa: E402
from capxai.explain import attribution_maps, localisation_scores, map_similarity  # noqa: E402
from capxai.models import load_checkpoint, randomisation_cascade, reinit_module  # noqa: E402

MODEL_METHODS = ("gradcam", "integrated_gradients", "saliency")


def compute_maps(model, arch, ds, methods, device, batch_size, ig_steps):
    out = {m: [] for m in methods}
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        y = torch.tensor([ds[i][1] for i in idx], device=device)
        for m in methods:
            out[m].append(attribution_maps(model, arch, x, y, m, ig_steps=ig_steps))
    return {m: np.concatenate(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=list(MODEL_METHODS), choices=MODEL_METHODS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.run / "best.pt", device)
    arch, name = ckpt["arch"], args.run.name

    # Correctly classified, boxed lesion images: the case where explanations matter most.
    per_image = args.run / "localisation_per_image.csv"
    frame = pd.read_csv(args.run / "test_frames.csv")
    frame = frame[frame["label"].isin(C.LESION_CLASSES) & frame["x_min"].notna()]
    if per_image.exists():
        ok = pd.read_csv(per_image).query("correct == 1")["filename"].unique()
        if len(ok):
            frame = frame[frame.filename.isin(ok)]
    frame = frame.sample(min(args.n_images, len(frame)), random_state=args.seed).reset_index(drop=True)
    ds = CapsuleDataset(frame, eval_transform())
    masks = [box_mask(frame.iloc[i]) for i in range(len(frame))]
    print(f"{len(frame)} images from {frame.video_id.nunique()} videos")

    stages = ["trained"] + randomisation_cascade(arch)
    rand_model = copy.deepcopy(model)
    maps_by_stage = {}
    rows = []
    for s, stage in enumerate(tqdm(stages, desc="cascade")):
        if stage != "trained":
            reinit_module(rand_model.get_submodule(stage), generator_seed=args.seed + s)
        rand_model.eval()
        maps = compute_maps(rand_model, arch, ds, args.methods, device, args.batch_size, args.ig_steps)
        maps_by_stage[stage] = maps
        for m in args.methods:
            ref = maps_by_stage["trained"][m]
            for i in range(len(ds)):
                loc = localisation_scores(maps[m][i], masks[i])
                rows.append({"stage_index": s, "randomised_down_to": stage, "method": m,
                             "filename": frame.at[i, "filename"], "video_id": frame.at[i, "video_id"],
                             "spearman_vs_trained": map_similarity(ref[i], maps[m][i]),
                             "energy_in_box": loc["energy_in_box"], "pointing_hit": loc["pointing_hit"]})
    res = pd.DataFrame(rows)
    summary = (res.groupby(["stage_index", "randomised_down_to", "method"])
                  .agg(spearman_mean=("spearman_vs_trained", "mean"),
                       spearman_std=("spearman_vs_trained", "std"),
                       energy_in_box=("energy_in_box", "mean"),
                       pointing_hit=("pointing_hit", "mean"),
                       n=("filename", "size")).reset_index())
    C.RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(C.RESULTS_DIR / f"sanity_{name}.csv", index=False)
    print(summary.round(3).to_string(index=False))

    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in args.methods:
        g = summary[summary.method == m]
        ax[0].errorbar(g.stage_index, g.spearman_mean, yerr=g.spearman_std, marker="o", capsize=3, label=m)
        ax[1].plot(g.stage_index, g.energy_in_box, marker="o", label=m)
    for a in ax:
        a.set_xticks(range(len(stages)), stages, rotation=40, ha="right", fontsize=8)
        a.set_xlabel("re-initialised from the head down to")
        a.legend(fontsize=8)
    ax[0].set_ylabel("Spearman correlation with trained-model map")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("energy inside clinician box")
    fig.suptitle(f"{name}: cascading model-parameter randomisation")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"sanity_{name}.png", dpi=150)
    plt.close(fig)

    # One example image across all stages, per method.
    i = 0
    fig, axes = plt.subplots(len(args.methods), len(stages), figsize=(1.9 * len(stages), 2 * len(args.methods)),
                             squeeze=False)
    img = denormalize(ds[i][0])
    for r, m in enumerate(args.methods):
        for c, stage in enumerate(stages):
            a = maps_by_stage[stage][m][i]
            axes[r, c].imshow(img)
            axes[r, c].imshow(np.abs(a) / (np.abs(a).max() + 1e-12), cmap="jet", alpha=0.5)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(stage, fontsize=8)
            if c == 0:
                axes[r, c].set_ylabel(m.replace("_", " "), fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"sanity_examples_{name}.png", dpi=130)
    plt.close(fig)
    print(f"[ok] wrote results/sanity_{name}.csv and figures/sanity_{name}.png")


if __name__ == "__main__":
    main()
