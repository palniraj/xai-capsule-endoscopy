"""Do the explanations point at the lesion? Compare attribution maps with clinician boxes.

For every test-split image of a lesion class that has a bounding box, the script explains
the true class with each method and scores the map against the box. Model-independent
reference maps ("random", "center") show what a map scores with no information about
the image, so a method is only useful if it clearly beats them.

Usage:
    python scripts/explain_eval.py --run runs/resnet50_fold0
    python scripts/explain_eval.py --run runs/resnet50_fold0 --max-images 300 --ig-steps 16

Writes runs/<run>/localisation_per_image.csv, results/localisation_<run>.csv,
results/localisation_by_class_<run>.csv and figures/explanations_<run>.png.
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
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402
from capxai.data import (CapsuleDataset, box_in_model_coords, box_mask, denormalize,  # noqa: E402
                         eval_transform, load_index, shared_video_ids)
from capxai.explain import METHODS, attribution_maps, localisation_scores  # noqa: E402
from capxai.metrics import cluster_bootstrap_ci  # noqa: E402
from capxai.models import load_checkpoint  # noqa: E402

METRICS = ("pointing_hit", "energy_in_box", "top_area_prec")


def boxed_test_frame(run: Path, max_images: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(run / "test_frames.csv")
    df = df[df["label"].isin(C.LESION_CLASSES) & df["x_min"].notna()].reset_index(drop=True)
    if max_images and len(df) > max_images:
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(max(1, round(max_images * len(g) / len(df))), random_state=seed))
                .reset_index(drop=True))
    return df


def summarise(per_image: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in per_image.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n_images"] = len(g)
        row["n_videos"] = g["video_id"].nunique()
        for m in METRICS:
            mean, lo, hi = cluster_bootstrap_ci(g[m], g["video_id"])
            row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = mean, lo, hi
        row["box_area_frac"] = g["box_area_frac"].mean()
        row["degenerate_rate"] = g["degenerate"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def overlay_figure(model, arch, ds, rows, methods, device, path, title, ig_steps):
    shown = [m for m in methods if m not in ("random", "center")]
    fig, axes = plt.subplots(len(rows), 1 + len(shown), figsize=(2.2 * (1 + len(shown)), 2.3 * len(rows)),
                             squeeze=False)
    for r, i in enumerate(rows):
        x, y, _ = ds[i]
        xb = x.unsqueeze(0).to(device)
        tgt = torch.tensor([y], device=device)
        img = denormalize(x)
        x0, y0, x1, y1 = box_in_model_coords(ds.df.iloc[i])
        for c, m in enumerate(["image"] + shown):
            ax = axes[r, c]
            ax.imshow(img)
            if m != "image":
                a = attribution_maps(model, arch, xb, tgt, m, ig_steps=ig_steps)[0]
                ax.imshow(a / (a.max() + 1e-12), cmap="jet", alpha=0.45)
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(m.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(ds.df.iloc[i]["label"], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--n-examples", type=int, default=7, help="rows in the example figure")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.run / "best.pt", device)
    arch, name = ckpt["arch"], args.run.name
    frame = boxed_test_frame(args.run, args.max_images, args.seed)
    ds = CapsuleDataset(frame, eval_transform())
    print(f"{len(frame)} boxed lesion images from {frame.video_id.nunique()} test videos")

    rng = np.random.default_rng(args.seed)
    rows = []
    for start in tqdm(range(0, len(ds), args.batch_size), desc="explaining"):
        idx = list(range(start, min(start + args.batch_size, len(ds))))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        y = torch.tensor([ds[i][1] for i in idx], device=device)
        with torch.no_grad():
            logits = model(x)
        prob = logits.softmax(1)
        pred = logits.argmax(1)
        masks = [box_mask(frame.iloc[i]) for i in idx]
        for m in args.methods:
            maps = attribution_maps(model, arch, x, y, m, rng=rng, ig_steps=args.ig_steps)
            for k, i in enumerate(idx):
                rows.append({"filename": frame.at[i, "filename"], "video_id": frame.at[i, "video_id"],
                             "label": frame.at[i, "label"], "method": m,
                             "pred": C.CLASSES[int(pred[k])], "correct": int(pred[k] == y[k]),
                             "p_true": float(prob[k, y[k]]),
                             **localisation_scores(maps[k], masks[k])})
    per_image = pd.DataFrame(rows)
    per_image.to_csv(args.run / "localisation_per_image.csv", index=False)

    unseen = ~per_image["video_id"].isin(shared_video_ids(load_index()))
    overall = pd.concat([
        summarise(per_image, ["method"]).assign(subset="all"),
        summarise(per_image[per_image.correct == 1], ["method"]).assign(subset="correctly_classified"),
        summarise(per_image[unseen], ["method"]).assign(subset="unseen_videos"),
    ], ignore_index=True)
    by_class = summarise(per_image, ["label", "method"])
    C.RESULTS_DIR.mkdir(exist_ok=True)
    overall.to_csv(C.RESULTS_DIR / f"localisation_{name}.csv", index=False)
    by_class.to_csv(C.RESULTS_DIR / f"localisation_by_class_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["subset", "method", "n_images", "n_videos", "pointing_hit", "pointing_hit_ci_low",
            "pointing_hit_ci_high", "energy_in_box", "top_area_prec", "box_area_frac", "degenerate_rate"]
    print(overall[cols].round(3).to_string(index=False))

    # Example figure: one correctly classified image per lesion class where possible.
    first = per_image[per_image.method == args.methods[0]]
    picks = []
    for label in C.LESION_CLASSES:
        g = first[(first.label == label)]
        g = g[g.correct == 1] if (g.correct == 1).any() else g
        if len(g):
            picks.append(int(frame.index[frame.filename == g.iloc[len(g) // 2]["filename"]][0]))
    C.FIGURES_DIR.mkdir(exist_ok=True)
    overlay_figure(model, arch, ds, picks[: args.n_examples], args.methods, device,
                   C.FIGURES_DIR / f"explanations_{name}.png",
                   f"{name}: explanations of the true class (green = clinician box)", args.ig_steps)
    print(f"[ok] wrote results/localisation_{name}.csv and figures/explanations_{name}.png")


if __name__ == "__main__":
    main()
