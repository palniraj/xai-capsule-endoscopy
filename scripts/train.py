"""Fine-tune an ImageNet backbone on one official Kvasir-Capsule split and test on the other.

Usage:
    python scripts/train.py --arch resnet50 --train-fold 0     # fold 0 -> test on split_1
    python scripts/train.py --arch resnet50 --train-fold 1     # fold 1 -> test on split_0

Outputs in runs/<arch>_fold<k>/: best.pt, history.csv, logits_{val,test}.npz, config.json
The checkpoint with the best validation MCC is kept (MCC is less sensitive than
accuracy to the 70% Normal class).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402
from capxai.data import (CapsuleDataset, class_weights, eval_transform, fold_frames,  # noqa: E402
                         load_index, seed_everything, train_transform)
from capxai.metrics import classification_metrics  # noqa: E402
from capxai.models import build_model, save_checkpoint  # noqa: E402


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out, ys, idx = [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
        idx.append(i)
    return torch.cat(out).numpy(), torch.cat(ys).numpy(), torch.cat(idx).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--train-fold", type=int, choices=(0, 1), required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap each of train/val/test (smoke tests only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = args.out or C.RUNS_DIR / f"{args.arch}_fold{args.train_fold}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_index()
    train_df, val_df, test_df = fold_frames(df, args.train_fold, args.seed)
    if args.max_images:
        cap = lambda d: d.sample(min(len(d), args.max_images), random_state=args.seed).reset_index(drop=True)  # noqa: E731
        train_df, val_df, test_df = cap(train_df), cap(val_df), cap(test_df)
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"{name:5s} {len(d):6d} images, {d.video_id.nunique():2d} videos, "
              f"classes: {d.label.value_counts().to_dict()}")
    train_df.to_csv(out / "train_frames.csv", index=False)
    val_df.to_csv(out / "val_frames.csv", index=False)
    test_df.to_csv(out / "test_frames.csv", index=False)

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=amp)
    train_dl = DataLoader(CapsuleDataset(train_df, train_transform()), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(CapsuleDataset(val_df, eval_transform()), **kw)
    test_dl = DataLoader(CapsuleDataset(test_df, eval_transform()), **kw)

    model = build_model(args.arch, len(C.CLASSES), pretrained=not args.no_pretrained).to(device)
    weights = class_weights(train_df["y"]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(train_dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    val_labels = sorted(val_df["y"].unique())

    history, best = [], -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        logits, yv, _ = predict(model, val_dl, device, amp)
        m = classification_metrics(yv, logits.argmax(1), val_labels)
        row = {"epoch": epoch, "train_loss": run_loss / max(n, 1), **{f"val_{k}": v for k, v in m.items()},
               "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val acc {m['accuracy']:.3f}  "
              f"macroF1 {m['macro_f1']:.3f}  MCC {m['mcc']:.3f}  ({row['seconds']:.0f}s)")
        if m["mcc"] > best:
            best = m["mcc"]
            save_checkpoint(out / "best.pt", model, args.arch, C.CLASSES,
                            {"epoch": epoch, "train_fold": args.train_fold, "val_mcc": best})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    for split, dl in (("val", val_dl), ("test", test_dl)):
        logits, y, idx = predict(model, dl, device, amp)
        frame = dl.dataset.df
        np.savez(out / f"logits_{split}.npz", logits=logits, y=y,
                 filename=frame["filename"].to_numpy()[idx], label=frame["label"].to_numpy()[idx])
    (out / "config.json").write_text(json.dumps({**vars(args), "out": str(out), "device": str(device),
                                                 "best_epoch": ckpt["epoch"], "best_val_mcc": best,
                                                 "class_weights": weights.tolist()}, indent=2, default=str))
    print(f"[ok] best epoch {ckpt['epoch']} (val MCC {best:.3f}); outputs in {out}")


if __name__ == "__main__":
    main()
