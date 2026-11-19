"""Dataset, transforms and splits for Kvasir-Capsule."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config as C


class RandomRot90:
    """Rotate by 0/90/180/270 degrees. Capsule frames have no canonical orientation."""

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3))


def train_transform(size: int = C.IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRot90(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def eval_transform(size: int = C.IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) float image in [0,1]."""
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def load_index(path=C.INDEX_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["y"] = df["label"].map(C.CLASS_TO_IDX)
    if df["y"].isna().any():
        bad = sorted(df.loc[df["y"].isna(), "label"].unique())
        raise ValueError(f"Unknown labels in index: {bad}")
    df["y"] = df["y"].astype(int)
    return df


def fold_frames(df: pd.DataFrame, train_fold: int, seed: int = C.SEED):
    """Return (train, val, test) frames for one direction of the official 2-fold protocol.

    The test set is the other official split. Validation is one video-grouped,
    class-stratified fold carved from the training split, so no video is shared
    between training and validation.
    """
    train_all = df[df["fold"] == train_fold].reset_index(drop=True)
    test = df[df["fold"] != train_fold].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=C.VAL_FRACTION_FOLDS, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(train_all, train_all["y"], groups=train_all["video_id"]))
    return (train_all.iloc[tr_idx].reset_index(drop=True),
            train_all.iloc[va_idx].reset_index(drop=True),
            test)


def shared_video_ids(df: pd.DataFrame) -> set[str]:
    """Videos with frames in both official splits.

    The official splits are not fully separated by video: 7 of the videos contribute
    frames (of different classes) to both splits. Test frames from these videos come
    from a patient the model has already seen in training, so results are also
    reported on the test frames from the remaining videos.
    """
    folds_per_video = df.groupby("video_id")["fold"].nunique()
    return set(folds_per_video[folds_per_video > 1].index)


def box_in_model_coords(row, size: int = C.IMG_SIZE):
    """Bounding box (x0, y0, x1, y1) scaled to the model input, or None."""
    if pd.isna(row["x_min"]):
        return None
    sx, sy = size / row["width"], size / row["height"]
    x0 = int(np.floor(row["x_min"] * sx))
    y0 = int(np.floor(row["y_min"] * sy))
    x1 = int(np.ceil(row["x_max"] * sx))
    y1 = int(np.ceil(row["y_max"] * sy))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(size, max(x1, x0 + 1)), min(size, max(y1, y0 + 1))
    return x0, y0, x1, y1


def box_mask(row, size: int = C.IMG_SIZE) -> np.ndarray | None:
    box = box_in_model_coords(row, size)
    if box is None:
        return None
    m = np.zeros((size, size), dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


class CapsuleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, image_dir=C.IMAGE_DIR):
        self.df = frame.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def load_pil(self, i: int) -> Image.Image:
        return Image.open(self.image_dir / self.df.at[i, "filename"]).convert("RGB")

    def __getitem__(self, i):
        return self.transform(self.load_pil(i)), int(self.df.at[i, "y"]), i


def class_weights(labels, n_classes: int = len(C.CLASSES)) -> torch.Tensor:
    """Inverse-square-root frequency weights, normalised to mean 1 over present classes.

    Classes absent from the training split get weight 0.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = 1.0 / np.sqrt(counts[present])
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
