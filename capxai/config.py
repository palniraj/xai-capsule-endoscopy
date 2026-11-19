"""Project-wide constants. Paths are relative to the project root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = DATA_DIR / "images"
INDEX_CSV = DATA_DIR / "index.csv"
RUNS_DIR = ROOT / "runs"          # checkpoints and per-image outputs (not committed)
RESULTS_DIR = ROOT / "results"    # small metric tables (committed)
FIGURES_DIR = ROOT / "figures"

# ---------------------------------------------------------------------------
# Data sources (Kvasir-Capsule, Smedsrud et al., Scientific Data 8:142, 2021)
# datasets.simula.no serves an incomplete TLS chain, so downloads are checked
# against the SHA-256 values below instead of relying on the TLS certificate.
# ---------------------------------------------------------------------------
IMAGES_ZIP_URL = "https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-images.zip"
IMAGES_ZIP_SHA256 = "f8fcaee9e45d58c76a1e156259cbdab946b67974ba93a0bfe4dbe9c69a81e9d6"
METADATA_URL = ("https://drive.usercontent.google.com/download?"
                "id=1nxp4S-yp3KSScVBolBG0N1az3QWzdKrO&export=download&confirm=t")
METADATA_SHA256 = "480373e840c48d7cad45aede92bdc8fa5a7c0064e6b22470d38af9f2032642f5"
SPLIT_URL = "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_{fold}.csv"
SPLIT_SHA256 = {
    0: "fc7ed45cb061a378403c1b31375345462c90408960ee555727f7c00f8c3c0b3d",
    1: "5757b5ac99bdba4642edc2fde7eb2e1dbb2e0392b591b6866ec837720508134a",
}

# ---------------------------------------------------------------------------
# Labels: the 11 classes used by the official two-fold splits.
# (Polyp, Blood - hematin and Ampulla of Vater are excluded by the authors.)
# ---------------------------------------------------------------------------
CLASSES = (
    "Angiectasia",
    "Blood",
    "Erosion",
    "Erythematous",
    "Foreign Bodies",
    "Ileo-cecal valve",
    "Lymphangiectasia",
    "Normal",
    "Pylorus",
    "Reduced Mucosal View",
    "Ulcer",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# Classes whose findings carry a bounding box in metadata.csv
LESION_CLASSES = (
    "Angiectasia",
    "Blood",
    "Erosion",
    "Erythematous",
    "Foreign Bodies",
    "Lymphangiectasia",
    "Ulcer",
)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VAL_FRACTION_FOLDS = 5     # 1 of 5 video-grouped folds of the training split -> validation
SEED = 42
