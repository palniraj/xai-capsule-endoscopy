"""Download Kvasir-Capsule labelled images, metadata and official splits, then build data/index.csv.

Usage:
    python scripts/prepare_data.py                      # full dataset (~550 MB download)
    python scripts/prepare_data.py --max-per-class 40   # small subset for a CPU smoke test

Every downloaded file is checked against a pinned SHA-256 (see capxai/config.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capxai import config as C  # noqa: E402

# metadata.csv finding_class -> label used by the official splits
META_TO_SPLIT = {
    "Angiectasia": "Angiectasia",
    "Blood - fresh": "Blood",
    "Erosion": "Erosion",
    "Erythema": "Erythematous",
    "Foreign Body": "Foreign Bodies",
    "Ileocecal valve": "Ileo-cecal valve",
    "Lymphangiectasia": "Lymphangiectasia",
    "Normal clean mucosa": "Normal",
    "Pylorus": "Pylorus",
    "Reduced Mucosal View": "Reduced Mucosal View",
    "Ulcer": "Ulcer",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str, verify_tls: bool = True):
    if dest.exists() and sha256(dest) == expected_sha:
        print(f"[ok] {dest.name} already present, checksum matches")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # datasets.simula.no does not send its intermediate certificate, so TLS
    # verification fails on standard CA bundles. Integrity is enforced by the
    # SHA-256 check below instead.
    ctx = None if verify_tls else ssl._create_unverified_context()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                bar.update(len(chunk))
    got = sha256(tmp)
    if got != expected_sha:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {dest.name}: expected {expected_sha}, got {got}")
    tmp.rename(dest)
    print(f"[ok] downloaded {dest.name}")


def extract_images(zip_path: Path, out_dir: Path, wanted: set[str], max_per_class: int | None):
    """Stream every class tarball in the zip and write the wanted JPEGs flat into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = Counter()
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".tar.gz")]
        for name in members:
            cls = Path(name).name.removesuffix(".tar.gz")
            with z.open(name) as fh, tarfile.open(fileobj=fh, mode="r|gz") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    fname = Path(m.name).name
                    if fname not in wanted:
                        continue
                    if max_per_class and written[cls] >= max_per_class:
                        break
                    dest = out_dir / fname
                    if not dest.exists():
                        dest.write_bytes(tar.extractfile(m).read())
                    written[cls] += 1
            print(f"  {cls:24s} {written[cls]:6d} images")
    return written


def build_index(meta_csv: Path, split_csvs: dict[int, Path], image_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv, sep=";")
    parts = []
    for fold, p in split_csvs.items():
        s = pd.read_csv(p)
        s["fold"] = fold
        parts.append(s)
    splits = pd.concat(parts, ignore_index=True)
    # A few frames carry two findings (e.g. Erosion seen at the Pylorus) and appear
    # once per finding in both files, so the join key is (filename, label).
    meta["label"] = meta["finding_class"].map(META_TO_SPLIT)
    meta = meta.dropna(subset=["label"]).drop_duplicates(["filename", "label"])
    df = splits.merge(meta, on=["filename", "label"], how="left", validate="one_to_one")
    missing = df[df["video_id"].isna()]
    if len(missing):
        raise RuntimeError(f"{len(missing)} split rows have no matching metadata row, "
                           f"e.g. {missing[['filename', 'label']].head(3).to_dict('records')}")
    xs, ys = df[["x1", "x2", "x3", "x4"]], df[["y1", "y2", "y3", "y4"]]
    df["x_min"], df["x_max"] = xs.min(axis=1), xs.max(axis=1)
    df["y_min"], df["y_max"] = ys.min(axis=1), ys.max(axis=1)
    df["multi_finding"] = df["filename"].duplicated(keep=False).astype(int)
    df = df[df["filename"].map(lambda f: (image_dir / f).exists())].reset_index(drop=True)
    sizes = [Image.open(image_dir / f).size for f in tqdm(df["filename"], desc="image sizes")]
    df["width"] = [w for w, _ in sizes]
    df["height"] = [h for _, h in sizes]
    keep = ["filename", "video_id", "frame_number", "fold", "label", "finding_class",
            "multi_finding", "x_min", "y_min", "x_max", "y_max", "width", "height"]
    return df[keep]


def audit(df: pd.DataFrame) -> dict:
    vids = {f: set(df.loc[df.fold == f, "video_id"]) for f in (0, 1)}
    shared = sorted(vids[0] & vids[1])
    per_shared = defaultdict(dict)
    for v in shared:
        for f in (0, 1):
            per_shared[v][f"fold{f}"] = df[(df.video_id == v) & (df.fold == f)]["label"].value_counts().to_dict()
    boxed = df[df["label"].isin(C.LESION_CLASSES) & df["x_min"].notna()]
    return {
        "n_images": int(len(df)),
        "images_per_fold": df["fold"].value_counts().sort_index().to_dict(),
        "videos_per_fold": {f: len(v) for f, v in vids.items()},
        "videos_in_both_folds": len(shared),
        "frames_from_shared_videos": {f: int(df[(df.fold == f) & df.video_id.isin(shared)].shape[0]) for f in (0, 1)},
        "shared_video_labels": per_shared,
        "rows_for_frames_with_two_findings": int(df["multi_finding"].sum()),
        "class_counts": df.groupby(["fold", "label"]).size().unstack(0, fill_value=0).to_dict(),
        "boxed_lesion_images_per_fold": boxed.groupby(["fold", "label"]).size().unstack(0, fill_value=0).to_dict(),
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in df.groupby(["width", "height"]).size().items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="extract at most N images per class tarball (smoke tests)")
    args = ap.parse_args()

    zip_path = C.RAW_DIR / "kvasir-capsule-labeled-images.zip"
    meta_path = C.RAW_DIR / "metadata.csv"
    split_paths = {f: C.RAW_DIR / f"split_{f}.csv" for f in (0, 1)}

    for f, p in split_paths.items():
        download(C.SPLIT_URL.format(fold=f), p, C.SPLIT_SHA256[f])
    download(C.METADATA_URL, meta_path, C.METADATA_SHA256)
    download(C.IMAGES_ZIP_URL, zip_path, C.IMAGES_ZIP_SHA256, verify_tls=False)

    wanted = set(pd.concat([pd.read_csv(p) for p in split_paths.values()])["filename"])
    print("Extracting images ...")
    extract_images(zip_path, C.IMAGE_DIR, wanted, args.max_per_class)

    df = build_index(meta_path, split_paths, C.IMAGE_DIR)
    df.to_csv(C.INDEX_CSV, index=False)
    print(f"[ok] wrote {C.INDEX_CSV} ({len(df)} rows)")

    report = audit(df)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = "data_audit.json" if not args.max_per_class else "data_audit_subset.json"
    (C.RESULTS_DIR / name).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("n_images", "images_per_fold", "videos_per_fold",
                                             "videos_in_both_folds", "frames_from_shared_videos",
                                             "rows_for_frames_with_two_findings")}, indent=2, default=str))


if __name__ == "__main__":
    main()
