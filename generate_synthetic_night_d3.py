"""
Synthetic Night Generator for Dataset-3
======================================
Transforms daytime road images in Dataset-3_yolo into photometrically realistic
synthetic night images (modeling headlight beam, ambient illumination, daylight removal,
auto-exposure, sensor shot/read noise, and scotopic desaturation).
Bounding boxes remain 100% valid as geometric coordinates are untouched.
"""

from pathlib import Path
import cv2
import numpy as np
import shutil
from tqdm import tqdm

W_BGR = np.array([0.114, 0.587, 0.299], np.float32)    # BGR2GRAY weights
GAMMA = 2.2
HEAD_TINT = np.array([0.80, 0.93, 1.00], np.float32)   # warm headlight
AMB_TINT = np.array([1.20, 1.00, 0.85], np.float32)    # cool bluish ambient

PRESETS = {
    "dusk":       dict(target_mean=70, peak=1500, read_std=0.004, beam=1.2, ambient=0.25),
    "night":      dict(target_mean=40, peak=800,  read_std=0.006, beam=1.8, ambient=0.12),
    "deep_night": dict(target_mean=22, peak=400,  read_std=0.010, beam=2.4, ambient=0.07),
}

def mean_luma(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

def _exposure_for_target(lin, target):
    small = cv2.resize(lin, (96, 96), interpolation=cv2.INTER_AREA)
    lo, hi = -14.0, 14.0
    for _ in range(32):
        mid = (lo + hi) / 2
        enc = np.clip(small * (2.0 ** mid), 0, 1) ** (1 / GAMMA)
        if 255.0 * float((enc @ W_BGR).mean()) < target:
            lo = mid
        else:
            hi = mid
    return 2.0 ** ((lo + hi) / 2)

def to_night(img, rng, target_mean=40, peak=800, read_std=0.006, beam=1.8, ambient=0.12,
             cx=0.5, cy=1.0, illum_strength=0.85, illum_sigma_frac=0.08, desat=0.3, max_ratio=0.7):
    h, w = img.shape[:2]
    before = mean_luma(img)
    target = float(min(target_mean, max_ratio * before))

    x = (img.astype(np.float32) / 255.0) ** GAMMA

    # 1. Daylight removal
    lum = x @ W_BGR
    illum = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(h, w) * illum_sigma_frac)
    base = max(float(illum.mean()), 1e-4)
    shading = np.clip(illum / base, 0.25, 4.0) ** illum_strength
    reflect = np.clip(x / (shading[..., None] * base), 0, 4.0)

    # 2. Headlight cone + ambient light
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / max(w - 1, 1) - cx) / 0.5) ** 2 + ((yy / max(h - 1, 1) - cy) / 0.9) ** 2
    cone = np.exp(-beam * d)[..., None].astype(np.float32)
    light = ambient * AMB_TINT + (1.0 - ambient) * cone * HEAD_TINT
    unit = reflect * light

    # 3. Auto-exposure to target luminance
    lit = np.clip(unit * _exposure_for_target(unit, target), 0, 1)

    # 4. Sensor noise: Poisson (shot) + Gaussian (read)
    lit = rng.poisson(lit * peak).astype(np.float32) / peak
    lit = lit + rng.normal(0, read_std, lit.shape).astype(np.float32)
    enc = np.clip(lit, 0, 1) ** (1 / GAMMA)

    # 5. Low-light desaturation
    out = (enc * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
    out = np.clip((1 - desat) * out + desat * gray, 0, 255).astype(np.uint8)
    return out

def sample_params(rng, looks=("dusk", "night")):
    name = str(rng.choice(list(looks)))
    p = dict(PRESETS[name])
    p["target_mean"] *= rng.uniform(0.8, 1.25)
    p["peak"] *= rng.uniform(0.7, 1.4)
    p["read_std"] *= rng.uniform(0.7, 1.6)
    p["beam"] *= rng.uniform(0.8, 1.3)
    p["cx"] = float(np.clip(0.5 + rng.uniform(-0.15, 0.15), 0.25, 0.75))
    p["cy"] = float(np.clip(1.0 + rng.uniform(-0.10, 0.10), 0.85, 1.15))
    return p

def transform_dataset(src_root, dst_root, seed=42):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    rng = np.random.default_rng(seed)

    print(f"Transforming {src_root} -> {dst_root}...")
    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(list(src_img_dir.glob("*.jpg")))
        print(f"Processing {split}: {len(images)} images...")

        for img_p in tqdm(images, desc=split):
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            params = sample_params(rng)
            night_img = to_night(img, rng, **params)

            cv2.imwrite(str(dst_img_dir / img_p.name), night_img)

            # Copy label exactly
            lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy(str(lbl_p), str(dst_lbl_dir / lbl_p.name))

    # Write data.yaml
    yaml_content = f"""path: {dst_root.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pothole']
"""
    (dst_root / "data.yaml").write_text(yaml_content)
    print(f"Completed! Synthetic night dataset created at {dst_root}")

if __name__ == "__main__":
    src = Path("Dataset-3_yolo")
    dst = Path("Dataset-3_synthetic_night")
    transform_dataset(src, dst)

