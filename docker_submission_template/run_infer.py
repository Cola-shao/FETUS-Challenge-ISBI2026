import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

from models.model import CSV_Baseline  # 按照你现有的 UNet 实现来导入

def _sorted_h5_files(directory: Path):
    """按数字优先、再按字符串排序 H5 文件名."""
    def sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(directory.glob("*.h5"), key=sort_key)


class InferenceDataset:
    def __init__(self, base_dir: Path, patch_size=(256, 256)):
        self.base_dir = Path(base_dir)
        self.patch_size = patch_size
        images_dir = self.base_dir / "images"

        if not images_dir.exists():
            raise FileNotFoundError(f"[ERROR] images dir not found: {images_dir}")

        self.images_dir = images_dir
        self.sample_paths = _sorted_h5_files(images_dir)

        if not self.sample_paths:
            raise RuntimeError(f"[ERROR] No .h5 files found in {images_dir}")

        print(f"[INFO]images dir = {self.images_dir}")
        print(f"[INFO] Total cases = {len(self.sample_paths)}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx: int):
        img_path = self.sample_paths[idx]
        case_name = img_path.stem

        with h5py.File(img_path, "r") as img_h5:
            image_long = img_h5["image_long"][:]   # H, W
            image_trans = img_h5["image_trans"][:] # H, W

        if image_long.ndim != 2 or image_trans.ndim != 2:
            raise ValueError(
                f"Expected 2D images, got shapes: "
                f"long={image_long.shape}, trans={image_trans.shape}"
            )

        orig_size_long = image_long.shape
        orig_size_trans = image_trans.shape

        # ------ resize to patch_size ------
        ph, pw = self.patch_size

        h, w = image_long.shape
        scale_h = ph / h
        scale_w = pw / w
        image_long_resized = zoom(image_long, (scale_h, scale_w), order=1)  # 双线性

        h, w = image_trans.shape
        scale_h = ph / h
        scale_w = pw / w
        image_trans_resized = zoom(image_trans, (scale_h, scale_w), order=1)

        # 归一化到 [0,1]
        image_long_resized = image_long_resized.astype(np.float32)
        image_trans_resized = image_trans_resized.astype(np.float32)

        if image_long_resized.max() > 1.0:
            image_long_resized /= 255.0
        if image_trans_resized.max() > 1.0:
            image_trans_resized /= 255.0

        # 转为 (C,H,W) = (1,H,W)
        image_long_t = torch.from_numpy(image_long_resized).unsqueeze(0)
        image_trans_t = torch.from_numpy(image_trans_resized).unsqueeze(0)

        return {
            "image_long": image_long_t,
            "image_trans": image_trans_t,
            "case_name": case_name,
            "orig_size_long": orig_size_long,
            "orig_size_trans": orig_size_trans,
            "img_path": str(img_path),
        }



def main():
    # -------- 路径和参数（支持环境变量）--------
    DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "/workspace/weights/model.pth"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/output/preds"))
    PATCH_SIZE = (256, 256)
    NUM_SEG_CLASSES = 3  # 背景+血管+斑块
    NUM_CLS = 2          # 二分类

    print(f"[INFO] DATA_ROOT  = {DATA_ROOT}")
    print(f"[INFO] MODEL_PATH = {MODEL_PATH}")
    print(f"[INFO] OUTPUT_DIR = {OUTPUT_DIR}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"[ERROR] DATA_ROOT not found: {DATA_ROOT}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 设备 & 模型加载 --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = CSV_Baseline(in_chns=1, seg_classes=NUM_SEG_CLASSES, cls_classes=NUM_CLS)
    checkpoint = torch.load(str(MODEL_PATH), map_location=device)

    # 兼容 state_dict 直接保存或 {'state_dict': ...}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded.")

    dataset = InferenceDataset(
        base_dir=DATA_ROOT,
        patch_size=PATCH_SIZE,
    )

    print("[INFO] Start inference ...")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        case_name = sample["case_name"]
        img_long = sample["image_long"].to(device)   # (1,H,W)
        img_trans = sample["image_trans"].to(device) # (1,H,W)

        # 加 batch 维 -> (1,1,H,W)
        img_long_b = img_long.unsqueeze(0)
        img_trans_b = img_trans.unsqueeze(0)

        orig_size_long = sample["orig_size_long"]
        orig_size_trans = sample["orig_size_trans"]

        with torch.no_grad():
            seg_long, seg_trans, cls_out = model(img_long_b, img_trans_b)

        cls_pred = torch.argmax(cls_out, dim=1).cpu().numpy().squeeze()

        seg_pred_long = torch.argmax(F.softmax(seg_long, dim=1), dim=1)
        seg_pred_trans = torch.argmax(F.softmax(seg_trans, dim=1), dim=1)

        seg_pred_long_np = seg_pred_long.cpu().numpy().squeeze()
        seg_pred_trans_np = seg_pred_trans.cpu().numpy().squeeze()

        oh_long, ow_long = orig_size_long
        h_long, w_long = seg_pred_long_np.shape
        if (h_long, w_long) != (oh_long, ow_long):
            scale_long = (oh_long / h_long, ow_long / w_long)
            seg_pred_long_np = zoom(seg_pred_long_np, scale_long, order=0)

        oh_trans, ow_trans = orig_size_trans
        h_trans, w_trans = seg_pred_trans_np.shape
        if (h_trans, w_trans) != (oh_trans, ow_trans):
            scale_trans = (oh_trans / h_trans, ow_trans / w_trans)
            seg_pred_trans_np = zoom(seg_pred_trans_np, scale_trans, order=0)

        seg_pred_long_np = seg_pred_long_np.astype(np.uint8)
        seg_pred_trans_np = seg_pred_trans_np.astype(np.uint8)
        cls_pred_np = np.array(cls_pred, dtype=np.uint8)

        out_path = OUTPUT_DIR / f"{case_name}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("pred_long", data=seg_pred_long_np, dtype="uint8")
            f.create_dataset("pred_trans", data=seg_pred_trans_np, dtype="uint8")
            f.create_dataset("cls_pred", data=cls_pred_np, dtype="uint8")
            # 可选元数据
            f.attrs["case_name"] = case_name
            f.attrs["model_path"] = str(MODEL_PATH)

        if idx % 10 == 0 or idx == len(dataset) - 1:
            print(
                f"[INFO] Saved {out_path}, "
                f"pred_long shape={seg_pred_long_np.shape}, "
                f"pred_trans shape={seg_pred_trans_np.shape}, "
                f"cls_pred={int(cls_pred)}"
            )

    print(f"[INFO] Inference done. Total cases = {len(dataset)}")
    print(f"[INFO] Predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
