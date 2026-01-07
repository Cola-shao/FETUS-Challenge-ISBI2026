# evaluate_unimatch.py
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from dataset.fetus_eval import FETUSEvalDataset


DEFAULT_CLS_ALLOWED: Dict[int, List[int]] = {
    0: [0, 1],
    1: [0, 2, 3],
    2: [4, 5],
    3: [2, 5, 6],
}

def parse_args():
    p = argparse.ArgumentParser("UniMatch Evaluate (requires GT)")
    p.add_argument("--valid-json", type=str, default="./data/valid.json")
    p.add_argument("--pred-dir", type=str, default="./output")
    p.add_argument("--save-dir", type=str, default="./eval_results")

    p.add_argument("--seg-num-classes", type=int, default=15)
    p.add_argument("--cls-num-classes", type=int, default=7)
    p.add_argument("--nsd-tol", type=float, default=2.0)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--cls-allowed", type=str, default=None,
                   help="JSON string or .json path: {view_id:[cls_ids...]}. If not set, uses default mapping.")
    return p.parse_args()


def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("UniMatch-Eval")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(save_dir, "eval_log.txt"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_json_arg(s: Optional[str]):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cls_allowed(arg: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(arg)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("--cls-allowed must be a JSON object: {view_id: [cls_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"cls_allowed[{k}] must be a list.")
        out[kk] = [int(x) for x in v]
    return out


def build_allowed_mask_np(views: np.ndarray, cls_allowed: Dict[int, List[int]], num_classes: int) -> np.ndarray:
    """
    views: (N,) int in [0..V-1]
    returns: (N,K) {0,1}
    """
    N = int(views.shape[0])
    K = int(num_classes)
    mask = np.zeros((N, K), dtype=np.uint8)
    for i in range(N):
        v = int(views[i])
        for k in cls_allowed.get(v, []):
            if 0 <= k < K:
                mask[i, k] = 1
    return mask


def masked_f1_from_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    views: np.ndarray,
    cls_allowed: Dict[int, List[int]],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    y_true/y_pred: (N,K) in {0,1}
    Computes per-class F1 on samples where allowed(view,k)=1, then macro average.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    N, K = y_true.shape
    allowed = build_allowed_mask_np(views, cls_allowed, num_classes=K)  # (N,K)

    per_class_f1 = np.zeros(K, dtype=np.float32)
    support = np.zeros(K, dtype=np.int32)

    for k in range(K):
        m = allowed[:, k] == 1
        support[k] = int(m.sum())
        if support[k] == 0:
            per_class_f1[k] = 0.0
        else:
            per_class_f1[k] = float(f1_score(y_true[m, k], y_pred[m, k], zero_division=0))

    macro_f1 = float(per_class_f1.mean()) if K > 0 else 0.0
    return macro_f1, per_class_f1, support


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1)
    return mask ^ eroded


def nsd_binary(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    s_pred = _surface(pred)
    s_gt = _surface(gt)

    dt_gt = distance_transform_edt(~s_gt)
    dt_pred = distance_transform_edt(~s_pred)

    d_pred_to_gt = dt_gt[s_pred]
    d_gt_to_pred = dt_pred[s_gt]

    p = float((d_pred_to_gt <= tol).mean()) if d_pred_to_gt.size else 0.0
    g = float((d_gt_to_pred <= tol).mean()) if d_gt_to_pred.size else 0.0
    return (p + g) / 2.0


def seg_metrics_accumulate(
    pred_hw: np.ndarray,
    gt_hw: np.ndarray,
    num_classes: int,
    tol: float,
    dice_sum: np.ndarray,
    nsd_sum: np.ndarray,
    cnt: np.ndarray,
):
    """
    Accumulate per-class Dice/NSD over foreground classes 1..C-1.
    """
    for cls in range(1, num_classes):
        gt_bin = (gt_hw == cls)
        pred_bin = (pred_hw == cls)

        union = int(pred_bin.sum() + gt_bin.sum())
        if union == 0:
            continue

        inter = int((pred_bin & gt_bin).sum())
        dice_sum[cls - 1] += (2.0 * inter) / (union + 1e-8)
        nsd_sum[cls - 1] += nsd_binary(pred_bin, gt_bin, tol=tol)
        cnt[cls - 1] += 1


def load_pred_by_image(pred_dir: str, image_h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    base = os.path.basename(image_h5_path)
    path = os.path.join(pred_dir, base)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file: {path} (from image {image_h5_path})")

    with h5py.File(path, "r") as f:
        if "mask" not in f or "label" not in f:
            keys = list(f.keys())
            raise KeyError(f"Prediction file {path} must contain datasets ['mask','label'], got: {keys}")
        pm = f["mask"][:]
        pl = f["label"][:]
    return pm, pl


def main():
    args = parse_args()
    logger = setup_logger(args.save_dir)
    logger.info(str(args))

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"--pred-dir not found: {args.pred_dir}")

    cls_allowed = load_cls_allowed(args.cls_allowed, DEFAULT_CLS_ALLOWED)
    logger.info(f"cls_allowed: {cls_allowed}")

    dataset = FETUSEvalDataset(args.valid_json)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(f"Valid samples: {len(dataset)}")

    C = int(args.seg_num_classes)
    K = int(args.cls_num_classes)
    tol = float(args.nsd_tol)

    dice_sum = np.zeros(C - 1, dtype=np.float64)
    nsd_sum = np.zeros(C - 1, dtype=np.float64)
    cnt = np.zeros(C - 1, dtype=np.int64)

    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    views_list: List[int] = []

    for batch in tqdm(loader, total=len(loader), desc="Eval"):
        # Expected: (image, view, mask, label, image_path)
        view = np.asarray(batch[1]).astype(np.int32).reshape(-1)      # (B,)
        gt_mask = np.asarray(batch[2]).astype(np.int32)               # (B,H,W)
        gt_label = np.asarray(batch[3]).astype(np.int32)              # (B,K)
        image_h5_path = batch[4]                                      # list[str] or str

        B = int(gt_mask.shape[0])
        if gt_label.shape[0] != B:
            raise RuntimeError(f"GT label batch mismatch: gt_mask B={B}, gt_label={gt_label.shape[0]}")

        # Normalize paths to list[str]
        if isinstance(image_h5_path, (list, tuple)):
            paths = list(image_h5_path)
        else:
            paths = [str(image_h5_path)]

        if len(paths) != B:
            raise RuntimeError(f"Path batch mismatch: B={B}, len(paths)={len(paths)}")

        for b in range(B):
            img_path = paths[b]
            pred_mask, pred_label = load_pred_by_image(args.pred_dir, img_path)

            gt_hw = gt_mask[b]
            if pred_mask.shape != gt_hw.shape:
                raise ValueError(
                    f"Seg shape mismatch for {os.path.basename(img_path)}: pred {pred_mask.shape} vs gt {gt_hw.shape}"
                )

            seg_metrics_accumulate(
                pred_hw=pred_mask.astype(np.int32),
                gt_hw=gt_hw.astype(np.int32),
                num_classes=C,
                tol=tol,
                dice_sum=dice_sum,
                nsd_sum=nsd_sum,
                cnt=cnt,
            )

            pred_label = np.asarray(pred_label)
            if pred_label.ndim != 1 or pred_label.shape[0] != K:
                raise ValueError(
                    f"Cls shape mismatch for {os.path.basename(img_path)}: pred {pred_label.shape} vs expected ({K},)"
                )

            pred_bin = (pred_label > 0.5).astype(np.int32)  # robust: pred should already be 0/1
            y_true_list.append(gt_label[b].astype(np.int32))
            y_pred_list.append(pred_bin.astype(np.int32))
            views_list.append(int(view[b]))

    dice_class = 100.0 * dice_sum / np.maximum(cnt, 1)
    nsd_class = 100.0 * nsd_sum / np.maximum(cnt, 1)

    valid_fg = cnt > 0
    mean_dice = float(dice_class[valid_fg].mean()) if valid_fg.any() else 0.0
    mean_nsd = float(nsd_class[valid_fg].mean()) if valid_fg.any() else 0.0
    
    y_true_all = np.stack(y_true_list, axis=0) if y_true_list else np.zeros((0, K), dtype=np.int32)
    y_pred_all = np.stack(y_pred_list, axis=0) if y_pred_list else np.zeros((0, K), dtype=np.int32)
    views_all = np.array(views_list, dtype=np.int32) if views_list else np.zeros((0,), dtype=np.int32)

    macro_f1, per_class_f1, support = masked_f1_from_binary(y_true_all, y_pred_all, views_all, cls_allowed)

    logger.info("====== Seg Metrics ======")
    for cls_idx in range(C - 1):
        logger.info(
            f"Class[{cls_idx+1:02d}] Dice={dice_class[cls_idx]:.2f}, NSD={nsd_class[cls_idx]:.2f} (N={cnt[cls_idx]})"
        )
    logger.info(f"MeanDice: {mean_dice:.2f}")
    logger.info(f"MeanNSD : {mean_nsd:.2f}")

    logger.info("====== Cls Metrics (View-masked, Binary) ======")
    logger.info(f"Macro-F1: {macro_f1:.4f}")
    for k in range(K):
        logger.info(f"Class[{k}] F1={per_class_f1[k]:.4f} | support={support[k]}")

    result = {
        "mean_dice": mean_dice,
        "mean_nsd": mean_nsd,
        "masked_macro_f1": macro_f1,
        "masked_per_class_f1": per_class_f1.tolist(),
        "masked_support": support.tolist(),
        "dice_class": dice_class.tolist(),
        "nsd_class": nsd_class.tolist(),
        "cnt": cnt.tolist(),
        "seg_num_classes": C,
        "cls_num_classes": K,
        "nsd_tol": tol,
        "cls_allowed": cls_allowed,
        "pred_dir": os.path.abspath(args.pred_dir),
        "valid_json": os.path.abspath(args.valid_json),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    summary_txt = os.path.join(args.save_dir, "summary.txt")
    summary_json = os.path.join(args.save_dir, "summary.json")

    with open(summary_txt, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved summary: {summary_txt}")
    logger.info(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
