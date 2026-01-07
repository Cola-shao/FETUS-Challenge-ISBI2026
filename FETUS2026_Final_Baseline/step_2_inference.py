# inference_unimatch.py
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.fetus_infer import FETUSInferDataset
from model.unet import UNet


DEFAULT_SEG_ALLOWED: Dict[int, List[int]] = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],           # 4CH
    1: [0, 1, 2, 4, 8],                    # LVOT
    2: [0, 6, 8, 9, 10, 11, 12],           # RVOT
    3: [0, 9, 12, 13, 14],                 # 3VT
}


def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("UniMatch-Infer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(save_dir, "infer_log.txt"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_checkpoint_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device, logger: logging.Logger):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded checkpoint: {ckpt_path}")


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


def load_seg_allowed(seg_allowed_arg: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(seg_allowed_arg)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("--seg-allowed must be a JSON object: {view_id: [class_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"seg_allowed[{k}] must be a list.")
        out[kk] = [int(x) for x in v]
    return out


def build_allowed_mat(device: torch.device, seg_allowed: Dict[int, List[int]], num_views: int, num_classes: int) -> torch.Tensor:
    mat = torch.zeros((num_views, num_classes), dtype=torch.bool, device=device)
    for v in range(num_views):
        if v not in seg_allowed:
            raise ValueError(f"seg_allowed is missing view {v}")
        cls_ids = seg_allowed[v]
        for c in cls_ids:
            if c < 0 or c >= num_classes:
                raise ValueError(f"seg_allowed[{v}] contains invalid class id {c} for num_classes={num_classes}")
        mat[v, cls_ids] = True
    return mat


def apply_view_mask_logits(
    logits: torch.Tensor,
    view_ids: torch.Tensor,
    allowed_mat: torch.Tensor,
    fill_value: Optional[float] = None,
) -> torch.Tensor:
    """
    logits: (B,C,H,W)
    view_ids: (B,) long
    allowed_mat: (V,C) bool
    """
    invalid = ~allowed_mat[view_ids]              # (B,C)
    invalid = invalid.unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)
    if fill_value is None:
        fill_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(invalid, fill_value)


def parse_thr_per_class(thr_str: str, K: int) -> Optional[np.ndarray]:
    if thr_str is None or thr_str.strip() == "":
        return None
    parts = [p.strip() for p in thr_str.split(",")]
    if len(parts) != K:
        raise ValueError(f"--cls-thr-per-class expects {K} values, got {len(parts)}: {thr_str}")
    thr = np.array([float(x) for x in parts], dtype=np.float32)
    if np.any(thr < 0.0) or np.any(thr > 1.0):
        raise ValueError(f"Per-class thresholds must be in [0,1], got: {thr.tolist()}")
    return thr


def prob_to_binary(prob: np.ndarray, thr_global: float, thr_per_class: Optional[np.ndarray]) -> np.ndarray:
    if thr_per_class is not None:
        return (prob >= thr_per_class).astype(np.uint8)
    return (prob >= float(thr_global)).astype(np.uint8)


def make_output_path(out_dir: str, image_h5_path: str) -> str:
    return os.path.join(out_dir, os.path.basename(image_h5_path))


def save_pred_h5(save_path: str, pred_mask_hw: np.ndarray, pred_label_k: np.ndarray):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mask", data=pred_mask_hw.astype(np.uint8), compression="gzip")
        f.create_dataset("label", data=pred_label_k.astype(np.uint8), compression="gzip")


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    allowed_mat: torch.Tensor,
    args,
    logger: logging.Logger,
):
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    thr_pc = parse_thr_per_class(args.cls_thr_per_class, args.cls_num_classes)
    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    for batch in tqdm(loader, total=len(loader), desc="Infer"):
        # Expected dataset output: (image, view_oracle, image_h5_path)
        image, view_oracle, image_h5_path = batch

        # image_h5_path is list[str] after default collate
        if isinstance(image_h5_path, (tuple, list)):
            paths = list(image_h5_path)
        else:
            paths = [str(image_h5_path)]

        image = image.to(device, non_blocking=True)  # (B,1,H,W)

        B, _, H, W = image.shape
        if B != len(paths):
            raise RuntimeError(f"Batch size mismatch: B={B}, len(paths)={len(paths)}")

        # view is required only when mask_mode == oracle
        if args.mask_mode == "oracle":
            view_oracle = view_oracle.to(device, non_blocking=True).long().view(-1)
            if view_oracle.numel() != B:
                raise RuntimeError(f"view_oracle size mismatch: got {view_oracle.numel()}, expected {B}")
            if (view_oracle.min() < 0) or (view_oracle.max() >= args.view_num_classes):
                raise ValueError(f"Invalid view id in batch: {view_oracle.detach().cpu().tolist()}")

        image_rs = F.interpolate(
            image, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # UNet returns (seg_logits, cls_logits)
            pred_mask_logits_rs, pred_class_out = model(image_rs)

        pred_mask_logits = F.interpolate(
            pred_mask_logits_rs, (H, W),
            mode="bilinear", align_corners=False
        )

        if args.mask_mode == "oracle":
            masked_logits = apply_view_mask_logits(pred_mask_logits, view_oracle, allowed_mat)
        elif args.mask_mode == "none":
            masked_logits = pred_mask_logits
        else:
            raise ValueError(f"Unknown mask_mode: {args.mask_mode}")

        pred_mask = masked_logits.argmax(dim=1)      # (B,H,W)
        pred_prob = torch.sigmoid(pred_class_out)    # (B,K)

        pred_mask_np = pred_mask.detach().cpu().numpy().astype(np.uint8)
        pred_prob_np = pred_prob.detach().cpu().numpy().astype(np.float32)

        for b in range(B):
            save_path = make_output_path(args.out_dir, paths[b])
            pm = pred_mask_np[b]
            prob = pred_prob_np[b]

            pl = prob_to_binary(prob, args.cls_thr, thr_pc)

            if (not args.overwrite) and os.path.exists(save_path):
                logger.warning(f"Skip existing (overwrite disabled): {save_path}")
                continue

            save_pred_h5(save_path, pm, pl)

    logger.info(f"Saved predictions to: {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser("UniMatch Inference (FETUS2026)")
    p.add_argument("--data-json", type=str, default='./data/valid.json',
                   help="JSON for inference (image paths; view ids required if --mask-mode oracle)")
    p.add_argument("--ckpt", type=str, default='./checkpoints/best.pth', help="checkpoint path (.pth)")
    p.add_argument("--out-dir", type=str, default="./output")

    p.add_argument("--resize-target", type=int, default=256)
    p.add_argument("--seg-num-classes", type=int, default=15)
    p.add_argument("--cls-num-classes", type=int, default=7)
    p.add_argument("--view-num-classes", type=int, default=4)

    p.add_argument("--label-mode", type=str, default="binary", choices=["binary"],
                   help="submission format: only binary labels are allowed")

    p.add_argument("--cls-thr", type=float, default=0.5,
                   help="global threshold for classification when label-mode=binary")
    p.add_argument("--cls-thr-per-class", type=str, default="",
                   help="comma-separated per-class thresholds, length=cls_num_classes; if set, overrides --cls-thr")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gpu", type=str, default="0")

    # No view head in current UNet, so only oracle / none are supported.
    p.add_argument("--mask-mode", type=str, default="oracle", choices=["oracle", "none"],
                   help="oracle: mask seg logits with dataset-provided view; none: no masking")

    p.add_argument("--seg-allowed", type=str, default=None,
                   help="JSON string or .json path for segmentation allowed mapping {view_id:[class_ids...]}")

    p.add_argument("--amp", action="store_true", help="enable autocast")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    p.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(args.out_dir)
    logger.info(str(args))
    logger.info(f"Device: {device}")

    seg_allowed = load_seg_allowed(args.seg_allowed, DEFAULT_SEG_ALLOWED)
    allowed_mat = build_allowed_mat(device, seg_allowed, args.view_num_classes, args.seg_num_classes)

    model = UNet(
        in_chns=1,
        seg_class_num=args.seg_num_classes,
        cls_class_num=args.cls_num_classes,
        view_num_classes=args.view_num_classes,
    ).to(device)
    logger.info(f"Total params: {count_params_m(model):.1f}M")

    load_checkpoint_strict(model, args.ckpt, device, logger)

    dataset = FETUSInferDataset(args.data_json)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(f"Samples: {len(dataset)}")

    run_inference(model, loader, device, allowed_mat, args, logger)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
