import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2gray
from models.model import UNet
import argparse
import logging
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ALLOWED = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],           # 4CH
    1: [0, 1, 2, 4, 8],                    # LVOT
    2: [0, 6, 8, 9, 10, 11, 12],           # RVOT
    3: [0, 9, 12, 13, 14],                 # 3VT
}

class FETUSInferDataset(Dataset):
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            self.case_list = json.load(f)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case = self.case_list[idx]
        image_h5_file = case["image"]

        with h5py.File(image_h5_file, "r") as f:
            image = f["image"][:]
            image = rgb2gray(image)
            if "view" in f:
                view = f["view"][:]
                view = int(np.array(view).reshape(-1)[0]) - 1
            else:
                view = -1  

        image_t = torch.from_numpy(image).unsqueeze(0).float()  # (1,H,W)
        view_t = torch.tensor(view, dtype=torch.long)           # ()

        return image_t, view_t, image_h5_file

def parse_thr_per_class(thr_str: str, K: int):
    if thr_str is None or thr_str.strip() == "":
        return None
    parts = [p.strip() for p in thr_str.split(",")]
    if len(parts) != K:
        raise ValueError(f"--cls-thr-per-class expects {K} values, got {len(parts)}: {thr_str}")
    thr = np.array([float(x) for x in parts], dtype=np.float32)
    return thr


def prob_to_binary(prob: np.ndarray, thr_global: float, thr_per_class: np.ndarray = None) -> np.ndarray:
    """
    prob: (K,) float32
    return: (K,) uint8 0/1
    """
    if thr_per_class is not None:
        return (prob >= thr_per_class).astype(np.uint8)
    else:
        return (prob >= float(thr_global)).astype(np.uint8)

def setup_logger(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("UniMatch Inference")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(os.path.join(save_dir, "infer_log.txt"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    if logger.handlers:
        logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device, logger: logging.Logger):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    logger.info(f"Loaded checkpoint: {ckpt_path}")


def apply_view_mask_logits(logits, view_ids, allowed_mat, fill_value=None):
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


def save_pred_h5(save_path: str, pred_mask_hw: np.ndarray, pred_label_k: np.ndarray):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        f.create_dataset("mask", data=pred_mask_hw.astype(np.uint8), compression="gzip")
        f.create_dataset("label", data=pred_label_k, compression="gzip")



def make_output_path(out_dir: str, image_h5_path: str):
    """
    xxx/abc.h5 -> out_dir/abc.h5
    """
    base = os.path.basename(image_h5_path)
    return os.path.join(out_dir, base)


@torch.no_grad()
def run_inference(model, loader, device, allowed_mat, args, logger):
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    for batch in tqdm(loader):
        image, view_oracle, image_h5_path = batch

        if isinstance(image_h5_path, (list, tuple)):
            image_h5_path = image_h5_path[0]

        image = image.to(device)  # (B,1,H,W)
        view_oracle = view_oracle.to(device).long().view(-1)  # (B,)

        B, _, H, W = image.shape

        image_rs = F.interpolate(
            image, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )

        pred_mask_logits, pred_class_out, pred_view_logits = model(image_rs)
        pred_mask_logits = F.interpolate(pred_mask_logits, (H, W), mode="bilinear", align_corners=False)

        view_pred = pred_view_logits.argmax(dim=1)  # (B,)

        if args.mask_mode == "oracle":
            use_view = torch.where(view_oracle >= 0, view_oracle, view_pred).long()
            masked_logits = apply_view_mask_logits(pred_mask_logits, use_view, allowed_mat)
        elif args.mask_mode == "pred":
            masked_logits = apply_view_mask_logits(pred_mask_logits, view_pred.long(), allowed_mat)
        else:  # "none"
            masked_logits = pred_mask_logits

        pred_mask = masked_logits.argmax(dim=1)      # (B,H,W)
        pred_prob = torch.sigmoid(pred_class_out)    # (B,K)

        assert B == 1, "For submission, please set --batch-size 1 to keep filename mapping simple."
        save_path = make_output_path(args.out_dir, image_h5_path)

        pm = pred_mask[0].detach().cpu().numpy()

        prob = pred_prob[0].detach().cpu().numpy().astype(np.float32)

        thr_pc = parse_thr_per_class(args.cls_thr_per_class, args.cls_num_classes)

        pl = prob_to_binary(prob, args.cls_thr, thr_pc).astype(np.uint8)

        save_pred_h5(save_path, pm, pl)

    logger.info(f"Saved predictions to: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser("UniMatch Inference Template (FETUS2026)")
    #--------------------------‼️--data-json do not modify ‼️--------------------------#
    parser.add_argument("--data-json", type=str, default='./data/valid_infer.json', help="json for inference (only needs image paths)")
    parser.add_argument("--ckpt", type=str, default="./weights/best.pth")   # your model checkpoint path
    parser.add_argument("--out-dir", type=str, default='./output')          # output path

    parser.add_argument("--resize_target", type=int, default=256)
    parser.add_argument("--seg_num_classes", type=int, default=15)
    parser.add_argument("--cls_num_classes", type=int, default=7)
    parser.add_argument("--view_num_classes", type=int, default=4)
    
    parser.add_argument("--label-mode", type=str, default="binary",
                    choices=["binary"],  
                    help="submission format: only binary labels are allowed")

    parser.add_argument("--cls-thr", type=float, default=0.5,
                        help="global threshold for classification when label-mode=binary")
    parser.add_argument("--cls-thr-per-class", type=str, default="",
                        help="comma-separated per-class thresholds, length=cls_num_classes; "
                            "if set, overrides --cls-thr")

    parser.add_argument("--batch-size", type=int, default=1) 
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="4")

    parser.add_argument("--mask-mode", type=str, default="oracle",
                        choices=["oracle", "pred", "none"],
                        help="oracle: use view from image_h5; pred: use view head; none: no masking")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.out_dir)
    logger.info(str(args))
    logger.info(f"Device: {device}")

    # allowed mat for seg
    V, C = args.view_num_classes, args.seg_num_classes
    allowed_mat = torch.zeros((V, C), dtype=torch.bool, device=device)
    for v in range(V):
        allowed_mat[v, ALLOWED[v]] = True

    model = UNet(
        in_chns=1,
        seg_class_num=args.seg_num_classes,
        cls_class_num=args.cls_num_classes,
        view_num_classes=args.view_num_classes
    ).to(device)
    logger.info("Total params: {:.1f}M".format(count_params(model)))
    load_checkpoint(model, args.ckpt, device, logger)

    dataset = FETUSInferDataset(args.data_json)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"Samples: {len(dataset)}")

    run_inference(model, loader, device, allowed_mat, args, logger)


if __name__ == "__main__":
    main()