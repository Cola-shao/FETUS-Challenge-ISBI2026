import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.fetus import FETUSSemiDataset
from model.unet import UNet
from util.utils import (
    AverageMeter,
    DiceLoss,
    apply_view_mask_logits,
    build_allowed_mat,
    build_same_view_perm,
    compute_pos_weight_from_loader,
    count_params,
    load_pretrained_flexible,
    log_train_tb,
    log_val_perclass_tb,
    log_val_tb,
    masked_bce_with_logits,
    masked_metrics_with_threshold_search,
    masked_mse,
    nsd_binary,
    update_meters,
)

DEFAULT_SEG_ALLOWED = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],           # 4CH
    1: [0, 1, 2, 4, 8],                    # LVOT
    2: [0, 6, 8, 9, 10, 11, 12],           # RVOT
    3: [0, 9, 12, 13, 14],                 # 3VT
}

DEFAULT_CLS_ALLOWED = {
    0: [0, 1],
    1: [0, 2, 3],
    2: [4, 5],
    3: [2, 5, 6],
}

DEFAULT_LOSS_WEIGHTS = {
    "x_seg": 1.0,
    "x_cls": 1.0,
    "x_fp_cls": 1.0,

    "u_s1_seg": 0.25,
    "u_s2_seg": 0.25,
    "u_w_fp_seg": 0.5,

    "u_s1_mix_seg": 0.25,
    "u_s2_mix_seg": 0.25,

    "u_pseudo_cls": 1.0,
    "u_w_mix_cls": 0.25,
    "u_w_mix_fp_cls": 0.5,
}


def _load_json_arg(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)


def load_allowed_mapping(value: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(value)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("Allowed mapping must be a JSON object: {view_id: [class_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"Allowed[{k}] must be a list of ints.")
        out[kk] = [int(x) for x in v]
    return out


def load_loss_weights(value: Optional[str], default: Dict[str, float]) -> Dict[str, float]:
    raw = _load_json_arg(value)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("loss-weights must be a JSON object: {name: weight}")
    out = dict(default)
    for k, v in raw.items():
        if k not in out:
            raise ValueError(f"Unknown loss weight key: {k}. Allowed keys: {sorted(out.keys())}")
        out[k] = float(v)
    return out

def build_model(args, device):
    if args.model == "unet":
        model = UNet(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            view_num_classes=args.view_num_classes,
        )
    elif args.model == "echocare":
        from model.Echocare import Echocare_UniMatch  # According to your project path
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            view_num_classes=args.view_num_classes,
            ssl_checkpoint=args.ssl_ckpt,  # You need to add --ssl-ckpt in args
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device)

def build_optimizer(args, model):
    opt_name = args.opt or args.model

    if opt_name == "unet":
        optimizer = SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        base_lrs = [args.base_lr]   # Only 1 group
        return optimizer, base_lrs

    if opt_name == "echocare":
        base_backbone_lr = 1e-4
        base_head_lr = 1e-3

        backbone_params, head_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "Swin_encoder" in n:      # You defined encoder as backbone
                backbone_params.append(p)
            else:
                head_params.append(p)
        print("backbone params:", sum(p.numel() for p in backbone_params)/1e6, "M")
        print("head params:", sum(p.numel() for p in head_params)/1e6, "M")

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": base_backbone_lr},
                {"params": head_params, "lr": base_head_lr},
            ],
            weight_decay=0.01,
        )
        base_lrs = [base_backbone_lr, base_head_lr]
        return optimizer, base_lrs

    raise ValueError(f"Unknown opt preset: {opt_name}")

def step_poly_lr(optimizer, base_lrs, iters, total_iters, power=0.9):
    factor = (1 - iters / total_iters) ** power
    for g, base_lr in zip(optimizer.param_groups, base_lrs):
        g["lr"] = base_lr * factor

def parse_args():
    p = argparse.ArgumentParser(
        description="UniMatch baseline for FETUS 2026 Challenge (http://119.29.231.17:90/)"
    )
    
    p.add_argument("--model", type=str, default="echocare", choices=["unet", "echocare"])
    p.add_argument("--opt", type=str, default='echocare', choices=[None, "unet", "echocare"])

    p.add_argument("--ssl-ckpt", type=str, default='./pretrained_weights/echocare_encoder.pth')  # for echocare
    
    p.add_argument("--train-labeled-json", type=str, default="data/train_labeled.json")
    p.add_argument("--train-unlabeled-json", type=str, default="data/train_unlabeled.json")
    p.add_argument("--valid-labeled-json", type=str, default="data/valid.json")

    p.add_argument("--train-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8) # 32 for unet; 8 for echocare
    p.add_argument("--base-lr", type=float, default=0.001)
    p.add_argument("--conf-thresh", type=float, default=0.9)

    p.add_argument("--seg-num-classes", type=int, default=15)
    p.add_argument("--cls-num-classes", type=int, default=7)
    p.add_argument("--view-num-classes", type=int, default=4)

    p.add_argument("--resize-target", type=int, default=256)
    p.add_argument("--save-path", type=str, default="./checkpoints_baseline_echocare")
    
    p.add_argument("--tb-logdir", type=str, default=None)
    p.add_argument("--tb-iter-freq", type=int, default=20)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--no-hard-view-mask", action="store_true")

    # Configurable masks / thresholds / weights (JSON string or JSON file path)
    p.add_argument(
        "--seg-allowed",
        type=str,
        default=None,
        help='JSON string or .json path for segmentation allowed mapping, e.g. \'{"0":[0,1], "1":[0,2]}\'',
    )
    p.add_argument(
        "--cls-allowed",
        type=str,
        default=None,
        help='JSON string or .json path for classification allowed mapping, e.g. \'{"0":[0,1], "1":[2,3]}\'',
    )
    p.add_argument("--pseudo-tau-pos", type=float, default=0.8)
    p.add_argument("--pseudo-tau-neg", type=float, default=0.8)
    p.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help='JSON string or .json path for loss weights. Keys: '
             + ",".join(sorted(DEFAULT_LOSS_WEIGHTS.keys())),
    )

    return p.parse_args()


def setup_logger(save_path: str) -> logging.Logger:
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger("UniMatch")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_seg_allowed_mat(device, seg_allowed: Dict[int, List[int]], num_views: int, num_classes: int) -> torch.Tensor:
    mat = torch.zeros((num_views, num_classes), dtype=torch.bool, device=device)
    for v in range(num_views):
        if v not in seg_allowed:
            raise ValueError(f"seg_allowed missing view {v}.")
        mat[v, seg_allowed[v]] = True
    return mat


def poly_lr(base_lr: float, iters: int, total_iters: int, power: float = 0.9) -> float:
    return base_lr * (1 - iters / total_iters) ** power


def maybe_resume(model, optimizer, scaler, ckpt_path: str, logger: logging.Logger):
    if not os.path.exists(ckpt_path):
        return -1, 0.0, -1, 0

    ckpt = load_pretrained_flexible(model, ckpt_path, logger=logger, key="model")
    start_epoch = int(ckpt.get("epoch", -1))
    best_score = float(ckpt.get("previous_best", 0.0))
    best_epoch = int(ckpt.get("best_epoch", -1))
    global_step = int(ckpt.get("global_step", 0))

    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("[Resume] Optimizer state loaded.")
        except ValueError as e:
            logger.warning(f"[Resume] Skip optimizer state (incompatible): {e}")

    if scaler is not None and ckpt.get("scaler", None) is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            logger.info("[Resume] GradScaler loaded.")
        except Exception as e:
            logger.warning(f"[Resume] Skip GradScaler (incompatible): {e}")

    logger.info(f"[Resume] epoch={start_epoch}, best_score={best_score:.4f}, best_epoch={best_epoch}, global_step={global_step}")
    return start_epoch, best_score, best_epoch, global_step


@torch.no_grad()
def teacher_pseudo(
    model,
    image_u_w_mix,
    view_u_mix,
    allowed_seg_mat,
    allowed_cls_mat,
    use_hard_view_mask: bool,
    tau_pos: float,
    tau_neg: float,
):
    model.eval()

    pred_u_w_mix, cls_u_w_mix = model(image_u_w_mix)
    p_t = torch.sigmoid(cls_u_w_mix)

    conf_mask = ((p_t >= tau_pos) | (p_t <= 1 - tau_neg)).float()
    mask_u_cls = conf_mask * allowed_cls_mat[view_u_mix]

    if use_hard_view_mask:
        pred_u_w_mix = apply_view_mask_logits(pred_u_w_mix, view_u_mix, allowed_seg_mat)

    conf_u_w_mix = pred_u_w_mix.float().softmax(dim=1).max(dim=1)[0]
    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

    return p_t, mask_u_cls, conf_u_w_mix, mask_u_w_mix


def train_one_epoch(
    args,
    model,
    optimizer,
    scaler,
    device,
    train_loader_l,
    train_loader_u,
    train_loader_u_mix,
    allowed_seg_mat,
    allowed_cls_mat,
    pos_weight,
    loss_w: Dict[str, float],
    writer,
    logger,
    epoch: int,
    base_lrs: List,
    global_step: int,
    total_iters: int,
):
    model.train()

    meters = {
        "loss": AverageMeter(),
        "x_seg": AverageMeter(),
        "x_cls": AverageMeter(),
        "x_fp_cls": AverageMeter(),
        "u_s1_seg": AverageMeter(),
        "u_s2_seg": AverageMeter(),
        "u_w_fp_seg": AverageMeter(),
        "u_s1_mix_seg": AverageMeter(),
        "u_s2_mix_seg": AverageMeter(),
        "u_pseudo_cls": AverageMeter(),
        "u_w_mix_cls": AverageMeter(),
        "u_w_mix_fp_cls": AverageMeter(),
        "seg_mask_ratio": AverageMeter(),
        "cls_mask_ratio": AverageMeter(),
    }

    criterion_seg_ce = torch.nn.CrossEntropyLoss()
    criterion_seg_dice = DiceLoss(n_classes=args.seg_num_classes)

    use_hard_view_mask = not args.no_hard_view_mask
    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    iters_per_epoch = len(train_loader_u)
    loader = zip(train_loader_l, train_loader_u, train_loader_u_mix)

    for i, (
        (image_x, view_x, mask_x, class_label_x),
        (image_u_w, view_u, image_u_s1, image_u_s2, cutmix_box1, cutmix_box2),
        (image_u_w_mix, view_u_mix, image_u_s1_mix, image_u_s2_mix, _, _),
    ) in enumerate(loader):
        image_x = image_x.to(device)
        view_x = view_x.to(device).long().view(-1)
        mask_x = mask_x.to(device)
        class_label_x = class_label_x.to(device)

        image_u_w = image_u_w.to(device)
        view_u = view_u.to(device).long().view(-1)
        image_u_s1 = image_u_s1.to(device)
        image_u_s2 = image_u_s2.to(device)
        cutmix_box1 = cutmix_box1.to(device)
        cutmix_box2 = cutmix_box2.to(device)

        image_u_w_mix = image_u_w_mix.to(device)
        view_u_mix = view_u_mix.to(device).long().view(-1)
        image_u_s1_mix = image_u_s1_mix.to(device)
        image_u_s2_mix = image_u_s2_mix.to(device)

        perm = build_same_view_perm(view_u, view_u_mix)
        image_u_w_mix = image_u_w_mix[perm]
        image_u_s1_mix = image_u_s1_mix[perm]
        image_u_s2_mix = image_u_s2_mix[perm]
        view_u_mix = view_u_mix[perm]

        mask_x_allowed = allowed_cls_mat[view_x]

        with torch.no_grad():
            p_t, mask_u_cls, conf_u_w_mix, mask_u_w_mix = teacher_pseudo(
                model=model,
                image_u_w_mix=image_u_w_mix,
                view_u_mix=view_u_mix,
                allowed_seg_mat=allowed_seg_mat,
                allowed_cls_mat=allowed_cls_mat,
                use_hard_view_mask=use_hard_view_mask,
                tau_pos=args.pseudo_tau_pos,
                tau_neg=args.pseudo_tau_neg,
            )

        m1 = cutmix_box1.unsqueeze(1).expand_as(image_u_s1) == 1
        m2 = cutmix_box2.unsqueeze(1).expand_as(image_u_s2) == 1
        image_u_s1 = image_u_s1.clone()
        image_u_s2 = image_u_s2.clone()
        image_u_s1[m1] = image_u_s1_mix[m1]
        image_u_s2[m2] = image_u_s2_mix[m2]

        num_l = image_x.shape[0]
        num_u = image_u_w.shape[0]
        
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            (preds, preds_fp), (preds_class, preds_class_fp) = model(torch.cat((image_x, image_u_w)), True)

            pred_x, pred_u_w = preds.split([num_l, num_u])
            _, pred_u_w_fp = preds_fp.split([num_l, num_u])

            pred_x_class, _ = preds_class.split([num_l, num_u])
            pred_x_class_fp, _ = preds_class_fp.split([num_l, num_u])

            pred_u_s_outs, _ = model(torch.cat((image_u_s1, image_u_s2)))
            pred_u_s1, pred_u_s2 = pred_u_s_outs.chunk(2)

            pred_u_s1_mix, pred_class_u_s1_mix = model(image_u_s1_mix)
            pred_u_s2_mix, pred_class_u_s2_mix = model(image_u_s2_mix)

            (_, _), (pred_class_u_w_mix, pred_class_u_w_mix_fp) = model(image_u_w_mix, True)

        pred_u_w_det = pred_u_w.detach()
        if use_hard_view_mask:
            pred_u_w_det = apply_view_mask_logits(pred_u_w_det, view_u, allowed_seg_mat)

        conf_u_w = pred_u_w_det.float().softmax(dim=1).max(dim=1)[0]
        mask_u_w = pred_u_w_det.argmax(dim=1)

        mask_u_w_c1 = mask_u_w.clone()
        conf_u_w_c1 = conf_u_w.clone()
        mask_u_w_c2 = mask_u_w.clone()
        conf_u_w_c2 = conf_u_w.clone()

        mask_u_w_c1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
        conf_u_w_c1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
        mask_u_w_c2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
        conf_u_w_c2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

        if use_hard_view_mask:
            pred_x = apply_view_mask_logits(pred_x, view_x, allowed_seg_mat)
            pred_u_w_fp = apply_view_mask_logits(pred_u_w_fp, view_u, allowed_seg_mat)
            pred_u_s1_mix = apply_view_mask_logits(pred_u_s1_mix, view_u_mix, allowed_seg_mat)
            pred_u_s2_mix = apply_view_mask_logits(pred_u_s2_mix, view_u_mix, allowed_seg_mat)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            loss_x_seg = (criterion_seg_ce(pred_x, mask_x) +
                          criterion_seg_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_x_cls = masked_bce_with_logits(
                pred_x_class, class_label_x.float(), mask_x_allowed, pos_weight=pos_weight
            )
            loss_x_fp_cls = masked_bce_with_logits(
                pred_x_class_fp, class_label_x.float(), mask_x_allowed, pos_weight=pos_weight
            )

            loss_u_s1_seg = criterion_seg_dice(
                pred_u_s1.softmax(dim=1),
                mask_u_w_c1.unsqueeze(1).float(),
                ignore=(conf_u_w_c1 < args.conf_thresh).float(),
            )
            loss_u_s2_seg = criterion_seg_dice(
                pred_u_s2.softmax(dim=1),
                mask_u_w_c2.unsqueeze(1).float(),
                ignore=(conf_u_w_c2 < args.conf_thresh).float(),
            )
            loss_u_w_fp_seg = criterion_seg_dice(
                pred_u_w_fp.softmax(dim=1),
                mask_u_w.unsqueeze(1).float(),
                ignore=(conf_u_w < args.conf_thresh).float(),
            )

            loss_u_s1_mix_seg = criterion_seg_dice(
                pred_u_s1_mix.softmax(dim=1),
                mask_u_w_mix.unsqueeze(1).float(),
                ignore=(conf_u_w_mix < args.conf_thresh).float(),
            )
            loss_u_s2_mix_seg = criterion_seg_dice(
                pred_u_s2_mix.softmax(dim=1),
                mask_u_w_mix.unsqueeze(1).float(),
                ignore=(conf_u_w_mix < args.conf_thresh).float(),
            )

            loss_u_s1_mix_cls = masked_bce_with_logits(
                pred_class_u_s1_mix, p_t, mask_u_cls, pos_weight=pos_weight
            )
            loss_u_s2_mix_cls = masked_bce_with_logits(
                pred_class_u_s2_mix, p_t, mask_u_cls, pos_weight=pos_weight
            )
            loss_u_pseudo_cls = 0.5 * (loss_u_s1_mix_cls + loss_u_s2_mix_cls)

            p_u_w_mix = torch.sigmoid(pred_class_u_w_mix)
            p_u_w_mix_fp = torch.sigmoid(pred_class_u_w_mix_fp)
            loss_u_w_mix_cls = masked_mse(p_u_w_mix, p_t, mask_u_cls)
            loss_u_w_mix_fp_cls = masked_mse(p_u_w_mix_fp, p_t, mask_u_cls)

            loss = (
                loss_w["x_seg"] * loss_x_seg
                + loss_w["x_cls"] * loss_x_cls
                + loss_w["x_fp_cls"] * loss_x_fp_cls
                + loss_w["u_s1_seg"] * loss_u_s1_seg
                + loss_w["u_s2_seg"] * loss_u_s2_seg
                + loss_w["u_w_fp_seg"] * loss_u_w_fp_seg
                + loss_w["u_s1_mix_seg"] * loss_u_s1_mix_seg
                + loss_w["u_s2_mix_seg"] * loss_u_s2_mix_seg
                + loss_w["u_pseudo_cls"] * loss_u_pseudo_cls
                + loss_w["u_w_mix_cls"] * loss_u_w_mix_cls
                + loss_w["u_w_mix_fp_cls"] * loss_u_w_mix_fp_cls
            )

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        global_step += 1

        seg_mask_ratio = (conf_u_w >= args.conf_thresh).float().mean().item()
        cls_mask_ratio = (mask_u_cls > 0).float().mean().item()

        update_meters(
            meters,
            {
                "loss": float(loss.item()),
                "x_seg": float(loss_x_seg.item()),
                "x_cls": float(loss_x_cls.item()),
                "x_fp_cls": float(loss_x_fp_cls.item()),
                "u_s1_seg": float(loss_u_s1_seg.item()),
                "u_s2_seg": float(loss_u_s2_seg.item()),
                "u_w_fp_seg": float(loss_u_w_fp_seg.item()),
                "u_s1_mix_seg": float(loss_u_s1_mix_seg.item()),
                "u_s2_mix_seg": float(loss_u_s2_mix_seg.item()),
                "u_pseudo_cls": float(loss_u_pseudo_cls.item()),
                "u_w_mix_cls": float(loss_u_w_mix_cls.item()),
                "u_w_mix_fp_cls": float(loss_u_w_mix_fp_cls.item()),
                "seg_mask_ratio": seg_mask_ratio,
                "cls_mask_ratio": cls_mask_ratio,
            },
        )

        if global_step % args.tb_iter_freq == 0:
            keys = [
                "loss",
                "seg_mask_ratio",
                "cls_mask_ratio",
                "x_seg",
                "x_cls",
                "x_fp_cls",
                "u_s1_seg",
                "u_s2_seg",
                "u_w_fp_seg",
                "u_s1_mix_seg",
                "u_s2_mix_seg",
                "u_pseudo_cls",
                "u_w_mix_cls",
                "u_w_mix_fp_cls",
            ]
            for gi, g in enumerate(optimizer.param_groups):
                writer.add_scalar(f"train/lr_group{gi}", g["lr"], global_step)

            log_train_tb(writer, meters, global_step, keys, prefix="train")

        iters = epoch * iters_per_epoch + i
        step_poly_lr(optimizer, base_lrs, iters, total_iters, power=0.9)


        if (len(train_loader_u) >= 8) and (i % max(1, (len(train_loader_u) // 8)) == 0):
            logger.info(
                "Iter {}/{} | L={:.3f} | "
                "X(seg={:.3f}, cls={:.3f}, fp_cls={:.3f}) | "
                "U(seg_s1={:.3f}, seg_s2={:.3f}, fp_seg={:.3f}, mix_s1={:.3f}, mix_s2={:.3f}, "
                "pseudo_cls={:.3f}, w_mix_cls={:.3f}, w_mix_fp_cls={:.3f}) | "
                "mask(seg={:.3f}, cls={:.3f})".format(
                    i,
                    len(train_loader_u),
                    meters["loss"].avg,
                    meters["x_seg"].avg,
                    meters["x_cls"].avg,
                    meters["x_fp_cls"].avg,
                    meters["u_s1_seg"].avg,
                    meters["u_s2_seg"].avg,
                    meters["u_w_fp_seg"].avg,
                    meters["u_s1_mix_seg"].avg,
                    meters["u_s2_mix_seg"].avg,
                    meters["u_pseudo_cls"].avg,
                    meters["u_w_mix_cls"].avg,
                    meters["u_w_mix_fp_cls"].avg,
                    meters["seg_mask_ratio"].avg,
                    meters["cls_mask_ratio"].avg,
                )
            )

    return global_step

@torch.no_grad()
def validate(args, model, device, valid_loader, allowed_seg_mat, cls_allowed):
    model.eval()

    C = args.seg_num_classes
    K = args.cls_num_classes
    tol = 2.0
    use_hard_view_mask = not args.no_hard_view_mask

    dice_sum = np.zeros(C - 1, dtype=np.float64)
    nsd_sum = np.zeros(C - 1, dtype=np.float64)
    cnt = np.zeros(C - 1, dtype=np.int64)

    y_true_all, y_prob_all, views_all = [], [], []

    for image, view, mask, class_label in valid_loader:
        image = image.to(device)
        gt_mask = mask.to(device)
        class_label = class_label.to(device)
        v = view.to(device).long().view(-1)

        h, w = image.shape[-2:]
        image_rs = F.interpolate(
            image, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )

        pred_mask_logits_rs, pred_class_out = model(image_rs)
        pred_mask_logits = F.interpolate(
            pred_mask_logits_rs, (h, w),
            mode="bilinear", align_corners=False
        )

        seg_logits = apply_view_mask_logits(pred_mask_logits, v, allowed_seg_mat) if use_hard_view_mask else pred_mask_logits
        pm = seg_logits.argmax(dim=1)

        pm = pm.cpu().numpy()[0].astype(np.int32)
        gt = gt_mask.cpu().numpy()[0].astype(np.int32)

        for cls in range(1, C):
            pred_bin = (pm == cls)
            gt_bin = (gt == cls)
            union = pred_bin.sum() + gt_bin.sum()
            if union == 0:
                continue
            inter = (pred_bin & gt_bin).sum()
            dice_sum[cls - 1] += (2.0 * inter) / (union + 1e-8)
            nsd_sum[cls - 1] += nsd_binary(pred_bin, gt_bin, tol=tol)
            cnt[cls - 1] += 1

        prob = torch.sigmoid(pred_class_out)
        y_true_all.append(class_label.cpu().numpy()[0])
        y_prob_all.append(prob.cpu().numpy()[0])
        views_all.append(v.cpu().numpy()[0])

    dice_class = 100.0 * dice_sum / np.maximum(cnt, 1)
    nsd_class = 100.0 * nsd_sum / np.maximum(cnt, 1)

    valid_mask = cnt > 0
    mean_dice = float(dice_class[valid_mask].mean()) if valid_mask.any() else 0.0
    mean_nsd = float(nsd_class[valid_mask].mean()) if valid_mask.any() else 0.0

    y_true_all = np.stack(y_true_all, axis=0) if len(y_true_all) else np.zeros((0, K), dtype=np.float32)
    y_prob_all = np.stack(y_prob_all, axis=0) if len(y_prob_all) else np.zeros((0, K), dtype=np.float32)
    views_all = np.array(views_all, dtype=np.int32) if len(views_all) else np.zeros((0,), dtype=np.int32)

    metrics = masked_metrics_with_threshold_search(y_true_all, y_prob_all, views_all, cls_allowed)
    macro_f1 = float(metrics["macro_f1@0.5"])
    score = (mean_dice + mean_nsd) / 2.0 + macro_f1 * 100.0

    return {
        "dice_class_view_masked": dice_class,
        "nsd_class_view_masked": nsd_class,
        "cnt_view_masked": cnt,
        "mean_dice_view_masked": mean_dice,
        "mean_nsd_view_masked": mean_nsd,
        "metrics": metrics,
        "macro_f1": macro_f1,
        "score": float(score),
    }



def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    seg_allowed = load_allowed_mapping(args.seg_allowed, DEFAULT_SEG_ALLOWED)
    cls_allowed = load_allowed_mapping(args.cls_allowed, DEFAULT_CLS_ALLOWED)
    loss_w = load_loss_weights(args.loss_weights, DEFAULT_LOSS_WEIGHTS)

    cudnn.enabled = True
    cudnn.benchmark = True

    logger = setup_logger(args.save_path)
    logger.info(str(args))
    logger.info(f"seg_allowed: {seg_allowed}")
    logger.info(f"cls_allowed: {cls_allowed}")
    logger.info(f"loss_weights: {loss_w}")
    logger.info(f"pseudo_tau_pos={args.pseudo_tau_pos}, pseudo_tau_neg={args.pseudo_tau_neg}")

    tb_logdir = args.tb_logdir or os.path.join(args.save_path, "tb")
    writer = SummaryWriter(log_dir=tb_logdir)
    logger.info(f"TensorBoard logdir: {tb_logdir}")

    model = build_model(args, device)
    optimizer, base_lrs = build_optimizer(args, model)

    
    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))

    allowed_seg_mat = build_seg_allowed_mat(device, seg_allowed, args.view_num_classes, args.seg_num_classes)
    allowed_cls_mat = build_allowed_mat(device, cls_allowed, num_views=args.view_num_classes, num_classes=args.cls_num_classes)

    db_train_u = FETUSSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = FETUSSemiDataset(
        args.train_labeled_json,
        "train_l",
        size=args.resize_target,
        n_sample=len(db_train_u.case_list),
    )
    db_valid = FETUSSemiDataset(args.valid_labeled_json, "valid")

    train_loader_l = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loader_u = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loader_u_mix = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    valid_loader = DataLoader(db_valid, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    pos_weight = compute_pos_weight_from_loader(train_loader_l, allowed_cls_mat, args.cls_num_classes, device).to(device)
    logger.info(f"pos_weight (masked by cls_allowed): {pos_weight.detach().cpu().numpy()}")

    total_iters = len(train_loader_u) * args.train_epochs
    ckpt_path = os.path.join(args.save_path, "latest.pth")
    start_epoch, best_score, best_epoch, global_step = maybe_resume(model, optimizer, scaler, ckpt_path, logger)

    for epoch in range(start_epoch + 1, args.train_epochs):
        logger.info(
            "===========> Epoch: {} | LR: {:.6f} | Best: {:.4f} @ epoch {}".format(
                epoch, optimizer.param_groups[0]["lr"], best_score, best_epoch
            )
        )

        global_step = train_one_epoch(
            args=args,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_loader_l=train_loader_l,
            train_loader_u=train_loader_u,
            train_loader_u_mix=train_loader_u_mix,
            allowed_seg_mat=allowed_seg_mat,
            allowed_cls_mat=allowed_cls_mat,
            pos_weight=pos_weight,
            loss_w=loss_w,
            writer=writer,
            logger=logger,
            epoch=epoch,
            base_lrs=base_lrs,
            global_step=global_step,
            total_iters=total_iters,
        )

        val = validate(args, model, device, valid_loader, allowed_seg_mat, cls_allowed)

        logger.info("===== Validation =====")
        for cls_idx in range(args.seg_num_classes - 1):
            logger.info(
                f"[Seg][ViewMasked] Class[{cls_idx+1}] "
                f"Dice={val['dice_class_view_masked'][cls_idx]:.2f}, "
                f"NSD={val['nsd_class_view_masked'][cls_idx]:.2f}, "
                f"N={val['cnt_view_masked'][cls_idx]}"
            )
        logger.info(f"[Seg][ViewMasked] MeanDice: {val['mean_dice_view_masked']:.2f}")
        logger.info(f"[Seg][ViewMasked] MeanNSD : {val['mean_nsd_view_masked']:.2f}")

        m = val["metrics"]
        logger.info(f"[Cls][Masked] Macro-F1@0.5  : {m['macro_f1@0.5']:.4f}")
        logger.info(f"[Cls][Masked] Macro-F1@best : {m['macro_f1@best']:.4f}")
        for k in range(args.cls_num_classes):
            logger.info(
                f"[Cls][Masked] Class[{k}] "
                f"F1@0.5={m['per_class_f1@0.5'][k]:.4f} | "
                f"F1@best={m['per_class_f1@best'][k]:.4f} | "
                f"best_thr={m['per_class_best_thr'][k]:.2f} | "
                f"AUPRC={m['per_class_auprc'][k]:.4f} | "
                f"support={m['support'][k]}"
            )

        score = val["score"]
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch

        logger.info(
            f"[Val] Epoch {epoch:03d} | score={score:.4f} | "
            f"best_score={best_score:.4f} @ epoch {best_epoch:03d} | is_best={is_best}"
        )

        log_val_tb(
            writer,
            {
                "MeanDice_ViewMasked": val["mean_dice_view_masked"],
                "MeanNSD_ViewMasked": val["mean_nsd_view_masked"],
                "MacroF1_best": val["macro_f1"],
                "Score": score,
            },
            epoch,
            prefix="val",
        )
        log_val_perclass_tb(writer, val["dice_class_view_masked"], val["nsd_class_view_masked"], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "previous_best": best_score,
            "best_epoch": best_epoch,
            "global_step": global_step,
        }
        torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))

    writer.close()


if __name__ == "__main__":
    main()
