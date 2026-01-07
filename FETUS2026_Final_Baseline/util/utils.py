import numpy as np
from torch import nn
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import torch.nn.functional as F

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1)
    return mask ^ eroded


def nsd_binary(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # both empty → perfect
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    # one empty → worst
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    s_pred = _surface(pred)
    s_gt = _surface(gt)

    dt_gt = distance_transform_edt(~s_gt)
    dt_pred = distance_transform_edt(~s_pred)

    d_pred_to_gt = dt_gt[s_pred]
    d_gt_to_pred = dt_pred[s_gt]

    p = (d_pred_to_gt <= tol).mean() if d_pred_to_gt.size else 0.0
    g = (d_gt_to_pred <= tol).mean() if d_gt_to_pred.size else 0.0

    return (p + g) / 2.0


def macro_f1_score(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    y_pred = (y_prob >= thr).astype(np.int32)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return macro, per_class


def apply_view_mask_logits(logits, view_ids, allowed_mat, fill_value=None):
    """
    logits: (B,C,H,W)
    view_ids: (B,) long
    allowed_mat: (V,C) bool
    """
    invalid = ~allowed_mat[view_ids]              # (B,C)
    invalid = invalid.unsqueeze(-1).unsqueeze(-1) # (B,C,1,1)

    if fill_value is None:
        # dtype-safe minimal finite value (fp16 -> -65504)
        fill_value = torch.finfo(logits.dtype).min

    return logits.masked_fill(invalid, fill_value)




def apply_view_mask_logits_cutmixed(
    logits,                 # (B,C,H,W)
    view_base,              # (B,) long, base image view id (image_u_w)
    view_mix,               # (B,) long, mix image view id (image_u_w_mix)
    cutmix_box,             # (B,H,W) 0/1, where pixels come from mix image
    allowed_mat,            # (V,C) bool
    fill_value=-1e9
):
    """
    Per-pixel allowed classes:
      - pixels where cutmix_box==0: use allowed of view_base
      - pixels where cutmix_box==1: use allowed of view_mix
    """
    B, C, H, W = logits.shape
    # (B,C,1,1) -> broadcast
    allowed_base = allowed_mat[view_base].view(B, C, 1, 1)
    allowed_mix  = allowed_mat[view_mix].view(B, C, 1, 1)

    box = cutmix_box.view(B, 1, H, W).bool()  # (B,1,H,W)

    # per-pixel allowed: choose base or mix
    allowed = torch.where(box, allowed_mix, allowed_base)  # (B,C,H,W) via broadcast

    return logits.masked_fill(~allowed, fill_value)




def invalid_mass_loss(logits, view_ids, allowed_mat):
    """
    Penalize probability mass assigned to invalid classes:
    mean_{pixels} sum_{c invalid} p(c)
    """
    p = F.softmax(logits, dim=1)                     # (B,C,H,W)
    invalid = (~allowed_mat[view_ids]).float()       # (B,C)
    invalid = invalid.unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)
    return (p * invalid).mean()


def update_meters(meters: dict, values: dict):
    """values: {name: float}"""
    for k, v in values.items():
        if k in meters:
            meters[k].update(float(v))
        else:
            # If you want to be stricter, you can raise; here we choose auto-create for convenience
            meters[k] = AverageMeter()
            meters[k].update(float(v))


def log_train_tb(writer, meters: dict, step: int, keys: list, prefix: str = "train"):
    """Log specified keys' .val from meters to tensorboard"""
    for k in keys:
        writer.add_scalar(f"{prefix}/{k}", meters[k].val, step)


def log_val_tb(writer, metrics: dict, epoch: int, prefix: str = "val"):
    """metrics: {name: float}"""
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", float(v), epoch)


def log_val_perclass_tb(writer, dice_oracle, nsd_oracle, epoch: int, prefix: str = "val_class"):
    C1 = len(dice_oracle)
    for i in range(C1):
        cls = i + 1
        writer.add_scalar(f"{prefix}/{cls}_Dice_Oracle", float(dice_oracle[i]), epoch)
        writer.add_scalar(f"{prefix}/{cls}_NSD_Oracle", float(nsd_oracle[i]), epoch)


def build_same_view_perm(view_base: torch.Tensor, view_mix: torch.Tensor):
    B = view_base.size(0)
    device = view_base.device
    perm = torch.empty(B, dtype=torch.long, device=device)
    all_idx_mix = torch.arange(B, device=device)

    for v in view_base.unique():
        idx_b = (view_base == v).nonzero(as_tuple=True)[0]
        idx_m = (view_mix == v).nonzero(as_tuple=True)[0]
        nb, nm = idx_b.numel(), idx_m.numel()

        if nm == 0:
            perm[idx_b] = all_idx_mix[torch.randint(0, B, (nb,), device=device)]
            continue

        if nm >= nb:
            # Without replacement
            perm[idx_b] = idx_m[torch.randperm(nm, device=device)[:nb]]
        else:
            # Minimal repetition: full round copy + padding
            reps = nb // nm
            rem  = nb % nm
            tiled = idx_m.repeat(reps)
            if rem > 0:
                extra = idx_m[torch.randperm(nm, device=device)[:rem]]
                tiled = torch.cat([tiled, extra], dim=0)
            perm[idx_b] = tiled
    return perm

def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    """
    logits:  (B, K)
    targets: (B, K) 0/1 or soft [0,1]
    mask:    (B, K) {0,1} 1=valid/readable
    pos_weight: torch.Tensor(K,) or None
    """
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss

def masked_mse(p, q, mask):
    """
    p,q: (B,K) probabilities in [0,1]
    mask: (B,K)
    """
    loss = ((p - q) ** 2) * mask
    return loss.sum() / (mask.sum() + 1e-6)


def build_allowed_mat(device, cls_allowed: dict, num_views=4, num_classes=7):
    allowed = torch.zeros((num_views, num_classes), dtype=torch.float32, device=device)
    for v in range(num_views):
        allowed[v, cls_allowed.get(v, [])] = 1.0
    return allowed

def build_allowed_mask_np(views: np.ndarray, cls_allowed: dict, num_classes=7):
    """
    views: (N,) in {0,1,2,3}
    return mask: (N,K) in {0,1}
    """
    N = views.shape[0]
    K = num_classes
    mask = np.zeros((N, K), dtype=np.int32)
    for i, v in enumerate(views):
        for k in cls_allowed.get(int(v), []):
            mask[i, k] = 1
    return mask

def masked_macro_f1_score(y_true: np.ndarray, y_prob: np.ndarray, views: np.ndarray,
                          cls_allowed: dict, thr: float = 0.5):
    """
    y_true: (N,K) 0/1
    y_prob: (N,K) [0,1]
    views:  (N,) 0..3
    returns:
      macro_f1 (float)
      per_class_f1 (K,)
      per_class_support (#allowed samples per class) (K,)
    """
    y_pred = (y_prob >= thr).astype(np.int32)
    mask = build_allowed_mask_np(views, cls_allowed, num_classes=y_true.shape[1])  # (N,K)

    K = y_true.shape[1]
    per_class = np.full((K,), np.nan, dtype=np.float32)
    support = np.zeros((K,), dtype=np.int32)

    for k in range(K):
        idx = mask[:, k].astype(bool)
        support[k] = int(idx.sum())
        if support[k] == 0:
            continue
        per_class[k] = f1_score(y_true[idx, k], y_pred[idx, k], zero_division=0)

    macro = float(np.nanmean(per_class)) if np.isfinite(per_class).any() else 0.0
    return macro, per_class, support


def compute_pos_weight_from_loader(train_loader_l, allowed_cls_mat, num_classes, device):
    """
    Count only on "class slots allowed for this sample's view":
      pos_count[k] = sum(y==1 and allowed==1)
      neg_count[k] = sum(y==0 and allowed==1)
    Return pos_weight[k] = neg/pos (standard usage for torch BCEWithLogitsLoss)
    """
    pos = torch.zeros(num_classes, dtype=torch.float64)
    neg = torch.zeros(num_classes, dtype=torch.float64)

    for (image_x, image_view, mask_x, class_label_x) in train_loader_l:
        v = image_view.to(device).long().view(-1)  # (B,)
        y = class_label_x.to(device).float()       # (B,K)
        allowed = allowed_cls_mat[v]               # (B,K) float{0,1}

        pos += ((y == 1) * allowed).sum(dim=0).double().cpu()
        neg += ((y == 0) * allowed).sum(dim=0).double().cpu()

    # Prevent division by zero; very rare classes can be very large, recommend clipping
    eps = 1.0
    pos_weight = (neg / (pos + eps)).clamp(min=1.0, max=50.0)  # You can also adjust max up/down
    return pos_weight.to(torch.float32)

def masked_metrics_with_threshold_search(
    y_true: np.ndarray,       # (N,K) 0/1
    y_prob: np.ndarray,       # (N,K) [0,1]
    views: np.ndarray,        # (N,) 0..V-1
    cls_allowed: dict,        # CLS_ALLOWED
    thr_grid: np.ndarray = None,
):
    """
    Only evaluate at positions where allowed(view,k)=1.
    Outputs:
      - macro_f1@0.5
      - macro_f1@best (find threshold separately for each class)
      - per_class_f1@0.5, per_class_f1@best
      - per_class_best_thr
      - per_class_AUPRC, per_class_AUROC
      - support (number of allowed samples)
    """
    N, K = y_true.shape
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)

    # build allowed mask (N,K)
    allowed = np.zeros((N, K), dtype=np.int32)
    for v, ks in cls_allowed.items():
        rows = np.where(views == v)[0]                 # Row indices of these samples
        cols = np.array(ks, dtype=np.int64)           # Column indices of these classes
        if rows.size > 0 and cols.size > 0:
            allowed[np.ix_(rows, cols)] = 1

    # -------- fixed thr=0.5 --------
    y_pred_05 = (y_prob >= 0.5).astype(np.int32)
    f1_05 = np.zeros(K, dtype=np.float32)
    support = np.zeros(K, dtype=np.int32)

    for k in range(K):
        m = allowed[:, k] == 1
        support[k] = int(m.sum())
        if support[k] == 0:
            f1_05[k] = 0.0
            continue
        f1_05[k] = f1_score(y_true[m, k], y_pred_05[m, k], zero_division=0)

    macro_f1_05 = float(f1_05.mean())

    # -------- per-class best threshold --------
    best_thr = np.full(K, 0.5, dtype=np.float32)
    f1_best = np.zeros(K, dtype=np.float32)

    for k in range(K):
        m = allowed[:, k] == 1
        if m.sum() == 0:
            continue
        yt = y_true[m, k]
        yp = y_prob[m, k]

        best = -1.0
        best_t = 0.5
        for t in thr_grid:
            yhat = (yp >= t).astype(np.int32)
            f1t = f1_score(yt, yhat, zero_division=0)
            if f1t > best:
                best = f1t
                best_t = t
        f1_best[k] = best if best >= 0 else 0.0
        best_thr[k] = best_t

    macro_f1_best = float(f1_best.mean())

    # -------- threshold-free metrics --------
    auprc = np.zeros(K, dtype=np.float32)
    auroc = np.zeros(K, dtype=np.float32)
    for k in range(K):
        m = allowed[:, k] == 1
        if m.sum() == 0:
            continue
        yt = y_true[m, k]
        yp = y_prob[m, k]
        # AUPRC: If all same class (all 0 or all 1), will raise error, add protection
        if yt.max() == yt.min():
            auprc[k] = 0.0
            auroc[k] = 0.0
        else:
            auprc[k] = float(average_precision_score(yt, yp))
            auroc[k] = float(roc_auc_score(yt, yp))

    return {
        "macro_f1@0.5": macro_f1_05,
        "macro_f1@best": macro_f1_best,
        "per_class_f1@0.5": f1_05,
        "per_class_f1@best": f1_best,
        "per_class_best_thr": best_thr,
        "per_class_auprc": auprc,
        "per_class_auroc": auroc,
        "support": support,
    }
    
def load_pretrained_flexible(model, ckpt_path, logger=None, key="model"):
    """
    Allowed:
      - In checkpoint but not in current model => ignore
      - In current model but not in checkpoint => keep random initialization
      - Same name but different shape => ignore (common when classification/segmentation heads change class numbers)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Compatible: ckpt is state_dict itself or dict contains "model"
    if isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
    else:
        state = ckpt

    model_state = model.state_dict()

    filtered = {}
    skipped_not_in_model = []
    skipped_shape = []

    for k, v in state.items():
        if k not in model_state:
            skipped_not_in_model.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    msg = f"[Pretrain] Load from {ckpt_path}: use {len(filtered)}/{len(state)} params"
    if logger: logger.info(msg)
    else: print(msg)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # missing are keys "in model but not loaded" (including new layers or filtered by shape)
    # unexpected should theoretically be empty, since we filter k not in model
    if logger:
        if skipped_not_in_model:
            logger.info(f"[Pretrain] Skip (not in current model): {len(skipped_not_in_model)} keys")
        if skipped_shape:
            logger.info(f"[Pretrain] Skip (shape mismatch): {len(skipped_shape)} keys")
            # Only print first few to avoid screen flooding
            for k, s_ckpt, s_now in skipped_shape[:20]:
                logger.info(f"  - {k}: ckpt{ s_ckpt } -> now{ s_now }")
        if missing:
            logger.info(f"[Pretrain] Missing in ckpt (kept init): {len(missing)} keys")
        if unexpected:
            logger.info(f"[Pretrain] Unexpected after filtering (rare): {len(unexpected)} keys")

    return ckpt  # Convenient for you to decide whether to load optimizer etc. later
