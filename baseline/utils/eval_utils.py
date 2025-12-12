# -*- coding: utf-8 -*-
# utils/eval_utils.py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from medpy import metric


def calculate_metric_percase(pred, gt, spacing=(1.0, 1.0), tolerance=2.0):
    """
    计算 Dice 系数和 NSD (Normalized Surface Distance)
    
    Args:
        pred: 预测的二值mask (numpy array)
        gt: 真实标注的二值mask (numpy array)
        spacing: 像素间距 (mm), 默认 (1.0, 1.0) 表示每个像素1mm
        tolerance: NSD的容差阈值 (mm), 默认2.0mm
    
    Returns:
        dice: Dice系数 (0-1)
        nsd: Normalized Surface Distance (0-1), 值越大越好
    """
    # 确保是二值mask
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    # 如果预测或真值为空，返回0
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0  # 都为空时认为完全匹配
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0  # 一个为空时无法匹配
    
    # ========== 计算 Dice ==========
    dice = metric.binary.dc(pred, gt)
    
    # ========== 计算 NSD ==========
    # 1. 提取表面点 (边界)
    # 表面 = 原mask - 腐蚀后的mask
    pred_surface = pred.astype(bool) ^ binary_erosion(pred).astype(bool)
    gt_surface = gt.astype(bool) ^ binary_erosion(gt).astype(bool)
    
    # 如果没有表面点（极小的区域），返回dice作为参考
    if not pred_surface.any() or not gt_surface.any():
        return dice, dice
    
    # 2. 计算距离变换
    # 对于预测表面的每个点，计算到GT表面的最近距离
    gt_distance_map = distance_transform_edt(~gt_surface, sampling=spacing)
    pred_distance_map = distance_transform_edt(~pred_surface, sampling=spacing)
    
    # 3. 获取表面点的距离值
    pred_surface_distances = gt_distance_map[pred_surface]
    gt_surface_distances = pred_distance_map[gt_surface]
    
    # 4. 计算在容差范围内的点的比例
    pred_within_tolerance = (pred_surface_distances <= tolerance).sum()
    gt_within_tolerance = (gt_surface_distances <= tolerance).sum()
    
    total_surface_points = len(pred_surface_distances) + len(gt_surface_distances)
    
    # NSD = 在容差范围内的表面点数 / 总表面点数
    nsd = (pred_within_tolerance + gt_within_tolerance) / total_surface_points
    
    return float(dice), float(nsd)


def calculate_hd95(pred, gt):
    """
    单独计算 HD95 (Hausdorff Distance 95th percentile)
    
    Args:
        pred: 预测的二值mask
        gt: 真实标注的二值mask
    
    Returns:
        hd95: 95th percentile Hausdorff Distance (mm)
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    if pred.sum() == 0 or gt.sum() == 0:
        return 100.0  # 返回一个较大的值表示失败
    
    try:
        hd95 = metric.binary.hd95(pred, gt)
        return float(hd95)
    except:
        return 100.0


def evaluate_metrics(seg_pred, seg_gt, cls_pred, cls_gt, classes):
    """
    计算分割(Dice, NSD, vessel/plaque单类Dice)与分类指标(Acc, Precision, Recall, F1, 混淆矩阵)
    
    Args:
        seg_pred: 分割预测 (torch.Tensor)
        seg_gt: 分割真值 (torch.Tensor)
        cls_pred: 分类预测logits (torch.Tensor)
        cls_gt: 分类真值 (torch.Tensor)
        classes: 分割类别数（包括背景）
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    seg_pred_np = seg_pred.cpu().numpy()
    seg_gt_np = seg_gt.cpu().numpy()

    dice_list, nsd_list = [], []
    per_class_dice = {}
    per_class_nsd = {}

    # ---------- 分割指标 ----------
    for i in range(1, classes):  # 忽略背景 0
        dice, nsd = calculate_metric_percase(seg_pred_np == i, seg_gt_np == i)
        dice_list.append(dice)
        nsd_list.append(nsd)
        
        # 每类指标单独保存
        if i == 1:
            per_class_dice["vessel_dice"] = dice
            per_class_nsd["vessel_nsd"] = nsd
        elif i == 2:
            per_class_dice["plaque_dice"] = dice
            per_class_nsd["plaque_nsd"] = nsd

    mean_dice = np.mean(dice_list)
    mean_nsd = np.mean(nsd_list)

    # ---------- 分类指标 ----------
    cls_pred_np = torch.argmax(cls_pred, dim=1).cpu().numpy()
    cls_gt_np = cls_gt.cpu().numpy()

    acc = accuracy_score(cls_gt_np, cls_pred_np)
    precision = precision_score(cls_gt_np, cls_pred_np, average='binary', zero_division=0)
    recall = recall_score(cls_gt_np, cls_pred_np, average='binary', zero_division=0)
    f1 = f1_score(cls_gt_np, cls_pred_np, average='binary', zero_division=0)
    cm = confusion_matrix(cls_gt_np, cls_pred_np, labels=[0, 1])

    return {
        "mean_dice": mean_dice,
        "mean_nsd": mean_nsd,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        **per_class_dice,
        **per_class_nsd
    }


# ========== 如果你需要batch处理的版本 ==========
def batch_calculate_metrics(pred_batch, gt_batch, spacing=(1.0, 1.0)):
    """
    批量计算Dice和NSD
    
    Args:
        pred_batch: (B, H, W) 预测mask
        gt_batch: (B, H, W) 真值mask
        spacing: 像素间距
    
    Returns:
        dice_scores: list of Dice scores
        nsd_scores: list of NSD scores
    """
    dice_scores = []
    nsd_scores = []
    
    for pred, gt in zip(pred_batch, gt_batch):
        dice, nsd = calculate_metric_percase(pred, gt, spacing)
        dice_scores.append(dice)
        nsd_scores.append(nsd)
    
    return dice_scores, nsd_scores