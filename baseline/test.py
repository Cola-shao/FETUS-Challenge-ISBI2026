import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Dataset import BaseDataSets, ValGenerator
from unet import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.eval_utils import calculate_metric_percase  
from utils.vis_utils import visualize_case_row, plot_confusion_matrix


# =========================================================
# 参数配置
# =========================================================
class Args:
    root_path = "/mnt/data1/mxl/home/SSL4MIS-master_h5/label_train"
    model_path = "/mnt/data1/mxl/home/SSL4MIS-master_h5/Carotid_MeanTeacher_FIXED_20_24_8/best_dice_26500.pth"
    batch_size = 1  
    num_classes = 3
    num_cls = 2
    patch_size = [256, 256]
    save_vis = "/mnt/data1/mxl/home/SSL4MIS-master_max1/Carotid_MeanTeacher_FIXED_20_24_8"
    
    spacing = (1.0, 1.0) 
    nsd_tolerance = 2.0
    
    # 可视化设置
    num_vis_cases = 5  # 可视化的病例数

args = Args()
os.makedirs(args.save_vis, exist_ok=True)


def test(args):
    """测试函数"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # =========================================================
    # 模型加载
    # =========================================================
    model = UNet(in_chns=1, seg_classes=args.num_classes, cls_classes=args.num_cls)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {args.model_path}")

    # =========================================================
    # 数据加载
    # =========================================================
    test_dataset = BaseDataSets(
        base_dir=args.root_path, 
        split="test", 
        transform=ValGenerator(args.patch_size)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    print(f"[INFO] Total test samples: {len(test_dataset)}")

    # =========================================================
    # 指标累积容器
    # =========================================================
    # 分类指标
    all_cls_preds, all_cls_gts, all_cls_probs = [], [], []
    
    # 分割指标 - Long axis
    vessel_dices_long, plaque_dices_long = [], []
    vessel_nsd_long, plaque_nsd_long = [], []
    
    # 分割指标 - Trans axis
    vessel_dices_trans, plaque_dices_trans = [], []
    vessel_nsd_trans, plaque_nsd_trans = [], []

    # =========================================================
    # 测试循环
    # =========================================================
    print("[INFO] Evaluating...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, ncols=80, desc="Testing")):
            img_long = batch["image_long"].to(device)
            img_trans = batch["image_trans"].to(device)
            label_long = batch["label_long"]
            label_trans = batch["label_trans"]
            cls_label = batch["cls_label"]

            # 前向传播
            seg_long, seg_trans, cls_out = model(img_long, img_trans)

            # ======== 分类预测 ========
            cls_prob = F.softmax(cls_out, dim=1)
            cls_pred = torch.argmax(cls_prob, dim=1)
            
            all_cls_preds.extend(cls_pred.cpu().numpy())
            all_cls_gts.extend(cls_label.numpy())
            all_cls_probs.extend(cls_prob[:, 1].cpu().numpy())  # 保存正类概率用于AUC

            # ======== 长轴分割预测 ========
            seg_pred_long = torch.argmax(F.softmax(seg_long, dim=1), dim=1)
            seg_pred_long_np = seg_pred_long.cpu().numpy()
            seg_gt_long_np = label_long.numpy()
            
            # 处理维度（确保是2D）
            if seg_pred_long_np.ndim == 3:
                seg_pred_long_np = seg_pred_long_np.squeeze(0)
            if seg_gt_long_np.ndim == 3:
                seg_gt_long_np = seg_gt_long_np.squeeze(0)

            # 血管 (class 1)
            if np.any(seg_gt_long_np == 1):  # 只有GT中有血管才计算
                dice_vessel, nsd_vessel = calculate_metric_percase(
                    seg_pred_long_np == 1, 
                    seg_gt_long_np == 1,
                    spacing=args.spacing,
                    tolerance=args.nsd_tolerance
                )
                vessel_dices_long.append(dice_vessel)
                vessel_nsd_long.append(nsd_vessel)
            
            # 斑块 (class 2)
            if np.any(seg_gt_long_np == 2):  # 只有GT中有斑块才计算
                dice_plaque, nsd_plaque = calculate_metric_percase(
                    seg_pred_long_np == 2, 
                    seg_gt_long_np == 2,
                    spacing=args.spacing,
                    tolerance=args.nsd_tolerance
                )
                plaque_dices_long.append(dice_plaque)
                plaque_nsd_long.append(nsd_plaque)
                
            # ======== 横轴分割预测 ========
            seg_pred_trans = torch.argmax(F.softmax(seg_trans, dim=1), dim=1)
            seg_pred_trans_np = seg_pred_trans.cpu().numpy()
            seg_gt_trans_np = label_trans.numpy()
            
            # 处理维度
            if seg_pred_trans_np.ndim == 3:
                seg_pred_trans_np = seg_pred_trans_np.squeeze(0)
            if seg_gt_trans_np.ndim == 3:
                seg_gt_trans_np = seg_gt_trans_np.squeeze(0)

            # 血管 (class 1)
            if np.any(seg_gt_trans_np == 1):
                dice_vessel, nsd_vessel = calculate_metric_percase(
                    seg_pred_trans_np == 1, 
                    seg_gt_trans_np == 1,
                    spacing=args.spacing,
                    tolerance=args.nsd_tolerance
                )
                vessel_dices_trans.append(dice_vessel)
                vessel_nsd_trans.append(nsd_vessel)
            
            # 斑块 (class 2)
            if np.any(seg_gt_trans_np == 2):
                dice_plaque, nsd_plaque = calculate_metric_percase(
                    seg_pred_trans_np == 2, 
                    seg_gt_trans_np == 2,
                    spacing=args.spacing,
                    tolerance=args.nsd_tolerance
                )
                plaque_dices_trans.append(dice_plaque)
                plaque_nsd_trans.append(nsd_plaque)

    # =========================================================
    # 分类指标计算
    # =========================================================
    cls_preds = np.array(all_cls_preds)
    cls_gts = np.array(all_cls_gts)
    cls_probs = np.array(all_cls_probs)

    acc = accuracy_score(cls_gts, cls_preds)
    precision = precision_score(cls_gts, cls_preds, zero_division=0)
    recall = recall_score(cls_gts, cls_preds, zero_division=0)
    f1 = f1_score(cls_gts, cls_preds, zero_division=0)
    
    # 计算AUC
    try:
        auc = roc_auc_score(cls_gts, cls_probs)
    except ValueError:
        auc = 0.0
        print("[WARNING] Cannot compute AUC (only one class in test set)")
    
    cm = confusion_matrix(cls_gts, cls_preds, labels=[0, 1])
    
    # 计算每个类别的准确率
    class0_mask = cls_gts == 0
    class1_mask = cls_gts == 1
    class0_acc = np.mean(cls_preds[class0_mask] == 0) if class0_mask.sum() > 0 else 0.0
    class1_acc = np.mean(cls_preds[class1_mask] == 1) if class1_mask.sum() > 0 else 0.0

    # =========================================================
    # 分割指标计算
    # =========================================================
    # 处理可能为空的列表
    mean_dice_vessel_long = np.mean(vessel_dices_long) if len(vessel_dices_long) > 0 else 0.0
    mean_dice_plaque_long = np.mean(plaque_dices_long) if len(plaque_dices_long) > 0 else 0.0
    mean_dice_long = (mean_dice_vessel_long + mean_dice_plaque_long) / 2

    mean_dice_vessel_trans = np.mean(vessel_dices_trans) if len(vessel_dices_trans) > 0 else 0.0
    mean_dice_plaque_trans = np.mean(plaque_dices_trans) if len(plaque_dices_trans) > 0 else 0.0
    mean_dice_trans = (mean_dice_vessel_trans + mean_dice_plaque_trans) / 2

    mean_dice_vessel = (mean_dice_vessel_long + mean_dice_vessel_trans) / 2
    mean_dice_plaque = (mean_dice_plaque_long + mean_dice_plaque_trans) / 2
    mean_dice = (mean_dice_long + mean_dice_trans) / 2

    # NSD 指标
    mean_nsd_vessel_long = np.mean(vessel_nsd_long) if len(vessel_nsd_long) > 0 else 0.0
    mean_nsd_plaque_long = np.mean(plaque_nsd_long) if len(plaque_nsd_long) > 0 else 0.0
    mean_nsd_long = (mean_nsd_vessel_long + mean_nsd_plaque_long) / 2

    mean_nsd_vessel_trans = np.mean(vessel_nsd_trans) if len(vessel_nsd_trans) > 0 else 0.0
    mean_nsd_plaque_trans = np.mean(plaque_nsd_trans) if len(plaque_nsd_trans) > 0 else 0.0
    mean_nsd_trans = (mean_nsd_vessel_trans + mean_nsd_plaque_trans) / 2

    mean_nsd_vessel = (mean_nsd_vessel_long + mean_nsd_vessel_trans) / 2
    mean_nsd_plaque = (mean_nsd_plaque_long + mean_nsd_plaque_trans) / 2
    mean_nsd = (mean_nsd_long + mean_nsd_trans) / 2

    # =========================================================
    # 输出结果
    # =========================================================
    print("\n" + "="*50)
    print("Classification Results".center(50))
    print("="*50)
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"AUC:        {auc:.4f}")
    print(f"Class0 Acc: {class0_acc:.4f} | Class1 Acc: {class1_acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\n" + "="*50)
    print("Segmentation Results - Dice Coefficient".center(50))
    print("="*50)
    print(f"Long-axis:  Vessel={mean_dice_vessel_long:.4f}, Plaque={mean_dice_plaque_long:.4f}, Mean={mean_dice_long:.4f}")
    print(f"Trans-axis: Vessel={mean_dice_vessel_trans:.4f}, Plaque={mean_dice_plaque_trans:.4f}, Mean={mean_dice_trans:.4f}")
    print(f"Combined:   Vessel={mean_dice_vessel:.4f}, Plaque={mean_dice_plaque:.4f}, Mean={mean_dice:.4f}")

    print("\n" + "="*50)
    print("Segmentation Results - NSD".center(50))
    print("="*50)
    print(f"Long-axis:  Vessel={mean_nsd_vessel_long:.4f}, Plaque={mean_nsd_plaque_long:.4f}, Mean={mean_nsd_long:.4f}")
    print(f"Trans-axis: Vessel={mean_nsd_vessel_trans:.4f}, Plaque={mean_nsd_plaque_trans:.4f}, Mean={mean_nsd_trans:.4f}")
    print(f"Combined:   Vessel={mean_nsd_vessel:.4f}, Plaque={mean_nsd_plaque:.4f}, Mean={mean_nsd:.4f}")

    # =========================================================
    # 保存结果到文件
    # =========================================================
    results_file = os.path.join(args.save_vis, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("="*50 + "\n")
        f.write("Classification Results\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:   {acc:.4f}\n")
        f.write(f"Precision:  {precision:.4f}\n")
        f.write(f"Recall:     {recall:.4f}\n")
        f.write(f"F1-score:   {f1:.4f}\n")
        f.write(f"AUC:        {auc:.4f}\n")
        f.write(f"Class0 Acc: {class0_acc:.4f} | Class1 Acc: {class1_acc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Segmentation Results - Dice\n")
        f.write("="*50 + "\n")
        f.write(f"Long-axis:  Vessel={mean_dice_vessel_long:.4f}, Plaque={mean_dice_plaque_long:.4f}, Mean={mean_dice_long:.4f}\n")
        f.write(f"Trans-axis: Vessel={mean_dice_vessel_trans:.4f}, Plaque={mean_dice_plaque_trans:.4f}, Mean={mean_dice_trans:.4f}\n")
        f.write(f"Combined:   Vessel={mean_dice_vessel:.4f}, Plaque={mean_dice_plaque:.4f}, Mean={mean_dice:.4f}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Segmentation Results - NSD\n")
        f.write("="*50 + "\n")
        f.write(f"Long-axis:  Vessel={mean_nsd_vessel_long:.4f}, Plaque={mean_nsd_plaque_long:.4f}, Mean={mean_nsd_long:.4f}\n")
        f.write(f"Trans-axis: Vessel={mean_nsd_vessel_trans:.4f}, Plaque={mean_nsd_plaque_trans:.4f}, Mean={mean_nsd_trans:.4f}\n")
        f.write(f"Combined:   Vessel={mean_nsd_vessel:.4f}, Plaque={mean_nsd_plaque:.4f}, Mean={mean_nsd:.4f}\n")
    
    print(f"\n[INFO] Results saved to {results_file}")

    # =========================================================
    # 绘制混淆矩阵
    # =========================================================
    plot_confusion_matrix(
        cm, 
        class_names=["RADS2(0)", "RADS3-4(1)"],
        save_path=os.path.join(args.save_vis, "confusion_matrix.png")
    )
    print(f"[INFO] Confusion matrix saved")

    # =========================================================
    # 随机可视化病例
    # =========================================================
    num_cases_to_vis = min(args.num_vis_cases, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_cases_to_vis)
    print(f"\n[INFO] Visualizing {num_cases_to_vis} random cases: {sample_indices}")

    for i in sample_indices:
        sample = test_dataset[i]
        patient_name = f"case_{i}"

        img_long = sample["image_long"].unsqueeze(0).to(device)
        img_trans = sample["image_trans"].unsqueeze(0).to(device)
        
        label_long = sample["label_long"]
        if isinstance(label_long, torch.Tensor):
            gt_long = label_long
        else:
            gt_long = torch.from_numpy(label_long)
        
        label_trans = sample["label_trans"]
        if isinstance(label_trans, torch.Tensor):
            gt_trans = label_trans
        else:
            gt_trans = torch.from_numpy(label_trans)

        cls_gt = sample["cls_label"]

        with torch.no_grad():
            seg_pred_long, seg_pred_trans, cls_out = model(img_long, img_trans)
            pred_long = torch.argmax(seg_pred_long, dim=1)[0].cpu()
            pred_trans = torch.argmax(seg_pred_trans, dim=1)[0].cpu()
            cls_pred = torch.argmax(cls_out, dim=1)[0].item()

        visualize_case_row(
            long_image=img_long[0].cpu(),
            trans_image=img_trans[0].cpu(),
            gt_long=gt_long,
            gt_trans=gt_trans,
            pred_long=pred_long,
            pred_trans=pred_trans,
            cls_pred=cls_pred,
            cls_gt=cls_gt,
            patient_name=patient_name,
            save_dir=args.save_vis
        )

    print(f"[INFO] Visualizations saved in: {args.save_vis}")
    print("\n Evaluation finished!")


if __name__ == "__main__":
    test(args)