# utils/vis_utils.py
# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize(writer, iter_num, image, label, pred, mode="train", prefix="long"):
    """原图 + 标签 + 预测 三图写入 TensorBoard"""
    image = image[0:1, ...]
    writer.add_image(f'{mode}/{prefix}_image', image, iter_num)
    writer.add_image(f'{mode}/{prefix}_gt', label.unsqueeze(0)*50, iter_num)
    writer.add_image(f'{mode}/{prefix}_pred', pred.unsqueeze(0)*50, iter_num)


def visualize_cls(writer, iter_num, preds, gts, mode="train"):
    """分类准确率实时记录"""
    cls_pred = torch.argmax(preds, dim=1)
    acc = (cls_pred == gts).float().mean()
    writer.add_scalar(f'{mode}/cls_acc', acc.item(), iter_num)



def visualize_case_row(long_image, trans_image,
                       gt_long, gt_trans,
                       pred_long, pred_trans,
                       cls_pred, cls_gt,
                       patient_name, save_dir):
    """
    一行展示长/短轴原图 + GT + Pred + 分类结果（7列）
    """

    os.makedirs(save_dir, exist_ok=True)

    # 转 numpy
    long_image = long_image.numpy().squeeze()
    trans_image = trans_image.numpy().squeeze()

    gt_long = gt_long.numpy().squeeze()
    gt_trans = gt_trans.numpy().squeeze()

    pred_long = pred_long.numpy().squeeze()
    pred_trans = pred_trans.numpy().squeeze()

    # 归一化灰度图
    long_image = (long_image - long_image.min()) / (long_image.max() - long_image.min() + 1e-8)
    trans_image = (trans_image - trans_image.min()) / (trans_image.max() - trans_image.min() + 1e-8)

    # 分类文字
    cls_label_map = {0: "RADS2", 1: "RADS3-4"}
    pred_text = f"pred: {cls_label_map.get(int(cls_pred), '?')}"
    gt_text = f"gt: {cls_label_map.get(int(cls_gt), '?')}"

    # 绘图：7列
    plt.figure(figsize=(28, 4))
    plt.suptitle(f"ID: {patient_name}", fontsize=15, y=1.05)

    # 1 长轴原图
    plt.subplot(1, 7, 1)
    plt.imshow(long_image, cmap="gray")
    plt.axis("off")
    plt.title("img_long")

    # 2 长轴GT
    plt.subplot(1, 7, 2)
    plt.imshow(long_image, cmap="gray")
    plt.imshow(gt_long, cmap="Blues", alpha=0.4)
    plt.axis("off")
    plt.title("long_GT")

    # 3 长轴Pred
    plt.subplot(1, 7, 3)
    plt.imshow(long_image, cmap="gray")
    plt.imshow(pred_long, cmap="Reds", alpha=0.4)
    plt.axis("off")
    plt.title("long_pred")

    # 4 短轴原图
    plt.subplot(1, 7, 4)
    plt.imshow(trans_image, cmap="gray")
    plt.axis("off")
    plt.title("img_trans")

    # 5 短轴GT
    plt.subplot(1, 7, 5)
    plt.imshow(trans_image, cmap="gray")
    plt.imshow(gt_trans, cmap="Blues", alpha=0.4)
    plt.axis("off")
    plt.title("trans_GT")

    # 6 短轴预测
    plt.subplot(1, 7, 6)
    plt.imshow(trans_image, cmap="gray")
    plt.imshow(pred_trans, cmap="Reds", alpha=0.4)
    plt.axis("off")
    plt.title("trans_pred")

    # 7 分类结果
    plt.subplot(1, 7, 7)
    plt.imshow(long_image, cmap="gray")
    plt.text(
        0.5, 0.5, f"{pred_text}\n{gt_text}",
        ha="center", va="center", fontsize=16, weight='bold',
        color='green' if int(cls_pred) == int(cls_gt) else 'red'
    )
    plt.axis("off")
    plt.title("cls")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patient_name}_summary.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


#  新增：分类混淆矩阵可视化
def plot_confusion_matrix(cm, class_names=["Class0", "Class1"], save_path=None, writer=None, iter_num=None, tag="val"):
    """绘制混淆矩阵"""
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if writer:
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        import PIL.Image
        image = PIL.Image.open(buf)
        writer.add_image(f"{tag}/confusion_matrix", np.array(image).transpose(2, 0, 1), iter_num)
    plt.close()
