import os
import torch
import random
import numpy as np
import h5py
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, Sampler
from scipy import ndimage
import itertools


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split="train", transform=None, num_labeled=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        
        self.sample_list = []
        self.num_labeled = 0  
        
        if self.split == "train":
            # 读取train列表
            train_list = os.path.join(base_dir, "train", "train.list")
            if os.path.exists(train_list):
                with open(train_list, "r") as f:
                    self.sample_list = [line.strip() for line in f.readlines()]
            
            # 使用num_labeled参数控制有标签数量
            if num_labeled is not None:
                self.num_labeled = num_labeled
            else:
                self.num_labeled = len(self.sample_list)
            
            print(f"[INFO] Train - Labeled: {self.num_labeled}, "
                  f"Unlabeled: {len(self.sample_list) - self.num_labeled}, "
                  f"Total: {len(self.sample_list)}")
            
        elif self.split == "val":
            val_list = os.path.join(base_dir, "val", "val.list")
            with open(val_list, "r") as f:
                self.sample_list = [line.strip() for line in f.readlines()]
            self.num_labeled = len(self.sample_list)
            print(f"[INFO] Val - Total: {len(self.sample_list)}")
        
        elif self.split == "test":
            test_list = os.path.join(base_dir, "test", "test.list")
            with open(test_list, "r") as f:
                self.sample_list = [line.strip() for line in f.readlines()]
            self.num_labeled = len(self.sample_list)
            print(f"[INFO] Test - Total: {len(self.sample_list)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        # 图像路径
        img_path = os.path.join(self._base_dir, self.split, "img", case + ".h5")
        # 标签路径
        mask_path = os.path.join(self._base_dir, self.split, "mask", case + "_label.h5")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        
        # 读取图像
        h5f = h5py.File(img_path, "r")
        image_long = h5f["long_img"][:]
        image_trans = h5f["trans_img"][:]
        h5f.close()
        
        # 读取标签
        if self.split == "train" and idx >= self.num_labeled:
            # unlabeled数据
            label_long = np.zeros_like(image_long, dtype=np.int64)
            label_trans = np.zeros_like(image_trans, dtype=np.int64)
            cls_label = -1
        else:
            # labeled数据
            if os.path.exists(mask_path):
                h5f = h5py.File(mask_path, "r")
                label_long = h5f["long_mask"][:]  
                label_trans = h5f["trans_mask"][:]  
                label_long = self.convert_labels(label_long)
                label_trans = self.convert_labels(label_trans)
                if "cls" in h5f:
                    cls_label = int(h5f["cls"][()])  
                else:
                    cls_label = -1
                h5f.close()
            else:
                label_long = np.zeros_like(image_long, dtype=np.int64)
                label_trans = np.zeros_like(image_trans, dtype=np.int64)
                cls_label = -1
        
        sample = {
            "image_long": image_long, 
            "image_trans": image_trans, 
            "label_long": label_long,
            "label_trans": label_trans,
            "cls_label": cls_label
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    @staticmethod
    def convert_labels(label):
        """
        将像素值转换为类别索引
        0 -> 0 (背景)
        255 -> 1 (血管)
        128 -> 2 (斑块)
        """
        label_converted = np.zeros_like(label, dtype=np.int64)
        label_converted[label == 255] = 1 
        label_converted[label == 128] = 2 
        return label_converted


# ===============================================
# 全局数据增强函数（按照官方写法）
# ===============================================
def random_rot_flip(image, label=None):
    """随机旋转和翻转"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    """随机旋转（小角度）"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# ===============================================
# Transform类
# ===============================================
class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        image_long = sample["image_long"] 
        image_trans = sample["image_trans"]
        label_long = sample["label_long"]
        label_trans = sample["label_trans"]
        
        # Resize long-axis
        h, w = image_long.shape
        scale_h = self.output_size[0] / h
        scale_w = self.output_size[1] / w
        image_long = zoom(image_long, (scale_h, scale_w), order=1)
        label_long = zoom(label_long, (scale_h, scale_w), order=0)
        
        # Resize trans-axis
        h, w = image_trans.shape
        scale_h = self.output_size[0] / h
        scale_w = self.output_size[1] / w
        image_trans = zoom(image_trans, (scale_h, scale_w), order=1)
        label_trans = zoom(label_trans, (scale_h, scale_w), order=0)
        
        # Convert to tensor
        image_long = torch.from_numpy(image_long.astype(np.float32)).unsqueeze(0) 
        image_trans = torch.from_numpy(image_trans.astype(np.float32)).unsqueeze(0)
        label_long = torch.from_numpy(label_long.astype(np.int64)) 
        label_trans = torch.from_numpy(label_trans.astype(np.int64))
        
        return {
            "image_long": image_long,
            "image_trans": image_trans,
            "label_long": label_long,
            "label_trans": label_trans,
            "cls_label": sample["cls_label"]
        }


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        image_long = sample["image_long"]
        image_trans = sample["image_trans"]
        label_long = sample["label_long"]
        label_trans = sample["label_trans"]
        
        # 数据增强：随机旋转翻转（调用全局函数）
        rand_value = random.random()
        if rand_value < 0.5:
            image_long, label_long = random_rot_flip(image_long, label_long)
            image_trans, label_trans = random_rot_flip(image_trans, label_trans)
        else:
            image_long, label_long = random_rotate(image_long, label_long)
            image_trans, label_trans = random_rotate(image_trans, label_trans)
        
        # Resize到目标尺寸
        h, w = image_long.shape
        image_long = zoom(image_long, (self.output_size[0]/h, self.output_size[1]/w), order=1)
        label_long = zoom(label_long, (self.output_size[0]/h, self.output_size[1]/w), order=0)
        
        h, w = image_trans.shape
        image_trans = zoom(image_trans, (self.output_size[0]/h, self.output_size[1]/w), order=1)
        label_trans = zoom(label_trans, (self.output_size[0]/h, self.output_size[1]/w), order=0)
        
        # Convert to tensor
        image_long = torch.from_numpy(image_long.astype(np.float32)).unsqueeze(0)
        image_trans = torch.from_numpy(image_trans.astype(np.float32)).unsqueeze(0)
        label_long = torch.from_numpy(label_long.astype(np.int64))
        label_trans = torch.from_numpy(label_trans.astype(np.int64))
        
        return {
            "image_long": image_long,
            "image_trans": image_trans,
            "label_long": label_long,
            "label_trans": label_trans,
            "cls_label": sample["cls_label"]
        }


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)