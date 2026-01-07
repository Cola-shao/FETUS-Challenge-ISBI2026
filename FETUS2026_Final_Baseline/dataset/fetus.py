from copy import deepcopy
import h5py
import math
import json
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage.color import rgb2gray
from torchvision import transforms
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box


class FETUSSemiDataset(Dataset):
    def __init__(self, json_file_path, mode, size=None, n_sample=None):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample

        if mode == 'train_l' or mode == 'train_u':
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)
            if mode == 'train_l' and n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]
        else:
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]
            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.from_numpy(image_view_id).long() - 1,
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]
            # basic augmentation
            if random.random() > 0.5:
                image, mask = random_rot_flip(image, mask)
            elif random.random() > 0.5:
                image, mask = random_rotate(image, mask)
            # resize
            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=0)
            mask = zoom(mask, (self.size / x, self.size / y), order=0)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.from_numpy(image_view_id).long() - 1,
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]
            # basic augmentation
            if random.random() > 0.5:
                image = random_rot_flip(image)
            elif random.random() > 0.5:
                image = random_rotate(image)
            # resize
            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=0)

            image = Image.fromarray((image * 255).astype(np.uint8))
            image_s1, image_s2 = deepcopy(image), deepcopy(image)
            image = torch.from_numpy(np.array(image)).unsqueeze(0).float() / 255.0

            if random.random() < 0.8:
                image_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s1)
            image_s1 = blur(image_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
            image_s1 = torch.from_numpy(np.array(image_s1)).unsqueeze(0).float() / 255.0

            if random.random() < 0.8:
                image_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s2)
            image_s2 = blur(image_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
            image_s2 = torch.from_numpy(np.array(image_s2)).unsqueeze(0).float() / 255.0

            return image, torch.from_numpy(image_view_id).long() - 1, image_s1, image_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.case_list)

