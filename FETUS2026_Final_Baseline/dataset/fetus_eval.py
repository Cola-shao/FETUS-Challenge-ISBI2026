import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2gray

class FETUSEvalDataset(Dataset):
    """
    Dataset for evaluation: loads both image_h5 and label_h5 files.
    JSON item must contain:
      {"image": ".../xxx.h5", "label": ".../xxx.h5"}
    """
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            self.case_list = json.load(f)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx: int):
        case = self.case_list[idx]
        image_h5_file = case["image"]
        label_h5_file = case["label"]

        with h5py.File(image_h5_file, "r") as f:
            image = f["image"][:]
            image = rgb2gray(image)  # (H,W) float32
            if "view" in f:
                view = f["view"][:]
                view = int(np.array(view).reshape(-1)[0]) - 1  # 0..3
            else:
                view = -1

        with h5py.File(label_h5_file, "r") as f:
            mask = f["mask"][:]     # (H,W) int
            label = f["label"][:]   # (K,) 0/1

        image_t = torch.from_numpy(image).unsqueeze(0).float()  # (1,H,W)
        view_t = torch.tensor(view, dtype=torch.long)           # ()
        mask_t = torch.from_numpy(mask).long()                  # (H,W)
        label_t = torch.from_numpy(label).long()                # (K,)

        # Return image_h5_file for finding corresponding predictions in pred_dir
        return image_t, view_t, mask_t, label_t, image_h5_file
