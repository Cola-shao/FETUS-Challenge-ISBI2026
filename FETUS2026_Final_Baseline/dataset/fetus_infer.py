# dataset/fetus_infer.py
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.color import rgb2gray

class FETUSInferDataset(Dataset):
    """
    Dataset for inference: loads only image_h5 files, no label_h5 needed.
    Compatible with two JSON formats:
      - [{"image": "/path/a.h5", "label": "...(optional)"}, ...]
      - [{"image": "/path/a.h5"}, ...]
    """
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
            # Convert view from 1-4 to 0-3 indexing
            if "view" in f:
                view = f["view"][:]
                view = int(np.array(view).reshape(-1)[0]) - 1
            else:
                view = -1  # Placeholder when view is not available

        image_t = torch.from_numpy(image).unsqueeze(0).float()  # (1,H,W)
        view_t = torch.tensor(view, dtype=torch.long)           # ()

        # Return original path for output file naming
        return image_t, view_t, image_h5_file
