🤰FETUS 2026 Challenge Baseline — Quick Start

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)
[![FETUS Challenge](https://img.shields.io/badge/Visit-Official%20Site-brightgreen?style=for-the-badge&logo=google-chrome)](http://119.29.231.17:90/index.html)

**🏆 Official Baseline for FETUS 2026 Challenge** — This repository provides the official baseline implementation used in the FETUS 2026 competition. It implements a semi-supervised UniMatch pipeline for fetal cardiac segmentation and multi-label classification, and supports two backbones: a lightweight UNet and the high-performance Echocare encoder (SwinUNETR-based).

For official rules, dataset downloads and the evaluation server, visit the [FETUS 2026 challenge page](http://119.29.231.17:90/index.html).

 • [🚀 Quick Start](#-1-prepare-data) • [🔧 Configuration](#️-configuration)

---

## ✨ What's Inside

| 🚀 Feature | Description |
|------------|-------------|
| **🎯 UniMatch Framework** | State-of-the-art semi-supervised learning |
| **❤️ Fetal Cardiac Analysis** | Multi-view ultrasound segmentation & classification |
| **🔬 Echocare Model** | Pre-trained foundation model with LoRA fine-tuning |
| **⚡ High Performance** | Optimized for RTX 3090 GPU |
| **📊 Comprehensive Eval** | Dice, NSD, F1-score, and runtime metrics |

---

## 📁 1. Prepare Data

### 🎯 Data Organization Structure

```
🌟 FETUS2026_Final_Baseline/
├── 📂 data/
│   ├── 📄 train_labeled.json      # 🏷️ Labeled training config
│   ├── 📄 train_unlabeled.json    # 🔄 Unlabeled training config
│   ├── 📄 valid.json             # ✅ Validation config
│   ├── 🖼️ images/                # Medical ultrasound images
│   │   ├── 1699.h5
│   │   └── ... (thousands of cases)
│   └── 🏷️ labels/                # Ground truth annotations
│       ├── 1699_label.h5
│       └── ... (matching labels)
├── 📂 dataset/                   # 📊 Data loading utilities
│   ├── fetus_eval.py            # Evaluation dataset
│   ├── fetus_infer.py           # Inference dataset
│   └── fetus.py                 # Training dataset
├── 📂 model/                     # 🧠 Model architectures
│   ├── Echocare.py              # Echocare model with LoRA
│   └── unet.py                  # Lightweight UNet model
├── 📂 util/                      # 🛠️ Utility functions
│   └── utils.py                 # Training utilities
├── 🧠 pretrained_weights/        # 🤖 Pre-trained models
├── 📊 step_0_split_train_valid_fold.py  # 📊 Data splitting
├── 📊 step_1_unimatch_train.py   # 🚀 Semi-supervised training
├── 📊 step_2_inference.py        # 🔍 Model inference
├── 📊 step_3_evaluate.py         # 📈 Performance evaluation
├── 📦 requirements.txt           # 📋 Python dependencies
└── README.md                  # 📖 Main documentation
```

**💡 Pro Tip**: Download the training data from the [FETUS 2026 Challenge website](http://119.29.231.17:90/index.html) and extract it to the `data/` directory.

**🤖 Pre-trained Weights**: For Echocare model training, download the pre-trained Echocare encoder weights from [this link](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg) and place the `echocare_encoder.pth` file in the `pretrained_weights/` directory.

## 🧰 2. Create Python Environment

### 🛠️ System Requirements

| Component | Specification | Status |
|-----------|---------------|---------|
| **🐍 Python** | 3.10+ | ✅ Recommended |
| **🎮 CUDA** | 12.1+ (cu121) | ✅ Optimal |
| **🧠 GPU** | RTX 3090 | 🚀 High Performance |
| **💾 RAM** | 16GB+ | 📈 Recommended |
| **💽 Storage** | 50GB+ | 💾 For datasets |

---

### 🚀 One-Click Setup

```bash
# 📥 Clone the repository
git clone https://github.com/Cola-shao/FETUS-Challenge-ISBI2026.git
cd FETUS2026_Final_Baseline

# 🎯 Create isolated environment
conda create -n fetus-baseline python=3.10 -y && conda activate fetus-baseline

# 🔥 Install PyTorch ecosystem
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 📦 Install dependencies
pip install -r requirements.txt
```

> ⚠️ **CUDA Compatibility Note:**
> If your CUDA version differs, visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for compatible wheels, then run:
> ```bash
> pip install -r requirements.txt --no-deps
> ```

## 📊 3. Create Local Train / Validation Split

If you need to create custom train/validation splits from your data:

```bash
python step_0_split_train_valid_fold.py \
    --root ./data \
    --n_image_per_view 20 \
    --seed 2026
```

This creates the required JSON configuration files (train_labeled.json, train_unlabeled.json, valid.json) under the `data/` directory.

### Data Format

The dataset uses HDF5 (.h5) files for images and labels:

#### Image Files (`images/*.h5`)
```python
{
    'image': np.ndarray,  # (512, 512, 3) uint8 RGB ultrasound image
    'view': np.ndarray    # (1,) int32 view ID (1: 4CH, 2: LVOT, 3: RVOT, 4: 3VT)
}
```

#### Label Files (`labels/*_label.h5`)
```python
{
    'mask': np.ndarray,   # (512, 512) uint8 segmentation mask (0-14 classes)
    'label': np.ndarray   # (7,) uint8 binary classification labels
}
```

## 🚀 4. Train

### 🏆 Choose Your Model

| Model | Performance | Memory Usage | Best For |
|-------|-------------|--------------|----------|
| **🧠 Echocare** | ⭐⭐⭐⭐⭐ | 🏋️ High (8GB+) | **Competition Winner** |
| **🔧 UNet** | ⭐⭐⭐⭐ | 💚 Low (4GB+) | Quick Experiments |

---

### 🏆 Premium Training (Echocare)

```bash
🎯 python step_1_unimatch_train.py \
    --model echocare \
    --ssl-ckpt ./pretrained_weights/echocare_encoder.pth \
    --train-labeled-json data/train_labeled.json \
    --train-unlabeled-json data/train_unlabeled.json \
    --valid-labeled-json data/valid.json \
    --save-path ./checkpoints_baseline_echocare \
    --train-epochs 300 \
    --batch-size 8 \
    --gpu 0 \
    --amp
```

### ⚡ Fast Training (UNet)

```bash
🚀 python step_1_unimatch_train.py \
    --model unet \
    --train-labeled-json data/train_labeled.json \
    --train-unlabeled-json data/train_unlabeled.json \
    --valid-labeled-json data/valid.json \
    --save-path ./checkpoints_baseline_unet \
    --train-epochs 300 \
    --batch-size 32 \
    --gpu 0 \
    --amp
```

---

### 📊 Training Artifacts

| File | Echocare Path | UNet Path | Description |
|------|---------------|-----------|-------------|
| **🏆 Best Model** | `./checkpoints_baseline_echocare/best.pth` | `./checkpoints_baseline_unet/best.pth` | Best validation score |
| **📈 Latest Model** | `./checkpoints_baseline_echocare/latest.pth` | `./checkpoints_baseline_unet/latest.pth` | Most recent checkpoint |
| **📊 TensorBoard** | `./checkpoints_baseline_echocare/tb/` | `./checkpoints_baseline_unet/tb/` | Training logs |

---

### 📈 Real-time Monitoring

```bash
# 🔥 Launch TensorBoard
tensorboard --logdir ./checkpoints_baseline_echocare/tb    # Echocare logs
# OR
tensorboard --logdir ./checkpoints_baseline_unet/tb       # UNet logs

# 🌐 Access at: http://localhost:6006
```

## 🔍 5. Inference

### 🎯 Run Predictions

```bash
🔮 python step_2_inference.py \
    --data-json data/valid.json \
    --ckpt ./checkpoints_baseline_echocare/best.pth \  # 🔥 Echocare model
    --out-dir ./predictions \
    --mask-mode oracle \
    --cls-thr 0.5 \
    --gpu 0

# 🔄 Alternative: Use UNet checkpoint
# --ckpt ./checkpoints_baseline_unet/best.pth
```

### 📂 Output Structure

```
🎯 ./predictions/
├── 📄 case_0001_pred.h5    # ❤️ Fetal cardiac analysis
├── 📄 case_0002_pred.h5    # 🏥 Segmentation & classification
└── 📄 ... (all test cases)

📊 Each H5 file contains:
├── 🖼️ mask:  Segmentation predictions (H×W)
└── 🏷️ label: Classification results (7 classes)
```

## 📊 6. Evaluation

### 🏅 Performance Analysis

```bash
📈 python step_3_evaluate.py \
    --valid-json data/valid.json \
    --pred-dir ./predictions \
    --save-dir ./evaluation_results
```

### 🏆 Evaluation Metrics

| 🎯 Metric | Description | Range |
|-----------|-------------|--------|
| **🎲 Dice Coefficient** | Segmentation accuracy | 0-100% |
| **📏 NSD (Surface Distance)** | Boundary precision | 0-100% |
| **🎯 Macro F1-Score** | Classification quality | 0-1.0 |
| **⚡ Processing Time Score** | Normalized inference time | 0-100 |
| **🏅 Overall Score** | Combined performance | 0-100 |

**⚠️ Validation Phase Note**: During the validation phase, **only Dice Coefficient, NSD, and Macro F1-Score are evaluated**. **Processing Time Score is NOT included**, and the **maximum score is 80 points**.

---

### 📈 Detailed Scoring Method

*Information sourced from the official [FETUS 2026 Challenge website](http://119.29.231.17:90/index.html).*

We will use the following four evaluation metrics to assess the performance of the algorithm, which together form the final scoring formula:

**🎲 Dice Similarity Coefficient (DSC)**: Quantifies the overlap between the predicted segmentation result and the ground truth, with a value range of 0–1. Higher values indicate better consistency between the two segmentations.

**📏 Normalized Surface Dice (NSD)**: Evaluating medical image segmentation performance that measures the overlap between predicted and ground truth segmentation surfaces at specified tolerance distances.

**🎯 F1-score**: Measures the comprehensive accuracy of a classification model, calculated as the harmonic mean of Precision (positive predictive value) and Recall (true positive rate). It balances the two metrics, making it suitable for imbalanced datasets, with a range of 0–1 and higher values meaning better performance.

**⚡ Processing Time**: Our time score is obtained using min–max normalization. We set two time thresholds (an upper and a lower bound) to prevent extreme values from causing fluctuations in the normalization process, ensuring that the time score is evaluated within a stable range.

---

### 🏇 Final Ranking Method

The final ranking is determined by a weighted combination of **segmentation score**, **classification score**, and **processing time score**, with the scores of the three parts accounting for **40%**, **40%**, and **20%** respectively. The overall score is calculated as follows:

```
Final Score = 0.4 × Segmentation Score + 0.4 × Classification Score + 0.2 × Processing Time Score
```

Where:
- **Classification Score**: Mean F1-score across all classes
- **Segmentation Score**: Average of DSC and NSD across all structures and views
- **Processing Time Score**: Min-max normalized inference time

**1. Segmentation Score**: Calculated based on the segmentation performance of 14 anatomical structures across four standard views, with DSC and NSD equally weighted (50% each).

**2. Classification Score**: Mean F1-score across all classes to handle class imbalance.

**3. Processing Time Score**: Calculated based on inference time relative to baseline, with adaptive thresholds to prevent extreme values from affecting normalization.

## 📦 7. Package Predictions for Submission

**⚠️ ATTENTION!**

This step is for the **validation phase**. During this phase, **at least 400 validation images** will be provided. Participants are required to save their validation results in H5 format (the same format as the training set labels) and package them into "preds.tar.gz" format for submission via the registration platform.

**📊 Validation Phase Scoring:**
- Only evaluates: **Dice coefficient**, **NSD**, and **Macro F1-score**
- **Weighted total score: 80 points maximum**
- **Note**: For the final test phase, the score will include runtime consumption, and you will submit a Docker container instead of prediction files.

```bash
cd predictions
tar -czvf fetus_predictions.tar.gz *.h5
```

## ⚙️ Configuration

### View-Class Mappings

The system uses predefined mappings between views and allowed classes:

#### Segmentation Classes per View
```python
SEG_ALLOWED = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],      # 4CH (view_id = 1)
    1: [0, 1, 2, 4, 8],                # LVOT (view_id = 2)
    2: [0, 6, 8, 9, 10, 11, 12],      # RVOT (view_id = 3)
    3: [0, 9, 12, 13, 14],             # 3VT (view_id = 4)
}
```

#### Classification Classes per View
```python
CLS_ALLOWED = {
    0: [0, 1],     # 4CH (view_id = 1)
    1: [0, 2, 3], # LVOT (view_id = 2)
    2: [4, 5],     # RVOT (view_id = 3)
    3: [2, 5, 6],  # 3VT (view_id = 4)
}
```

### Loss Weights

Customize loss component weights:

```bash
python step_1_unimatch_train.py \
    --loss-weights '{
        "x_seg": 1.0, "x_cls": 1.0, "x_fp_cls": 1.0,
        "u_s1_seg": 0.25, "u_s2_seg": 0.25, "u_w_fp_seg": 0.5,
        "u_s1_mix_seg": 0.25, "u_s2_mix_seg": 0.25,
        "u_pseudo_cls": 1.0, "u_w_mix_cls": 0.25, "u_w_mix_fp_cls": 0.5
    }'
```

## 🧠 8. Notes & Tips

- 🏆 Use `--model echocare` for the best performance.
- 💚 Use `--model unet` for limited GPU memory.
- 🎯 Adjust `--batch-size` based on your GPU memory (8 for Echocare, 32 for UNet).
- 🔧 Use `--amp` for faster training with automatic mixed precision.
- 📊 Monitor training progress with TensorBoard: use the paths shown above for your chosen model

## 📚 Baseline Method Acknowledgement

This baseline is developed based on and inspired by the following works:

> **[1]** Yang L, Qi L, Feng L, et al.  
> *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation.*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

> **[2]** Zhang H, Wu Y, Zhao M, et al.  
> *A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications.*  
> arXiv preprint arXiv:2509.11752, 2025.

> **[3]** Hatamizadeh A, Nath V, Tang Y, et al.  
> *Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images.*  
> International MICCAI brainlesion workshop. Cham: Springer International Publishing, 2021: 272-284.

> **[4]** Hu E J, Shen Y, Wallis P, et al.  
> *Lora: Low-rank adaptation of large language models.*  
> ICLR, 2022, 1(2): 3.

The training method for the baseline is built upon the method in [1].
Echocare [2] is a self-supervised ultrasound foundation model trained on the Swin UNETR [3] architecture.
We further fine-tune the encoder of Echocare with LoRA [4] to adapt it to our task.

---

## 🏁 Good Luck & Happy Research!

We look forward to your participation in the FETUS 2026 Challenge 🚀

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
