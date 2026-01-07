# 🧑‍🍼 FETUS 2026 Challenge Baseline — Quick Start

This repository provides the official baseline implementation for the **FETUS Challenge**, including:
- Semi-supervised learning with UniMatch framework
- Fetal cardiac segmentation and classification
- Multi-view ultrasound image analysis (4CH, LVOT, RVOT, 3VT)

The baseline is designed to be **reproducible, extensible, and competition-ready**.

---

## 📁 1. Prepare Data

Place the downloaded training archive (provided by organizers) under the repository root and organize it as follows:

```text
FETUS2026_Final_Baseline/
└─ data/
   ├─ train_labeled.json      # Labeled training data configuration
   ├─ train_unlabeled.json    # Unlabeled training data configuration
   ├─ valid.json             # Validation/test data configuration
   ├─ images/                # Directory containing all image H5 files
   │  ├── 1699.h5
   │  └── ...
   └─ labels/                # Directory containing all label H5 files
      ├── 1699_label.h55!
      └── ...
```

## 🧰 2. Create Python Environment

We recommend **Python 3.10** and **CUDA 12.1 (cu121)**.
Minimum recommended PyTorch version: **>= 2.0.0**.

```bash
conda create -n fetus-baseline python=3.10 -y
conda activate fetus-baseline

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> ⚠️ **Important:**
> If your CUDA version differs, install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/) first, then run:
> ```bash
> pip install -r requirements.txt --no-deps
> ```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 16GB+ RAM, 8GB+ GPU memory
- **Storage**: 50GB+ for datasets and checkpoints

## 📊 3. Create Local Train / Validation Split

If you need to create custom train/validation splits from your data:

```bash
python step_0_split_train_valid_fold.py \
    --data-root ./data \
    --output-dir ./data_splits \
    --labeled-ratio 0.1 \
    --fold 0
```

This creates the required JSON configuration files under the `data/` directory.

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

### Recommended (High Performance — Echocare)

```bash
python step_1_unimatch_train.py \
  --model echocare \
  --ssl-ckpt ./pretrained_weights/echocare_encoder.pth \
  --train-labeled-json data/train_labeled.json \
  --train-unlabeled-json data/train_unlabeled.json \
  --valid-labeled-json data/valid.json \
  --save-path ./checkpoints_echocare \
  --train-epochs 300 \
  --batch-size 8 \
  --gpu 0
  --amp
```

### Lightweight Option (Lower Memory — UNet)

```bash
python step_1_unimatch_train.py \
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

Training checkpoints:
```text
./checkpoints_echocare/best.pth
./checkpoints_echocare/latest.pth
```

### Training Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./checkpoints_echocare/tb
```

## 🔍 5. Inference

Run inference on test/validation images:

```bash
python step_2_inference.py \
  --data-json data/valid.json \
  --ckpt ./checkpoints_echocare/best.pth \
  --out-dir ./predictions \
  --mask-mode oracle \
  --cls-thr 0.5 \
  --gpu 0
```

Predictions are saved to:

```text
./predictions/{case_name}_pred.h5
```

Each file contains:
- `mask`: Predicted segmentation mask
- `label`: Predicted classification labels

## 📊 6. Evaluation

Evaluate model performance on validation set:

```bash
python step_3_evaluate.py \
  --valid-json data/valid.json \
  --pred-dir ./predictions \
  --save-dir ./evaluation_results
```

### Evaluation Metrics
- **Segmentation**: Dice coefficient, Normalized Surface Distance (NSD)
- **Classification**: Macro F1-score, per-class F1-score
- **Overall Score**: Combined segmentation and classification performance

## 📦 7. Package Predictions for Submission

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
- 🪶 Use `--model unet` for limited GPU memory.
- 🎯 Adjust `--batch-size` based on your GPU memory (8 for Echocare, 32 for UNet).
- 🔧 Use `--amp` for faster training with automatic mixed precision.
- 📊 Monitor training progress with TensorBoard: `tensorboard --logdir ./checkpoints/tb`

## 📚 Baseline Method Acknowledgement

This baseline is developed based on and inspired by the following works:

> **[1]** Berthelot D, Carlini N, Goodfellow I, et al.  
> *MixMatch: A Holistic Approach to Semi-Supervised Learning.*  
> Advances in Neural Information Processing Systems (NeurIPS), 2019.

> **[2]** Yang L, Qi L, Feng L, et al.  
> *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation.*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

> **[3]** Zhang H, Wu Y, Zhao M, et al.  
> *A Fully Open and Generalizable Foundation Model for Ultrasound Clinical Applications.*  
> arXiv preprint arXiv:2509.11752, 2025.

These studies provide the theoretical foundation and training paradigm upon which this baseline is constructed.

---

## 🏁 Good Luck & Happy Research!

We look forward to your participation in the FETUS 2026 Challenge 🚀

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.