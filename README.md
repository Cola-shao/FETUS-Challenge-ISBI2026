# FETUS-Challenge

## Project Overview

This project implements a semi-supervised carotid plaque segmentation and classification system. It adopts a Mean Teacher framework to perform multi-task learning on carotid ultrasound images (longitudinal and transverse views):

- Segmentation task: identify vessel and plaque regions (3-class segmentation: background, vessel, plaque)
- Classification task: determine plaque risk level (RADS2 vs RADS3-4, binary classification)

## Project Structure

```
release_version/
├── baseline/                          # Baseline model training code
│   ├── Dockerfile                     # Docker image configuration
│   ├── Dataset.py                     # Dataset loading and data augmentation
│   ├── model.py                       # Model definition
│   ├── train.py                       # Semi-supervised training script
│   ├── test.py                        # Testing and evaluation script
│   ├── augmentations/                 # Data augmentation modules
│   │   ├── __init__.py
│   │   └── ctaugment.py               # CTAugment augmentation strategy
│   └── utils/                         # Utility functions
│       ├── eval_utils.py              # Evaluation metrics (Dice, NSD, etc.)
│       ├── losses.py                  # Loss functions
│       ├── ramps.py                   # Consistency weight scheduling
│       ├── util.py                    # General utilities
│       └── vis_utils.py               # Visualization utilities
│
├── docker_submission_template/        # Docker submission template
│   ├── run_infer.py                   # Inference script
│   ├── Dockerfile                     # Docker image configuration
│   ├── requirements.txt               # Python dependencies
│   ├── models/                        # Directory to place model code (e.g. model.py)
│   └── weights/                       # Directory to place model weights (.pth files)
│
└── Readme.md                          # Project documentation
```

## Usage Guide

### 1. Get the Project Files

```bash
git clone https://github.com/xxx/xxx.git
```

The repository contains two Docker-based components: `baseline` and `docker_submission_template`, which correspond to the baseline model and the submission template, respectively.

### 2. Train the Model

#### Data Preparation

The data should be organized in the following structure:

```
data/
└── train/
    ├── train.list              # List of training samples
    ├── img/                    # Image files (.h5 format)
    │   ├── case_001.h5         # Contains long_img and trans_img datasets
    │   └── ...
    └── mask/                   # Annotation files (.h5 format)
        ├── case_001_label.h5   # Contains long_mask, trans_mask, cls datasets
        └── ...
```

#### Training Parameter Configuration

Edit the `Args` class in `baseline/train.py`:

```python
class Args:
    root_path = "/path/to/data"           # Path to the data
    max_iterations = 30000                # Maximum number of iterations
    batch_size = 24                       # Total batch size
    labeled_bs = 8                        # Batch size for labeled samples
    num_labeled = 200                     # Number of labeled samples
    base_lr = 0.01                        # Initial learning rate
    ema_decay = 0.99                      # EMA decay rate
    seg_consistency = 0.1                 # Segmentation consistency weight
    cls_consistency = 0.1                 # Classification consistency weight
```

#### Baseline Model Training

```bash
python train.py
```

During training:

- Training logs are printed every 250 iterations
- Validation is performed every 500 iterations
- Checkpoints are saved every 3000 iterations
- The best Dice model and the best classification Score model are saved automatically
- Logs and TensorBoard records are saved under `result/experiment_name/`

### 3. Model Testing

#### Configure Testing Parameters

Edit the `Args` class in `baseline/test.py`:

```python
class Args:
    root_path = "/path/to/data"            # Path to test data
    model_path = "/path/to/model.pth"      # Path to model weights
    batch_size = 1                         # Test batch size
    save_vis = "/path/to/save/results"     # Directory to save results
    num_vis_cases = 5                      # Number of cases to visualize
```

#### Run Testing

```bash
python test.py
```

### 4. About model.py ⭐

`model.py` is the core file. You can freely modify its internal implementation, but you must adhere to the following interface specification:

#### Required Interface

```python
class Model(nn.Module):
    def __init__(self, in_chns=1, seg_classes=2, cls_classes=2):
        """Initialize model"""
        pass
    
    def forward(self, x_long, x_trans):
        """
        Perform inference
        
        Args:
            data_root: Input data root directory (/input/ in Docker)
            output_dir: Output directory (/output/ in Docker)
            batch_size: Batch size
        """
        pass
```

### 4. About run_infer.py ⭐

`run_infer.py` is the core script for testing within the Docker submission environment. Please do not modify its arguments or interface; otherwise, the evaluation procedure will fail.

## Submission

#### What to Submit

- Name your model file `model.py` and place it in `docker_submission_template/models/`
- Name your weight file `best_model.pth` and place it in `docker_submission_template/weights/`

First, organize your trained model weights and inference scripts according to the directory structure above. Then submit using one of the following methods:

#### Preparation

```bash
# Enter the docker.submission_template directory and build a Docker image
cd docker_submission_template
docker build -t my-submission:latest .
```

#### Method A: Docker Hub (Recommended)

```bash
docker login
docker tag my-submission:latest YOUR_USERNAME/my-submission:latest
docker push YOUR_USERNAME/my-submission:latest

# Send the image address to the organizing committee:
# YOUR_USERNAME/my-submission:latest
```

#### Method B: Save as File

```bash
docker save -o my-submission.tar my-submission:latest

# Upload my-submission.tar to your preferred cloud storage
# Send the download link to the organizing committee
```

We provide three submission channels:

- Docker Hub (for Method A): Push the Docker image to Docker Hub and send the image link by email
- Google Drive (for Method B): Upload the Docker image to Google Drive and submit the share link by email
- Baidu Netdisk (for Method B): Upload the Docker image to Baidu Netdisk and submit the share link by email

[EMAIL LINK](mailto:zhuzhiyuan113@163.com)

## FAQ

### Q1: Docker image push failed frequently?

Answer:

- From mainland China, you may need to use a proxy server to access Docker Hub
- Alternatively, you can submit your image via Google Drive or Baidu Netdisk
