import torch
print(torch.__version__)
print(torch.__file__)
export LD_LIBRARY_PATH=$HOME/path/to/my/venv3115/lib64/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
/mnt/data1/yjx/anaconda3/envs/fetus-baseline/lib/python3.10/site-packages/nvidia/nvjitlink/lib

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121