FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG USER_ID=1000
ARG GROUP_ID=1000

# 创建一个非 root 用户
RUN groupadd -g ${GROUP_ID} app && \
    useradd -m -u ${USER_ID} -g app app

# 工作目录
WORKDIR /workspace

# 先复制 requirements 并安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制剩余代码
COPY . .

# 修改权限
RUN chown -R app:app /workspace
USER app

# 接口约定：通过环境变量告诉 run_inference 输入输出和权重路径
ENV DATA_DIR=/data \
    OUTPUT_DIR=/output/preds \
    MODEL_PATH=/workspace/weights/best_model.pth

# 默认启动命令：直接跑推理
CMD ["python", "run_infer.py"]
