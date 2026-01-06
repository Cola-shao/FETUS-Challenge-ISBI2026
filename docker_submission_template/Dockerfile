FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG USER_ID=1000
ARG GROUP_ID=1000

# create a non-root user
RUN groupadd -g ${GROUP_ID} app && \
    useradd -m -u ${USER_ID} -g app app

# workspace
WORKDIR /ws

# copy the requirements.txt and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy other code
COPY . .

# modify permissions
RUN chown -R app:app /ws
USER app

# set environment variables
ENV DATA_DIR=/data \
    OUTPUT_DIR=/output/preds \
    MODEL_PATH=/ws/weights/best.pth

# default command
CMD ["python", "run_infer.py"]
