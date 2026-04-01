#!/bin/bash
set -e

# === SAM 3D Body Environment Setup (following official guide) ===

eval "$(conda shell.bash hook)"

# Step 1: Create or reuse conda env
ENV_NAME=fast_sam_3d_body
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "=== Reusing existing conda env: $ENV_NAME ==="
else
    echo "=== Creating conda env: $ENV_NAME ==="
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

# Step 2: Install a CUDA 12.4 build toolchain for PyTorch CUDA extensions.
# This keeps the working leaner than the full cuda-toolkit meta-package while
# still providing the dev headers Detectron2 needs (e.g. cusparse.h).
echo "=== Installing CUDA compiler/runtime headers ==="
conda install -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev ninja -y

# Use the conda-provided CUDA toolchain for extension builds.
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# Step 3: Install PyTorch (CUDA 12.4)
echo "=== Installing PyTorch ==="
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Step 4: Install Python dependencies
echo "=== Installing Python dependencies ==="
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope \
    ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black \
    pycocotools tensorboard huggingface_hub

# Step 5: Install Detectron2
echo "=== Installing Detectron2 ==="
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# Step 6: Install YOLO (ultralytics, for human detection)
echo "=== Installing YOLO ==="
pip install ultralytics

# Step 7: Install MoGe
echo "=== Installing MoGe ==="
pip install git+https://github.com/microsoft/MoGe.git

# Step 8: Install ONNX tools
echo "=== Installing ONNX tools ==="
pip install onnx onnxruntime-gpu nvtx

# Step 9: Install TensorRT
# TensorRT wheels are large and are hosted on NVIDIA's Python index.
# Set INSTALL_TENSORRT=0 to skip this step when engine conversion is not needed.
if [ "${INSTALL_TENSORRT:-1}" = "1" ]; then
    echo "=== Installing TensorRT from NVIDIA Python index (large download) ==="
    pip install --extra-index-url https://pypi.nvidia.com \
        tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs
else
    echo "=== Skipping TensorRT (optional). Set INSTALL_TENSORRT=1 to enable ==="
fi


pip install smplx numpy scipy opencv-python tqdm pyzmq pyrealsense2
pip install chumpy --no-build-isolation

# Step 10: Install SAM3 (optional, uncomment if needed)
# echo "=== Installing SAM3 ==="
# cd /tmp
# rm -rf sam3
# git clone https://github.com/facebookresearch/sam3.git
# cd sam3
# pip install -e .
# pip install decord psutil

echo "=== Environment setup complete! ==="
