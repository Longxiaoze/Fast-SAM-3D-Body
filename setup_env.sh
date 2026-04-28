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

detect_gpu_profile() {
    local gpu_names
    gpu_names="${GPU_NAME_OVERRIDE:-}"
    if [ -z "$gpu_names" ] && command -v nvidia-smi >/dev/null 2>&1; then
        gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    fi

    if echo "$gpu_names" | grep -qi "5090"; then
        echo "5090"
    elif echo "$gpu_names" | grep -qi "4090"; then
        echo "4090"
    else
        echo "unknown"
    fi
}

GPU_PROFILE=${GPU_PROFILE:-$(detect_gpu_profile)}
case "$GPU_PROFILE" in
    4090)
        DEFAULT_CUDA_CONDA_LABEL=cuda-12.4.0
        DEFAULT_CUDA_TOOLKIT_VERSION=12.4
        DEFAULT_PYTORCH_CUDA=cu124
        DEFAULT_PYTORCH_VERSION=2.5.1
        DEFAULT_TORCHVISION_VERSION=0.20.1
        ;;
    5090)
        DEFAULT_CUDA_CONDA_LABEL=cuda-12.8.0
        DEFAULT_CUDA_TOOLKIT_VERSION=12.8
        DEFAULT_PYTORCH_CUDA=cu128
        DEFAULT_PYTORCH_VERSION=2.7.1
        DEFAULT_TORCHVISION_VERSION=0.22.1
        ;;
    *)
        echo "=== Could not detect RTX 4090/5090; defaulting to RTX 5090-compatible CUDA 12.8 ==="
        DEFAULT_CUDA_CONDA_LABEL=cuda-12.8.0
        DEFAULT_CUDA_TOOLKIT_VERSION=12.8
        DEFAULT_PYTORCH_CUDA=cu128
        DEFAULT_PYTORCH_VERSION=2.7.1
        DEFAULT_TORCHVISION_VERSION=0.22.1
        ;;
esac

# Override any of these variables from the shell if a machine needs a custom build.
CUDA_CONDA_LABEL=${CUDA_CONDA_LABEL:-$DEFAULT_CUDA_CONDA_LABEL}
CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION:-$DEFAULT_CUDA_TOOLKIT_VERSION}
PYTORCH_CUDA=${PYTORCH_CUDA:-$DEFAULT_PYTORCH_CUDA}
PYTORCH_VERSION=${PYTORCH_VERSION:-$DEFAULT_PYTORCH_VERSION}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-$DEFAULT_TORCHVISION_VERSION}

echo "=== GPU profile: $GPU_PROFILE ==="
echo "=== CUDA toolkit: $CUDA_TOOLKIT_VERSION (${PYTORCH_CUDA}), PyTorch: $PYTORCH_VERSION, TorchVision: $TORCHVISION_VERSION ==="

# Step 2: Install a CUDA build toolchain for PyTorch CUDA extensions.
# This keeps the working leaner than the full cuda-toolkit meta-package while
# still providing the dev headers Detectron2 needs (e.g. cusparse.h).
echo "=== Installing CUDA compiler/runtime headers ==="
if [ "$GPU_PROFILE" = "4090" ]; then
    conda install -c nvidia/label/cuda-12.4.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev ninja -y
else
    conda install -c "nvidia/label/${CUDA_CONDA_LABEL}" \
        "cuda-nvcc=${CUDA_TOOLKIT_VERSION}.*" \
        "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}.*" \
        "cuda-libraries-dev=${CUDA_TOOLKIT_VERSION}.*" \
        ninja -y
fi

# Use the conda-provided CUDA toolchain for extension builds.
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

# Step 3: Install PyTorch
echo "=== Installing PyTorch ==="
if [ "$GPU_PROFILE" = "4090" ]; then
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124
else
    pip install "torch==${PYTORCH_VERSION}+${PYTORCH_CUDA}" "torchvision==${TORCHVISION_VERSION}+${PYTORCH_CUDA}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
fi

# Step 4: Install Python dependencies
echo "=== Installing Python dependencies ==="
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
    dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
    webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope \
    ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore iopath==0.1.9 black \
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
pip install onnx "onnxslim>=0.1.71" onnxruntime-gpu nvtx

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

pip install rerun-sdk==0.19.1

echo "=== Environment setup complete! ==="
