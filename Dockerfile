FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    USE_TF=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LAND_MODEL_PATH=/opt/app/data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask \
    LAND_MASK_DATASET=/opt/app/data/masks \
    LAND_OUTPUTS_DIR=/opt/app/web_outputs

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY requirements-inference.txt .
RUN python3 -m pip install --no-cache-dir --timeout 120 --retries 10 \
        --index-url https://pypi.org/simple pip==25.2
# Select CUDA wheels explicitly; resolve their dependencies from PyPI rather
# than the CUDA index's NVIDIA mirror. Keep download hash verification enabled.
RUN python3 -m pip install --no-cache-dir --timeout 120 --retries 10 \
        --resume-retries 10 --index-url https://pypi.org/simple \
        --find-links https://download.pytorch.org/whl/cu124/torch/ \
        --find-links https://download.pytorch.org/whl/cu124/torchvision/ \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    && python3 -m pip install --no-cache-dir --timeout 120 --retries 10 \
        --resume-retries 10 --index-url https://pypi.org/simple \
        -r requirements-inference.txt

COPY src ./src
COPY LICENSE NOTICE ./
COPY data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask/ ./data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask/
COPY data/masks/ ./data/masks/
EXPOSE 7860
ENTRYPOINT ["python3"]
CMD ["src/inference_ldm_app.py"]
