#!/usr/bin/env python3
import os
import argparse
import sys
import csv
import glob
import time
import uuid
import shutil
import traceback
from pathlib import Path
from threading import Lock, Thread
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
from flask import Flask, request, send_file, send_from_directory, url_for, render_template_string, abort
from PIL import Image
from scipy.ndimage import label, center_of_mass

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils.utils_lidc3D import LIDCVolumes, CondLatentDiffusionPipeline_LIDC3D
from diffusers import DDPMScheduler
from unet.unet import UNetModel
from vae.autoencoder_kl import AutoencoderKlReducedMaisi


# -----------------------------------------------------------------------------
# Runtime configuration (environment defaults, overridden by CLI options)
# -----------------------------------------------------------------------------
HOST = os.environ.get("LAND_HOST", "0.0.0.0")
PORT = int(os.environ.get("LAND_PORT", "7860"))
WEB_OUTPUTS_DIR = os.environ.get("LAND_OUTPUTS_DIR", str(REPO_DIR / "web_outputs"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional HTTPS support:
# set these env vars if you want HTTPS:
#   SSL_CERT_FILE=/path/to/cert.pem
#   SSL_KEY_FILE=/path/to/key.pem
SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE")
SSL_KEY_FILE = os.environ.get("SSL_KEY_FILE")

CKPT_DIR = os.environ.get(
    "LAND_MODEL_PATH",
    str(REPO_DIR / "data" / "ckpts" / "2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask"),
)

# MASK_DATASET = "/media/diskA/roger/MICCAI25_data/Datasets/LIDC_preprocdata_256_v4"
MASK_DATASET = os.environ.get("LAND_MASK_DATASET", str(REPO_DIR / "data" / "masks"))

LATENTS_DIR = os.environ.get("LAND_LATENTS_DIR")
CREATE_VIDEOS = True

# Model list shown in the UI.
# Add as many checkpoints as you want here.
MODEL_CONFIGS = {
    "ldm_nodule_lung": {
        "label": "LDM conditioned to nodule + lung masks",
        "model_path": CKPT_DIR,
        "mask_mode": os.environ.get("LAND_MASK_MODE", "nodule+lung"),
        "vae_path": os.environ.get("LAND_VAE_PATH"),
        "vae_mask_path": os.environ.get("LAND_VAE_MASK_PATH"),
        "mask_dataset": MASK_DATASET,
        "patchbased": False,
    },
    # Example additional entries:
    # "ldm_nodule": {
    #     "label": "LDM conditioned to nodule masks",
    #     "model_path": CKPT_DIR,
    #     "mask_mode": "nodule",
    #     "mask_dataset": MASK_DATASET,
    #     "patchbased": False,
    # },
    # "ldm_unconditional": {
    #     "label": "LDM unconditional",
    #     "model_path": CKPT_DIR,
    #     "mask_mode": "none",
    #     "mask_dataset": None,
    #     "patchbased": False,
    # },
}

MODEL_CHOICES = [(k, v["label"]) for k, v in MODEL_CONFIGS.items()]

PARAM_LIMITS = {
    "n_images": {"min": 1, "max": 100},
    "batch_size": {"min": 1, "max": 2},
    "inference_steps": {"min": 1, "max": 1000},
}

RUN_SUBDIRS = {
    "input_masks": "input_masks",
    "output_images": "output_images",
    "preview_png": "preview_png",
    "preview_mp4": "preview_mp4",
}

# -----------------------------------------------------------------------------
# App state
# -----------------------------------------------------------------------------
app = Flask(__name__)
RUNS = {}
RUNS_LOCK = Lock()
PREVIEW_MASK_CACHE = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def uint8_from_float01(im: np.ndarray) -> np.ndarray:
    im = np.clip(im, 0.0, 1.0)
    return (im * 255.0).astype(np.uint8)


def extract_centroids(mask):
    components, count = label(mask > 0)
    return center_of_mass(mask > 0, components, range(1, count + 1))


def get_preview_mask_and_centroid(mask_dataset: str, mask_mode: str, sample_idx: int = 0):
    inference_dataset = LIDCVolumes(mask_dataset, mask_mode=mask_mode, masks_only=True)
    volume = inference_dataset[sample_idx]
    mask = volume["mask"]

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # remove channel dim if present
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]

    # centroid should come from a binary nodule mask
    if mask_mode == "nodule+lung+texture":
        nodule_mask = ((mask * 5) >= 1).astype(np.float32)
    else:
        nodule_mask = (mask >= 1).astype(np.float32)

    centroids = extract_centroids(nodule_mask)
    if not centroids:
        return mask.astype(np.float32), nodule_mask, tuple(s // 2 for s in mask.shape)

    centroid = tuple(map(int, map(round, centroids[0])))
    return mask.astype(np.float32), nodule_mask.astype(np.float32), centroid


def ensure_run_subdirs(out_dir: str) -> dict[str, str]:
    subdirs = {name: os.path.join(out_dir, dirname) for name, dirname in RUN_SUBDIRS.items()}
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs


def load_input_mask(mask_dataset: str, mask_mode: str, sample_idx: int) -> np.ndarray:
    inference_dataset = LIDCVolumes(mask_dataset, mask_mode=mask_mode, masks_only=True)
    volume = inference_dataset[sample_idx]
    mask = volume["mask"]

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    return mask.astype(np.float32)


def save_input_mask(mask_dataset: str, mask_mode: str, sample_idx: int, out_path: str):
    mask = load_input_mask(mask_dataset=mask_dataset, mask_mode=mask_mode, sample_idx=sample_idx)
    np.save(out_path, mask)


def create_preview_png_from_generated_volume(
    npy_path: str,
    png_path: str,
    mask_dataset: str,
    mask_mode: str,
    sample_idx: int = 0,
):
    from utils.preproc_lidc_npy import get_ct_thumbnail, get_mask_thumbnail
    vol = np.load(npy_path).astype(np.float32)

    # Accept:
    #   (256,256,256)
    #   (1,256,256,256)
    #   (3,256,256,256)
    if vol.ndim == 4:
        # channel-first volume
        if vol.shape[0] == 1:
            vol = vol[0]
        elif vol.shape[0] == 3:
            # If the generated sample is RGB-like with identical or near-identical channels,
            # use the first channel for CT preview.
            # If channels differ, averaging is a safe visualization fallback.
            if np.allclose(vol[0], vol[1]) and np.allclose(vol[1], vol[2]):
                vol = vol[0]
            else:
                vol = vol.mean(axis=0)
        else:
            raise RuntimeError(f"Unsupported 4D generated volume shape {vol.shape}")
    elif vol.ndim != 3:
        raise RuntimeError(f"Expected generated volume to be 3D or 4D, got shape {vol.shape}")

    # normalize CT volume to 0-1 for display
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)

    mask_for_display, nodule_mask, centroid = get_preview_mask_and_centroid(
        mask_dataset=mask_dataset,
        mask_mode=mask_mode,
        sample_idx=sample_idx,
    )

    mask_thumb = get_mask_thumbnail(mask_for_display, centroid)
    ct_thumb = get_ct_thumbnail(vol, nodule_mask, centroid)

    preview = np.vstack([mask_thumb, ct_thumb])
    Image.fromarray(uint8_from_float01(preview)).save(png_path)


def human_readable_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0


def validate_int_param(name: str, value: int) -> int:
    limits = PARAM_LIMITS[name]
    value = int(value)
    if value < limits["min"] or value > limits["max"]:
        raise ValueError(f"{name} must be between {limits['min']} and {limits['max']}.")
    return value


def zip_output_dir(out_dir: str) -> str:
    parent_dir = os.path.dirname(out_dir)
    base_name = os.path.join(parent_dir, Path(out_dir).name + "_synthetic_data")
    return shutil.make_archive(base_name, "zip", root_dir=out_dir)


def normalize_slice_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def create_preview_png_from_npy(npy_path: str, png_path: str):
    vol = np.load(npy_path)

    # Expecting a 3D volume. If shape is not 3D, do a best-effort preview.
    if vol.ndim == 3:
        z = vol.shape[0] // 2
        img = vol[z]
    elif vol.ndim == 4:
        # Common fallback if channel dimension exists
        img = vol[0, vol.shape[1] // 2]
    else:
        img = np.squeeze(vol)
        if img.ndim != 2:
            raise RuntimeError(f"Cannot create preview from shape {vol.shape}")

    img_u8 = normalize_slice_to_uint8(img)
    Image.fromarray(img_u8).save(png_path)


def find_generated_npy_paths(output_images_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(output_images_dir, "image_*.npy")))


def find_first_generated_npy(output_images_dir: str) -> str:
    npy_paths = find_generated_npy_paths(output_images_dir)
    if not npy_paths:
        raise RuntimeError("No generated .npy files were found.")
    return npy_paths[0]


def volume_to_3d_array(arr: np.ndarray, name: str) -> np.ndarray:
    """Convert common channel-first 3D volume layouts to a single 3D array."""
    arr = np.asarray(arr)

    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] == 3:
            if np.allclose(arr[0], arr[1]) and np.allclose(arr[1], arr[2]):
                arr = arr[0]
            else:
                arr = arr.mean(axis=0)
        else:
            raise RuntimeError(f"Unsupported {name} volume shape {arr.shape}; expected 3D or channel-first 4D.")

    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported {name} volume shape {arr.shape}; expected 3D volume.")

    if arr.shape != (256, 256, 256):
        raise RuntimeError(f"Unsupported {name} volume shape {arr.shape}; expected (256, 256, 256).")

    return arr.astype(np.float32)


def normalize_volume_to_uint8(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.uint8)
    vol = (vol - vmin) / (vmax - vmin)
    return (vol * 255.0).clip(0, 255).astype(np.uint8)


def orthogonal_slice_frame(vol_u8: np.ndarray, idx: int) -> np.ndarray:
    frame = np.hstack([
        vol_u8[idx, :, :],
        vol_u8[:, idx, :],
        vol_u8[:, :, idx],
    ])
    return np.repeat(frame[:, :, None], 3, axis=2)


def save_orthogonal_slice_video(npy_path: str, mp4_path: str, name: str, fps: int = 16):
    if imageio is None:
        raise RuntimeError("imageio is required to write MP4 previews. Install it with: pip install imageio imageio-ffmpeg")

    vol = volume_to_3d_array(np.load(npy_path), name=name)
    vol_u8 = normalize_volume_to_uint8(vol)
    with imageio.get_writer(mp4_path, fps=fps, macro_block_size=16) as writer:
        for i in range(256):
            writer.append_data(orthogonal_slice_frame(vol_u8, i))


def overlay_mask_on_ct_slice(
    ct_slice_u8: np.ndarray,
    mask_slice: np.ndarray,
    opacity: float = 0.3,
) -> np.ndarray:
    """Overlay lung and nodule mask values on a grayscale CT slice.

    Mask convention:
      - 0.0: no overlay
      - 0.5: lung, pink/purple
      - >=1.0: nodule, light blue
    """
    ct_rgb = np.repeat(ct_slice_u8[:, :, None], 3, axis=2).astype(np.float32)

    # Colors chosen to match the attached example preview.
    lung_color = np.array([150, 30, 105], dtype=np.float32)
    nodule_color = np.array([120, 220, 210], dtype=np.float32)

    lung_region = np.isclose(mask_slice, 0.5, atol=1e-3)
    nodule_region = mask_slice >= 1.0

    overlay = ct_rgb.copy()
    overlay[lung_region] = (1.0 - opacity) * overlay[lung_region] + opacity * lung_color
    overlay[nodule_region] = (1.0 - opacity) * overlay[nodule_region] + opacity * nodule_color

    return overlay.clip(0, 255).astype(np.uint8)


def orthogonal_overlay_frame(ct_u8: np.ndarray, mask: np.ndarray, idx: int, opacity: float = 0.3) -> np.ndarray:
    frames = [
        overlay_mask_on_ct_slice(ct_u8[idx, :, :], mask[idx, :, :], opacity=opacity),
        overlay_mask_on_ct_slice(ct_u8[:, idx, :], mask[:, idx, :], opacity=opacity),
        overlay_mask_on_ct_slice(ct_u8[:, :, idx], mask[:, :, idx], opacity=opacity),
    ]
    return np.hstack(frames)


def save_orthogonal_overlay_video(
    output_npy_path: str,
    input_mask_npy_path: str,
    mp4_path: str,
    opacity: float = 0.3,
    fps: int = 16,
    mask_mode: str = "nodule+lung",
):
    if imageio is None:
        raise RuntimeError("imageio is required to write MP4 previews. Install it with: pip install imageio imageio-ffmpeg")

    ct = volume_to_3d_array(np.load(output_npy_path), name="output CT")
    mask = volume_to_3d_array(np.load(input_mask_npy_path), name="input mask")
    if "texture" in mask_mode:
        mask = mask * 5
    ct_u8 = normalize_volume_to_uint8(ct)

    with imageio.get_writer(mp4_path, fps=fps, macro_block_size=16) as writer:
        for i in range(256):
            writer.append_data(orthogonal_overlay_frame(ct_u8, mask, i, opacity=opacity))


def save_nodule_slice_preview_pngs(
    output_npy_path: str,
    input_mask_npy_path: str,
    preview_png_dir: str,
    sample_idx: int,
    mask_dataset: str,
    mask_mode: str,
    opacity: float = 0.3,
):
    """Save CT, mask, and overlay PNGs using nodule-centered orthogonal views.

    The slice coordinates come from get_preview_mask_and_centroid(), so each PNG
    contains the axial, coronal, and sagittal views passing through the detected
    nodule centroid, horizontally stacked as [axis 0, axis 1, axis 2].
    """
    ct = volume_to_3d_array(np.load(output_npy_path), name="output CT")
    mask = volume_to_3d_array(np.load(input_mask_npy_path), name="input mask")
    ct_u8 = normalize_volume_to_uint8(ct)
    mask_u8 = normalize_volume_to_uint8(mask)

    nodules = (mask * 5 >= 1) if "texture" in mask_mode else (mask >= 1)
    centroids = extract_centroids(nodules)
    centroid = centroids[0] if centroids else tuple(s // 2 for s in mask.shape)
    if "texture" in mask_mode:
        mask = mask * 5
    axis0_idx = int(np.clip(int(round(centroid[0])), 0, ct_u8.shape[0] - 1))
    axis1_idx = int(np.clip(int(round(centroid[1])), 0, ct_u8.shape[1] - 1))
    axis2_idx = int(np.clip(int(round(centroid[2])), 0, ct_u8.shape[2] - 1))

    ct_png_path = os.path.join(preview_png_dir, f"ct_{sample_idx:05d}.png")
    mask_png_path = os.path.join(preview_png_dir, f"mask_{sample_idx:05d}.png")
    overlay_png_path = os.path.join(preview_png_dir, f"overlay_{sample_idx:05d}.png")

    ct_preview = np.hstack([
        ct_u8[axis0_idx, :, :],
        ct_u8[:, axis1_idx, :],
        ct_u8[:, :, axis2_idx],
    ])
    mask_preview = np.hstack([
        mask_u8[axis0_idx, :, :],
        mask_u8[:, axis1_idx, :],
        mask_u8[:, :, axis2_idx],
    ])

    overlay_preview = np.hstack([
        overlay_mask_on_ct_slice(ct_u8[axis0_idx, :, :], mask[axis0_idx, :, :], opacity=opacity),
        overlay_mask_on_ct_slice(ct_u8[:, axis1_idx, :], mask[:, axis1_idx, :], opacity=opacity),
        overlay_mask_on_ct_slice(ct_u8[:, :, axis2_idx], mask[:, :, axis2_idx], opacity=opacity),
    ])

    Image.fromarray(ct_preview).save(ct_png_path)
    Image.fromarray(mask_preview).save(mask_png_path)
    Image.fromarray(overlay_preview).save(overlay_png_path)

    return ct_png_path, mask_png_path, overlay_png_path


def create_preview_mp4_videos(output_npy_path: str, input_mask_npy_path: str, preview_mp4_dir: str, sample_idx: int, mask_mode: str = "nodule+lung"):
    ct_mp4_path = os.path.join(preview_mp4_dir, f"ct_{sample_idx:05d}.mp4")
    mask_mp4_path = os.path.join(preview_mp4_dir, f"mask_{sample_idx:05d}.mp4")
    overlay_mp4_path = os.path.join(preview_mp4_dir, f"overlay_{sample_idx:05d}.mp4")

    save_orthogonal_slice_video(output_npy_path, ct_mp4_path, name="output CT")
    save_orthogonal_slice_video(input_mask_npy_path, mask_mp4_path, name="input mask")
    save_orthogonal_overlay_video(output_npy_path, input_mask_npy_path, overlay_mp4_path, opacity=0.3, mask_mode=mask_mode)

    return ct_mp4_path, mask_mp4_path, overlay_mp4_path


class RunLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def write(self, text):
        if not text:
            return
        with RUNS_LOCK:
            if self.run_id in RUNS:
                RUNS[self.run_id]["logs"] += text
                RUNS[self.run_id]["updated_at"] = time.time()

    def flush(self):
        pass


def append_run_log(run_id: str, text: str):
    if not text:
        return
    with RUNS_LOCK:
        if run_id in RUNS:
            RUNS[run_id]["logs"] += text + "\n"
            RUNS[run_id]["updated_at"] = time.time()


# -----------------------------------------------------------------------------
# LDM pipeline loading and generation logic
# -----------------------------------------------------------------------------
def load_ldm_pipeline(
    model_path: str,
    mask_mode: str = "none",
    mask_dataset: str | None = None,
    patchbased: bool = False,
    vae_path: str | None = None,
    vae_mask_path: str | None = None,
):
    print("Loading Diffusion pipeline from:")
    print(f"    - {model_path}\n")

    unet = UNetModel.from_pretrained(model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.eval()
    unet = unet.to(DEVICE)

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"U-Net total params: {total_params}")
    print(f"U-Net trainable params: {trainable_params}")
    print("U-Net model loaded in eval mode")

    print("Loading VAE...")
    vae = AutoencoderKlReducedMaisi.from_pretrained(vae_path or os.path.join(model_path, "vae"))
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(DEVICE)
    latent_channels = vae.latent_channels
    print("VAE loaded")

    params_vae = sum(p.numel() for p in vae.parameters())
    trainable_params_vae = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE total params: {params_vae}")
    print(f"VAE trainable params: {trainable_params_vae}")

    mask_encoder = None
    mask_encoder_path = vae_mask_path or os.path.join(model_path, "vae_mask")
    if vae_mask_path or os.path.exists(mask_encoder_path):
        print("Loading mask encoder...")
        mask_encoder = AutoencoderKlReducedMaisi.from_pretrained(mask_encoder_path)
        mask_encoder.requires_grad_(False)
        mask_encoder.eval()
        mask_encoder = mask_encoder.to(DEVICE)
        print("Mask encoder loaded in eval mode")

        params_mask = sum(p.numel() for p in mask_encoder.parameters())
        trainable_params_mask = sum(p.numel() for p in mask_encoder.parameters() if p.requires_grad)
        print(f"Mask encoder total params: {params_mask}")
        print(f"Mask encoder trainable params: {trainable_params_mask}")

    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipeline = CondLatentDiffusionPipeline_LIDC3D(
        unet=unet,
        scheduler=noise_scheduler,
        vae=vae,
        maskEncoder=mask_encoder,
        latent_channels=latent_channels,
        patchbased=patchbased,
        mask_mode=mask_mode,
        mask_dataset=mask_dataset,
    )
    print("Diffusion pipeline is ready\n")
    return pipeline


def maybe_load_latent(latents_dir: str | None, idx: int, device):
    if not latents_dir:
        return None

    latent_path = os.path.join(latents_dir, f"latent_{idx}.pt")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    latent = torch.load(latent_path, map_location=device)
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent.to(device)


def maybe_save_labels(save_dir: str, out_paths: list[str], classes):
    csv_path = os.path.join(save_dir, "image_labels.csv")
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for path, cls in zip(out_paths, classes):
            writer.writerow([os.path.basename(path), cls])


def generate_images_ldm_web(
    pipeline,
    out_dir: str,
    n_images: int,
    batch_size: int,
    inference_steps: int,
    mask_mode: str,
    mask_dataset: str | None,
    latents_dir: str | None = None,
    logger=None,
):
    def log(msg: str):
        if logger is not None:
            logger.write(msg.rstrip() + "\n")

    subdirs = ensure_run_subdirs(out_dir)
    input_masks_dir = subdirs["input_masks"]
    output_images_dir = subdirs["output_images"]
    preview_png_dir = subdirs["preview_png"]
    preview_mp4_dir = subdirs["preview_mp4"]

    existing_files = sorted([f for f in os.listdir(output_images_dir) if f.endswith(".npy")])
    log(f"save dir is {out_dir}")
    log(f"input masks dir is {input_masks_dir}")
    log(f"output images dir is {output_images_dir}")
    log(f"preview PNG dir is {preview_png_dir}")
    log(f"preview MP4 dir is {preview_mp4_dir}")
    log(f"Found {len(existing_files)} existing output samples in {output_images_dir}")

    mask_count = None
    if mask_mode != "none":
        mask_count = len(LIDCVolumes(mask_dataset, mask_mode=mask_mode, masks_only=True))
        if mask_count == 0:
            raise ValueError("No conditioning masks were found.")

    start_idx = 0
    while start_idx < n_images:
        end_idx = min(start_idx + batch_size, n_images)
        mask_start_idx = start_idx
        if mask_count is not None:
            mask_start_idx = start_idx % mask_count
            # Keep each pipeline batch within the dataset, then restart at mask 0.
            end_idx = min(end_idx, start_idx + mask_count - mask_start_idx)

        out_paths = []
        latents = []

        for i in range(start_idx, end_idx):
            out_path = os.path.join(output_images_dir, f"image_{i:05d}.npy")
            mask_path = os.path.join(input_masks_dir, f"mask_{i:05d}.npy")

            out_paths.append(out_path)

            latent = maybe_load_latent(latents_dir, i, pipeline.unet.device) if latents_dir else None
            if latent is not None:
                latents.append(latent)

        if len(out_paths) == 0:
            continue

        if len(latents) > 0:
            latents = torch.stack(latents, dim=0)
        else:
            latents = None

        current_batch_size = len(out_paths)
        log(f"\nSampling indices {start_idx}–{end_idx}, generating {current_batch_size} new samples.")
        
        with torch.inference_mode():
            output = pipeline(
                latents=latents,
                height=256,
                width=256,
                batch_size=current_batch_size,
                num_inference_steps=inference_steps,
                output_type="numpy",
                return_dict=False,
                renormalize=False,
                return_latents=False,
                start_indx=mask_start_idx,
            )

        if mask_mode != "none":
            if "texture" in mask_mode:
                images_, conditioning_masks, _, classes = output
                maybe_save_labels(output_images_dir, out_paths, classes)
            else:
                images_, conditioning_masks, _ = output
        else:
            images_ = output

        for batch_idx, (img, out_path) in enumerate(zip(images_, out_paths)):
            # img.shape is (3, 256, 256, 256) because output vals were replicated 3 times for RGB-like display
            if img.shape[0] == 3:
                img = img[0]
                assert img.shape[0] == 256
                assert img.shape[1] == 256
                assert img.shape[2] == 256
            np.save(out_path, img)

            if mask_mode != "none" and mask_dataset:
                sample_idx = int(Path(out_path).stem.split("_")[-1])
                mask_path = os.path.join(input_masks_dir, f"mask_{sample_idx:05d}.npy")
                mask = conditioning_masks[batch_idx]
                if mask.shape[0] > 1:
                    # Convert the actual one-hot conditioning back to display values.
                    class_ids = mask.argmax(axis=0)
                    lung_id = 6 if "texture" in mask_mode else 2
                    mask = class_ids.astype(np.float32)
                    if "texture" in mask_mode:
                        mask /= 5.0
                    mask[class_ids == lung_id] = 0.1 if "texture" in mask_mode else 0.5
                np.save(mask_path, mask)

        del images_, latents
        start_idx = end_idx

    log("Finished inference")

# -----------------------------------------------------------------------------
# Generator app
# -----------------------------------------------------------------------------
class GeneratorApp:
    def __init__(self):
        self.pipelines = {}
        self.lock = Lock()

    def get_model_config(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")
        return MODEL_CONFIGS[model_key]

    def load_pipeline_for_model(self, model_key: str):
        if model_key in self.pipelines:
            return self.pipelines[model_key]

        cfg = self.get_model_config(model_key)
        model_path = cfg["model_path"]
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found for '{model_key}': {model_path}")

        pipeline = load_ldm_pipeline(
            model_path=model_path,
            mask_mode=cfg.get("mask_mode", "none"),
            mask_dataset=cfg.get("mask_dataset"),
            patchbased=cfg.get("patchbased", False),
            vae_path=cfg.get("vae_path"),
            vae_mask_path=cfg.get("vae_mask_path"),
        )
        self.pipelines[model_key] = pipeline
        return pipeline

    def create_run(self, model_key: str, n_images: int, batch_size: int, inference_steps: int):
        n_images = validate_int_param("n_images", n_images)
        batch_size = validate_int_param("batch_size", batch_size)
        inference_steps = validate_int_param("inference_steps", inference_steps)
        cfg = self.get_model_config(model_key)
        if not os.path.isdir(cfg["model_path"]):
            raise ValueError(f"Checkpoint directory does not exist: {cfg['model_path']}")
        warning = ""
        if cfg.get("mask_mode", "none") != "none":
            dataset = LIDCVolumes(cfg["mask_dataset"], mask_mode=cfg["mask_mode"], masks_only=True)
            if len(dataset) == 0:
                raise ValueError("No conditioning masks were found.")
            if len(dataset) < n_images:
                warning = (
                    f"WARNING: Requested {n_images} samples, but only {len(dataset)} conditioning masks are available. "
                    "The available masks will be reused cyclically until the requested sample count is reached. "
                    "Please contact the software authors for more conditioning masks.\n"
                )

        run_id = f"{model_key}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = os.path.join(WEB_OUTPUTS_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)

        with RUNS_LOCK:
            RUNS[run_id] = {
                "status": "running",
                "logs": warning,
                "error": None,
                "result": None,
                "updated_at": time.time(),
                "form_values": {
                    "model_key": model_key,
                    "n_images": n_images,
                    "batch_size": batch_size,
                    "inference_steps": inference_steps,
                },
            }

        thread = Thread(
            target=self._run_generation_task,
            args=(run_id, model_key, n_images, batch_size, inference_steps, out_dir),
            daemon=True,
        )
        thread.start()
        return run_id

    def _run_generation_task(
        self,
        run_id: str,
        model_key: str,
        n_images: int,
        batch_size: int,
        inference_steps: int,
        out_dir: str,
    ):
        logger = RunLogger(run_id)

        try:
            cfg = self.get_model_config(model_key)
            append_run_log(run_id, f"Selected model: {cfg['label']}")
            append_run_log(run_id, f"Model path: {cfg['model_path']}")
            append_run_log(run_id, "Loading pipeline...")

            with self.lock:
                with redirect_stdout(logger), redirect_stderr(logger):
                    pipeline = self.load_pipeline_for_model(model_key)
                    append_run_log(run_id, "Pipeline loaded")
                    device = pipeline.unet.device
                    device_type = "GPU" if device.type == "cuda" else "CPU"
                    print(f"Inference device: {device_type} ({device})", flush=True)
                    append_run_log(run_id, "Starting image generation...")
                    generate_images_ldm_web(
                        pipeline=pipeline,
                        out_dir=out_dir,
                        n_images=n_images,
                        batch_size=batch_size,
                        inference_steps=inference_steps,
                        mask_mode=cfg.get("mask_mode", "none"),
                        mask_dataset=cfg.get("mask_dataset"),
                        latents_dir=LATENTS_DIR,
                        logger=logger,
                    )
            
            
            append_run_log(run_id, "Creating preview PNG image(s) and MP4 video(s)...")
            t0 = time.time()
            subdirs = ensure_run_subdirs(out_dir)
            input_masks_dir = subdirs["input_masks"]
            output_images_dir = subdirs["output_images"]
            preview_png_dir = subdirs["preview_png"]
            preview_mp4_dir = subdirs["preview_mp4"]
            generated_npy_paths = find_generated_npy_paths(output_images_dir)
            if not generated_npy_paths:
                raise RuntimeError("No generated .npy files were found.")

            preview_png = None
            for npy_path in generated_npy_paths:
                sample_idx = int(Path(npy_path).stem.split("_")[-1])
                input_mask_npy = os.path.join(input_masks_dir, f"mask_{sample_idx:05d}.npy")

                if cfg.get("mask_dataset") and cfg.get("mask_mode", "none") != "none" and os.path.isfile(input_mask_npy):
                    _, _, overlay_png_path = save_nodule_slice_preview_pngs(
                        output_npy_path=npy_path,
                        input_mask_npy_path=input_mask_npy,
                        preview_png_dir=preview_png_dir,
                        sample_idx=sample_idx,
                        mask_dataset=cfg["mask_dataset"],
                        mask_mode=cfg.get("mask_mode", "none"),
                        opacity=0.3,
                    )
                    current_preview_png = overlay_png_path

                    if CREATE_VIDEOS:
                        create_preview_mp4_videos(
                            output_npy_path=npy_path,
                            input_mask_npy_path=input_mask_npy,
                            preview_mp4_dir=preview_mp4_dir,
                            sample_idx=sample_idx,
                            mask_mode=cfg.get("mask_mode", "none"),
                        )
                else:
                    current_preview_png = os.path.join(preview_png_dir, f"ct_{sample_idx:05d}.png")
                    create_preview_png_from_npy(npy_path, current_preview_png)

                if preview_png is None:
                    preview_png = current_preview_png

            elapsed_time = time.time() - t0
            append_run_log(run_id, f"done in {elapsed_time:.2f} seconds")

            append_run_log(run_id, "Creating zip file with input masks, output images, PNG previews, and MP4 previews. This can take some time for large files...")
            t0 = time.time()
            zip_path = zip_output_dir(out_dir)
            zip_size_bytes = os.path.getsize(zip_path)
            elapsed_time = time.time() - t0
            append_run_log(run_id, f"done in {elapsed_time:.2f} seconds")

            result = {
                "model_label": cfg["label"],
                "out_dir": out_dir,
                "input_masks_dir": input_masks_dir,
                "output_images_dir": output_images_dir,
                "preview_png_dir": preview_png_dir,
                "preview_mp4_dir": preview_mp4_dir,
                "preview_path": preview_png,
                "zip_path": zip_path,
                "zip_size_bytes": zip_size_bytes,
                "zip_size_human": human_readable_size(zip_size_bytes),
                "run_id": Path(out_dir).name,
                "n_images": n_images,
                "preview_name": Path(preview_png).name,
            }

            with RUNS_LOCK:
                RUNS[run_id]["status"] = "done"
                RUNS[run_id]["result"] = result
                RUNS[run_id]["updated_at"] = time.time()

            append_run_log(
                run_id,
                f"Zip created: {Path(zip_path).name} ({human_readable_size(zip_size_bytes)})",
            )
            append_run_log(run_id, "Run completed successfully")

        except Exception as e:
            err_text = f"{e}\n\n{traceback.format_exc()}"
            with RUNS_LOCK:
                if run_id in RUNS:
                    RUNS[run_id]["status"] = "error"
                    RUNS[run_id]["error"] = str(e)
                    RUNS[run_id]["updated_at"] = time.time()
            append_run_log(run_id, f"ERROR:\n{err_text}")


generator_app = GeneratorApp()

# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Synthetic Image Generator</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 1200px; }
    h1 { margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 28px; align-items: start; }
    .panel { border: 1px solid #ddd; border-radius: 10px; padding: 18px; margin-bottom: 20px; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    select, input[type=number] { width: 100%; padding: 10px; margin-top: 6px; box-sizing: border-box; }
    button, .download-btn {
      margin-top: 18px; padding: 12px 18px; border: 0; border-radius: 8px;
      background: #0b5fff; color: white; cursor: pointer; text-decoration: none; display: inline-block;
    }
    .download-btn { background: #198754; }
    .thumb { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
    .row { display: flex; gap: 18px; align-items: center; }
    .error { background: #ffe8e8; color: #8a1f1f; border: 1px solid #f1b5b5; padding: 12px; border-radius: 8px; margin-bottom: 18px; }
    .meta { white-space: pre-line; color: #333; }
    .muted { color: #666; font-size: 14px; }
    .status { margin-top: 12px; font-weight: 600; }
    .log-box {
        height: 420px;
        overflow: auto;
        background: #111;
        color: #eee;
        padding: 12px;
        border-radius: 8px;
        white-space: pre;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.4;
        border: 1px solid #333;
    }
  </style>
</head>
<body>
  <h1>Synthetic Image Generator</h1>

  {% if error %}
    <div class="error">{{ error }}</div>
  {% endif %}

  <div class="grid">
    <div>
      <div class="panel">
        <form method="post" action="/generate">
          <label for="model_key">Model checkpoint</label>
          <select name="model_key" id="model_key">
            {% for value, label in model_choices %}
              <option value="{{ value }}" {% if form_values.model_key == value %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>

          <label for="n_images">Number of images</label>
          <input
            type="number"
            id="n_images"
            name="n_images"
            min="{{ param_limits.n_images.min }}"
            max="{{ param_limits.n_images.max }}"
            step="1"
            value="{{ form_values.n_images }}"
          >
          <div class="muted">
            Number of images to generate (min: {{ param_limits.n_images.min }}, max: {{ param_limits.n_images.max }})
          </div>

          <label for="batch_size">Batch size</label>
          <input
            type="number"
            id="batch_size"
            name="batch_size"
            min="{{ param_limits.batch_size.min }}"
            max="{{ param_limits.batch_size.max }}"
            step="1"
            value="{{ form_values.batch_size }}"
          >
          <div class="muted">
            Number of images that will be generated simultaneously (min: {{ param_limits.batch_size.min }}, max: {{ param_limits.batch_size.max }})
          </div>

          <label for="inference_steps">Inference steps</label>
          <input
            type="number"
            id="inference_steps"
            name="inference_steps"
            min="{{ param_limits.inference_steps.min }}"
            max="{{ param_limits.inference_steps.max }}"
            step="1"
            value="{{ form_values.inference_steps }}"
          >
          <div class="muted">
            Number of generative steps, higher values provide higher quality outputs but require longer computation time (min: {{ param_limits.inference_steps.min }}, max: {{ param_limits.inference_steps.max }})
          </div>

          <button type="submit">Generate synthetic data</button>
        </form>
      </div>
    </div>

    <div>
      <div class="panel">
        {% if result %}
          <div class="row">
            <div style="min-width: 320px;">
              <img class="thumb" src="{{ result.preview_url }}" alt="Thumbnail">
            </div>
            <div>
              <a class="download-btn" href="{{ result.download_url }}">
                Download synthetic data ({{ result.zip_size_human }})
              </a>
              <div class="meta" style="margin-top: 16px;">
Model: {{ result.model_label }}
Generated {{ result.n_images }} image(s)
Preview: {{ result.preview_name }}
Zip: {{ result.zip_name }}
Zip size: {{ result.zip_size_human }}
Output folder: {{ result.out_dir }}
Input masks folder: {{ result.input_masks_dir }}
Output images folder: {{ result.output_images_dir }}
Preview PNG folder: {{ result.preview_png_dir }}
Preview MP4 folder: {{ result.preview_mp4_dir }}
              </div>
            </div>
          </div>
        {% else %}
          <div class="muted">No generation result yet.</div>
        {% endif %}
      </div>
    </div>
  </div>

  {% if run_id %}
  <div class="panel">
    <div class="status">Status: <span id="run-status">{{ run_status }}</span></div>
    <div class="muted" style="margin-top: 8px;">Command line output</div>
    <pre id="log-box" class="log-box">{{ logs }}</pre>
  </div>
  {% endif %}

{% if run_id and run_status == "running" %}
<script>
  async function pollRun(runId) {
    const logBox = document.getElementById("log-box");
    const statusEl = document.getElementById("run-status");

    while (true) {
      const res = await fetch(`/run_status/${runId}`);
      const data = await res.json();

      if (statusEl) {
        statusEl.textContent = data.status;
      }
      if (logBox) {
        logBox.textContent = data.logs || "";
        logBox.scrollTop = logBox.scrollHeight;
      }

      if (data.status === "done") {
        window.location.href = `/run/${runId}`;
        break;
      }

      if (data.status === "error") {
        if (statusEl) {
          statusEl.textContent = "error";
        }
        break;
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  pollRun("{{ run_id }}");
</script>
{% endif %}
</body>
</html>
"""


def default_form_values():
    return {
        "model_key": MODEL_CHOICES[0][0] if MODEL_CHOICES else "",
        "n_images": 1,
        "batch_size": 1,
        "inference_steps": 100,
    }


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE,
        model_choices=MODEL_CHOICES,
        result=None,
        error=None,
        form_values=default_form_values(),
        param_limits=PARAM_LIMITS,
        run_id=None,
        run_status=None,
        logs="",
    )


@app.route("/generate", methods=["POST"])
def generate():
    form_values = {
        "model_key": request.form.get("model_key", default_form_values()["model_key"]),
        "n_images": request.form.get("n_images", "4"),
        "batch_size": request.form.get("batch_size", "1"),
        "inference_steps": request.form.get("inference_steps", "100"),
    }

    try:
        run_id = generator_app.create_run(
            model_key=form_values["model_key"],
            n_images=int(form_values["n_images"]),
            batch_size=int(form_values["batch_size"]),
            inference_steps=int(form_values["inference_steps"]),
        )

        with RUNS_LOCK:
            run = RUNS[run_id]

        return render_template_string(
            HTML_TEMPLATE,
            model_choices=MODEL_CHOICES,
            result=None,
            error=None,
            form_values=form_values,
            param_limits=PARAM_LIMITS,
            run_id=run_id,
            run_status=run["status"],
            logs=run["logs"],
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            model_choices=MODEL_CHOICES,
            result=None,
            error=str(e),
            form_values=form_values,
            param_limits=PARAM_LIMITS,
            run_id=None,
            run_status=None,
            logs="",
        ), 400


@app.route("/run_status/<run_id>", methods=["GET"])
def run_status(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            abort(404)
        return {
            "status": run["status"],
            "logs": run["logs"],
            "error": run["error"],
        }


@app.route("/run/<run_id>", methods=["GET"])
def run_result(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            abort(404)

        result = run["result"]
        error = run["error"]
        status = run["status"]
        form_values = run["form_values"]
        logs = run["logs"]

    if result is not None:
        result = dict(result)
        result["preview_url"] = url_for(
            "preview_file",
            run_id=result["run_id"],
            filename=os.path.relpath(result["preview_path"], os.path.join(WEB_OUTPUTS_DIR, result["run_id"])),
        )
        result["download_url"] = url_for("download_zip", run_id=result["run_id"])
        result["zip_name"] = Path(result["zip_path"]).name

    return render_template_string(
        HTML_TEMPLATE,
        model_choices=MODEL_CHOICES,
        result=result,
        error=error,
        form_values=form_values,
        param_limits=PARAM_LIMITS,
        run_id=run_id,
        run_status=status,
        logs=logs,
    )


@app.route("/preview/<run_id>/<path:filename>", methods=["GET"])
def preview_file(run_id, filename):
    with RUNS_LOCK:
        if run_id not in RUNS:
            abort(404)
    return send_from_directory(os.path.join(WEB_OUTPUTS_DIR, run_id), filename)


@app.route("/download/<run_id>", methods=["GET"])
def download_zip(run_id):
    path = os.path.join(WEB_OUTPUTS_DIR, f"{run_id}_synthetic_data.zip")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{run_id}_synthetic_data.zip")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    global HOST, PORT, WEB_OUTPUTS_DIR, DEVICE, LATENTS_DIR, CREATE_VIDEOS
    parser = argparse.ArgumentParser(description="Launch the LAND 3D chest CT inference web app.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--model-path", "--model_path", default=CKPT_DIR)
    parser.add_argument("--vae-path", "--vae_path", default=os.environ.get("LAND_VAE_PATH"))
    parser.add_argument("--vae-mask-path", "--vae_mask_dir", default=os.environ.get("LAND_VAE_MASK_PATH"))
    parser.add_argument("--mask-dataset", "--mask_dataset", default=MASK_DATASET)
    parser.add_argument("--mask-mode", "--mask_mode", choices=["none", "nodule", "nodule+lung", "nodule+lung+texture"], default=os.environ.get("LAND_MASK_MODE", "nodule+lung"))
    parser.add_argument("--outputs-dir", default=WEB_OUTPUTS_DIR)
    parser.add_argument("--latents-dir", default=LATENTS_DIR)
    parser.add_argument("--device", default=os.environ.get("LAND_DEVICE", DEVICE))
    parser.add_argument("--no-videos", action="store_true", help="Skip MP4 previews.")
    parser.add_argument("--patchbased", action="store_true")
    args = parser.parse_args(argv)
    HOST, PORT, DEVICE = args.host, args.port, args.device
    WEB_OUTPUTS_DIR = str(Path(args.outputs_dir).expanduser().resolve())
    LATENTS_DIR, CREATE_VIDEOS = args.latents_dir, not args.no_videos
    cfg = MODEL_CONFIGS["ldm_nodule_lung"]
    cfg.update(model_path=str(Path(args.model_path).expanduser().resolve()),
               vae_path=args.vae_path, vae_mask_path=args.vae_mask_path,
               mask_dataset=args.mask_dataset if args.mask_mode != "none" else None,
               mask_mode=args.mask_mode, patchbased=args.patchbased,
               label=f"LDM conditioning: {args.mask_mode}")
    MODEL_CHOICES[:] = [(key, value["label"]) for key, value in MODEL_CONFIGS.items()]
    os.makedirs(WEB_OUTPUTS_DIR, exist_ok=True)
    device_type = "GPU" if torch.device(DEVICE).type == "cuda" else "CPU"
    print(f"Inference device: {device_type} ({DEVICE})", flush=True)

    ssl_context = None
    if SSL_CERT_FILE and SSL_KEY_FILE:
        ssl_context = (SSL_CERT_FILE, SSL_KEY_FILE)
        print(f"Starting HTTPS server on https://{HOST}:{PORT}")
    else:
        print(f"Starting HTTP server on http://{HOST}:{PORT}")
        print("To enable HTTPS, set SSL_CERT_FILE and SSL_KEY_FILE.")

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
