# LAND 3D inference app

Run commands from the repository root. The app serves the generation form,
live progress logs, PNG previews, and a ZIP download at http://localhost:7860.
Each run includes generated CT volumes and conditioning masks as NumPy files,
nodule-centered CT/mask/overlay PNGs, and orthogonal MP4 videos.

## Run directly

Use the existing environment without installing another copy:

```bash
/media/diskA/roger/conda_envs/land_env/bin/python src/inference_ldm_app.py \
  --model-path ./data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask \
  --mask-dataset ./data/masks \
  --mask-mode nodule+lung \
  --outputs-dir ./web_outputs
```

The script also works from another working directory, using its absolute path,
or as `python -m src.inference_ldm_app`. Use `--help` to list options.
Omitting `--model-path` and `--mask-dataset` uses the checkpoint and masks above,
resolved relative to the repository directory regardless of the current working
directory. The repository includes the first 10 conditioning masks from the
original dataset. Requests exceeding 10 volumes reuse those masks cyclically
and display a warning asking users to contact the software authors for more
conditioning masks.

The checkpoint directory must contain `unet/`, `scheduler/`, and `vae/`.
An optional `vae_mask/` is loaded automatically. If the VAEs are stored separately,
pass `--vae-path /path/to/ct_vae` and `--vae-mask-path /path/to/mask_vae`.
These paths refer directly to folders containing `config.json` and weights.
The architectures and mask mode must match the trained checkpoint.

The conditioning dataset uses `PATIENT/mask/*.npy` files below its root;
`chest_ct/` files are not required. The app uses sorted mask order and cycles
back to the first mask when the requested sample count exceeds the available
masks. Each output still receives a separate diffusion noise sample. For unconditional checkpoints,
use `--mask-mode none`. Use `--no-videos` to reduce preview time and disk space.
Generation defaults to CUDA when available; override with `--device cpu` or
`--device cuda:1`. Full 256³ volume inference is intended for a GPU.

## Build and run Docker

The host needs Docker, NVIDIA drivers, and NVIDIA Container Toolkit for GPU
execution. The image uses `python:3.10-slim-bookworm` and installs PyTorch 2.6
with CUDA 12.4 plus the inference dependencies. CUDA runtime libraries and cuDNN
come from PyTorch's pip dependencies, avoiding a second copy in an NVIDIA CUDA
base image. The host still supplies the NVIDIA driver through Container Toolkit.
The default checkpoint (including its CT and mask
VAEs and scheduler) and the 10 conditioning masks are copied into the image.
Build from a repository containing these files under `data/ckpts/` and
`data/masks/`; no checkpoint or dataset mounts are needed at deployment time.

```bash
docker build -t land-3dchestct:latest .
docker run --rm --gpus all -p 7860:7860 land-3dchestct:latest
```

To preserve generated files after the container exits, optionally mount an
output directory:

```bash
mkdir -p web_outputs
docker run --rm --gpus all -p 7860:7860 \
  -v "$(pwd)/web_outputs:/opt/app/web_outputs" \
  land-3dchestct:latest
```

The image defaults to `python3 src/inference_ldm_app.py`, following the 2D
repository's Python entrypoint convention. To pass app options, include the
script name:

```bash
docker run --rm land-3dchestct:latest src/inference_ldm_app.py --help
docker run --rm --gpus all land-3dchestct:latest \
  -c 'import torch; print(torch.cuda.is_available())'
```

The build checks that the app imports and that PyTorch has CUDA 12.4 support.
To verify GPU computation after rebuilding:

```bash
docker run --rm --gpus all land-3dchestct:latest \
  -c 'import torch; assert torch.cuda.is_available(); x = torch.ones(4, device="cuda"); print(torch.cuda.get_device_name()); print((x + x).cpu())'
```

For a software hub requiring an explicit entry point, use
`/usr/local/bin/python3 /opt/app/src/inference_ldm_app.py`. The previous
`/usr/bin/python3` path remains available as a compatibility symlink.

The image includes all inference inputs for this checkpoint and requires no
model downloads at runtime. To use other checkpoints, separately stored VAEs,
or additional masks, mount their directories read-only and supply the
corresponding app options.

To deploy on a different machine, push the built image to your container
registry, or transfer it as an archive:

```bash
# On the build machine:
docker save -o land-3dchestct.tar land-3dchestct:latest
# Transfer land-3dchestct.tar to the deployment machine, then run there:
docker load -i land-3dchestct.tar
docker run --rm --gpus all -p 7860:7860 land-3dchestct:latest
```

Configuration can also use `LAND_MODEL_PATH`, `LAND_VAE_PATH`,
`LAND_VAE_MASK_PATH`, `LAND_MASK_DATASET`, `LAND_MASK_MODE`, `LAND_OUTPUTS_DIR`,
`LAND_LATENTS_DIR`, `LAND_DEVICE`, `LAND_HOST`, and `LAND_PORT`. Command-line
options take precedence. Optional HTTPS uses `SSL_CERT_FILE` and `SSL_KEY_FILE`;
mount those files when running in Docker.

The server keeps jobs and loaded models in process memory and serializes model
generation. Run one app process so polling requests see the same job state.
Restarting the app clears the job registry; generated files remain on disk.

## Remove or rebuild the image

To remove the local image:

```bash
docker image rm land-3dchestct:latest
```

If a container still uses the image, identify it first:

```bash
docker ps -a --filter ancestor=land-3dchestct:latest
```

Stop a running container and remove it using its ID or name, then retry image
removal:

```bash
docker stop <container_id_or_name>
docker rm <container_id_or_name>
docker image rm land-3dchestct:latest
```

Removing a container deletes outputs stored only inside that container. Outputs
in the optional host bind mount remain on the host. Removing the local image
does not remove an uploaded image or an existing software hub deployment;
manage those through the hub separately.

To apply code changes, rebuilding with the same tag is sufficient; deleting the
image first is optional:

```bash
docker build -t land-3dchestct:latest .
```

Docker normally reuses unchanged build layers. To rebuild every layer from
scratch, for example after changing dependency installation or when diagnosing
a stale build cache, use `--no-cache`:

```bash
docker build --no-cache -t land-3dchestct:latest .
```

Existing containers continue using the previous image. Start a new container
or replace the hub deployment with the rebuilt image to apply the changes.
Prefix Docker commands with `sudo` if your account cannot access the daemon.

## Verification

Run regression tests using the supplied environment:

```bash
/media/diskA/roger/conda_envs/land_env/bin/python -B -m unittest discover -s tests -v
```
