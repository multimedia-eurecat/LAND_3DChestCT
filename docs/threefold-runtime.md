# LAND 3DChestCT: TFGrid Full VM runtime

This document describes how to rebuild and publish the LAND 3D inference app
as a TFGrid Full VM flist. The Full VM contains its own Ubuntu kernel,
initramfs, kernel modules, NVIDIA driver, and systemd services. Use
`Dockerfile.fullvm` for GPU deployment.

## Target provider and GPU

This flist is prepared specifically for the following deployment profile:

- IaaS provider: `iaas-fujitsu-gpu-1`;
- GPU selector: `0000:01:00.0/10de/2b85`;
- GPU: NVIDIA GB202 (GeForce RTX 5090), PCI vendor/device ID `10de:2b85`;
- default guest driver: `nvidia-driver-580-open`;
- NVIDIA UVM HMM disabled for CUDA compatibility in the Full VM;
- CUDA userspace: PyTorch `2.7.0+cu128` (CUDA 12.8).

The GPU setup script is not hard-locked to this PCI device: it accepts any
NVIDIA GPU (`10de:*`). The default driver and tested deployment assumptions are
specific to the profile above, however. For another NVIDIA GPU or provider,
verify driver compatibility and override `NVIDIA_DRIVER_PACKAGE` when needed.
The Full VM kernel is included in this flist and is required for GPU inference;
it includes the kernel support needed for loadable NVIDIA modules.
The exact GPU exposed to a running VM can be checked with:

```bash
lspci -nnk | grep -A3 -i nvidia
```

## Runtime layout

The image is built with `Dockerfile.fullvm`. TFGrid detects the kernel under
`/boot` and boots systemd. `/boot/vmlinuz` and `/boot/initrd.img` must be
regular files, not symbolic links; the build script validates this. The
workload entrypoint must be empty.

systemd starts five LAND services:

- `land-runtime-env` mounts `/dev/vda` at `/mnt/zos` and exposes workload
  variables from `zosrc`;
- `land-gpu-setup` attempts to configure and validate the NVIDIA GPU, then allows
  LAND to fall back to CPU if the GPU is absent or initialization fails;
- `land-app` starts `src/inference_ldm_app.py`;
- `land-nginx` serves public HTTP after `land-app`;
- `land-sshd` provides optional key-only SSH access.

The HTTP path is:

```text
public port 80 -> nginx -> Flask on 127.0.0.1:7860
```

The nginx configuration is in `nginx/default.conf`.

The image contains the checkpoint and the ten conditioning masks. The build
context must contain:

```text
data/ckpts/2025-09-10_17-51-07_256_bsz1_lr1e-5_nodule+lung_mask/
data/masks/
```

## After code changes

Run the versioned build script from the repository root:

```bash
chmod +x scripts/build-fullvm-flist.sh
./scripts/build-fullvm-flist.sh v8
```

The script validates the kernel-bearing rootfs and creates:

```text
land-3dchestct-fullvm-v8.tar.gz
```

Upload that archive to the Software Repository. Do not use `docker save`. The
Hub converts the root filesystem archive into the `.flist` used by deployment.

## Deployment settings

Set the workload entrypoint to an empty string:

```text
""
```

Expose public HTTP port `80`. The application itself remains on internal port
`7860` and should not be used as the public port when nginx is enabled.

For GPU inference, the workload must provide an NVIDIA GPU. The image contains
the CUDA userspace libraries supplied by the PyTorch CUDA wheel, its own kernel,
matching modules and headers, and the open NVIDIA driver. At first boot,
`land-gpu-setup` loads and validates the driver. If the GPU is absent or cannot
be initialized, setup logs a warning and LAND starts on CPU.

Runtime paths can be overridden with these environment variables:

- `LAND_MODEL_PATH` - checkpoint directory;
- `LAND_MASK_DATASET` - conditioning mask dataset;
- `LAND_MASK_MODE` - for example `nodule+lung`;
- `LAND_OUTPUTS_DIR` - output directory, default `/opt/app/web_outputs`;
- `LAND_DEVICE` - optional explicit device, for example `cuda` or `cuda:1`;
  leave unset to select CUDA when available and otherwise fall back to CPU;
- `NVIDIA_DRIVER_PACKAGE` - open NVIDIA driver package, default
  `nvidia-driver-580-open`;
- `LAND_PORT` - internal Flask port, default `7860`.

Set `LAND_OUTPUTS_DIR` to a writable mounted disk path when generated files
must survive VM replacement.

## SSH

The image includes an SSH service that is disabled by default. An empty
entrypoint does not enable SSH. To enable it,
make the VM reachable over the selected TFGrid network and configure these
Software Repository runtime variables:

```text
ENABLE_SSH=1
SSH_PUBLIC_KEY=ssh-ed25519 AAAA... user@host
```

Declare `ENABLE_SSH` as a static value or a user input with default `1`.
Declare `SSH_PUBLIC_KEY` as a required user input and paste the contents of the
user's `.pub` file, never the private key. Do not declare `SSH_KEY`: the Hub may
inject it automatically as the operator public key.

The startup script writes every non-empty key to `/root/.ssh/authorized_keys`:

- `SSH_PUBLIC_KEY` authorizes the deploying user;
- `SSH_KEY` authorizes the platform operator when the Hub injects it.

Both keys work independently and simultaneously. SSH allows the `root` account
with public-key authentication only; password and keyboard-interactive login
are explicitly disabled. The service exits with an error when SSH is enabled
without either key.

Connect with the private key matching `SSH_PUBLIC_KEY`:

```bash
ssh -i ~/.ssh/id_ed25519 root@<VM_IP>
```

For an IPv6 address, use:

```bash
ssh -i ~/.ssh/id_ed25519 root@<VM_IPV6>
```

## Troubleshooting

Check the services inside the VM:

```bash
systemctl --no-pager --full status \
  land-runtime-env land-gpu-setup land-app land-nginx land-sshd
journalctl -b -u land-gpu-setup -u land-app --no-pager
ss -lntp | grep -E ':(22|80|7860)'
```

If nginx returns `502`, the `land-app` service is not listening on port 7860
or is still loading the model. Check `journalctl -u land-app` first.

Warnings from `land-gpu-setup` do not stop the application; LAND falls back to CPU.
To restore GPU inference when no NVIDIA PCI device is visible, fix GPU
assignment on the provider or deployment side; rebuilding the flist cannot
expose missing hardware. Verify the packaged Full VM kernel and driver with
`uname -r`, `/proc/modules`, `nvidia-smi`, and `torch.cuda.is_available()`.
