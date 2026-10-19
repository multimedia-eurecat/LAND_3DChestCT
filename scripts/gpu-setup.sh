#!/bin/sh
set -eu

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
[ ! -r /usr/local/lib/land-runtime-env.sh ] || . /usr/local/lib/land-runtime-env.sh

verify_cuda() {
    python_bin="${LAND_PYTHON:-/opt/venv/bin/python3}"
    if [ ! -x "$python_bin" ]; then
        python_bin=/usr/bin/python3
    fi
    modprobe nvidia >/dev/null 2>&1 || return 1
    modprobe nvidia_uvm >/dev/null 2>&1 || return 1
    nvidia-smi >/dev/null 2>&1 || return 1
    "$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize CUDA")

torch.empty(1, device="cuda")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY
}

if ! lspci -Dn | grep -qi '^.* 10de:'; then
    echo "WARNING: No NVIDIA PCI device is visible; LAND will use CPU" >&2
    exit 0
fi

if [ ! -e /proc/modules ]; then
    echo "WARNING: VM kernel $(uname -r) cannot load modules; NVIDIA driver installation is not possible and LAND will use CPU" >&2
    exit 0
fi

if verify_cuda; then
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
driver_package="${NVIDIA_DRIVER_PACKAGE:-nvidia-driver-580-open}"

if ! apt-get update; then
    echo "WARNING: Failed to update APT metadata; LAND will use CPU" >&2
    exit 0
fi

if ! apt-get install -y --no-install-recommends \
    "linux-headers-$(uname -r)" \
    "$driver_package"; then
    echo "WARNING: Failed to install the NVIDIA driver; LAND will use CPU" >&2
    exit 0
fi

for attempt in $(seq 1 30); do
    if verify_cuda; then
        exit 0
    fi
    sleep 2
done

echo "WARNING: CUDA validation failed after waiting for the NVIDIA driver; LAND will use CPU" >&2

exit 0
