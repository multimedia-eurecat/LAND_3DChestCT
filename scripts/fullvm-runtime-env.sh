#!/bin/sh
set -eu

mount_dir=/mnt/zos
mkdir -p "$mount_dir"

if mountpoint -q "$mount_dir"; then
    exit 0
fi

if [ ! -b /dev/vda ]; then
    echo "WARNING: ThreeFold runtime environment device /dev/vda is unavailable" >&2
    exit 0
fi

mount -o ro /dev/vda "$mount_dir"

if [ ! -r "$mount_dir/zosrc" ]; then
    echo "WARNING: ThreeFold runtime environment file /mnt/zos/zosrc is unavailable" >&2
fi
