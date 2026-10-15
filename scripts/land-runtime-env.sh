#!/bin/sh

if [ -r /mnt/zos/zosrc ]; then
    set -a
    . /mnt/zos/zosrc
    set +a
fi
