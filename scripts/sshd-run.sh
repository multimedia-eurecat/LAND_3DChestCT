#!/bin/sh
set -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
[ ! -r /usr/local/lib/land-runtime-env.sh ] || . /usr/local/lib/land-runtime-env.sh

if [ "${ENABLE_SSH:-0}" != "1" ]; then
    echo "[info] SSH disabled (ENABLE_SSH != 1)"
    exec sleep infinity
fi

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

if [ -z "${SSH_PUBLIC_KEY:-}" ] && [ -z "${SSH_KEY:-}" ]; then
    echo "[error] ENABLE_SSH=1 requires SSH_PUBLIC_KEY or SSH_KEY" >&2
    exit 1
fi

{
    [ -z "${SSH_PUBLIC_KEY:-}" ] || printf '%s\n' "$SSH_PUBLIC_KEY"
    [ -z "${SSH_KEY:-}" ] || printf '%s\n' "$SSH_KEY"
} > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

ssh-keygen -A
exec /usr/sbin/sshd -D -e \
    -o AuthenticationMethods=publickey \
    -o KbdInteractiveAuthentication=no \
    -o PasswordAuthentication=no \
    -o PermitRootLogin=prohibit-password \
    -o PubkeyAuthentication=yes
