#!/usr/bin/env bash
set -euo pipefail

version="${1:-v8}"
archive="land-3dchestct-fullvm-$version.tar.gz"
partial_archive="$archive.partial"
archive_listing="$(mktemp -t "land-fullvm-$version-listing-XXXXXX")"
archive_details="$(mktemp -t "land-fullvm-$version-details-XXXXXX")"

cleanup() {
    rm -f "$partial_archive" "$archive_listing" "$archive_details"
}
trap cleanup EXIT INT TERM

docker buildx build \
    --progress plain \
    --platform linux/amd64 \
    -f Dockerfile.fullvm \
    --output type=tar,dest=- \
    . | gzip -1 > "$partial_archive"

mv "$partial_archive" "$archive"
tar -tzf "$archive" > "$archive_listing"
tar -tvzf "$archive" > "$archive_details"

grep -Eq '^-[[:alnum:]-]{9}[[:space:]].*[[:space:]](\./)?boot/vmlinuz$' "$archive_details"
grep -Eq '^-[[:alnum:]-]{9}[[:space:]].*[[:space:]](\./)?boot/initrd\.img$' "$archive_details"
grep -Eq '^(\./)?(usr/)?lib/modules/' "$archive_listing"
grep -Eq '^(\./)?etc/systemd/system/land-app.service$' "$archive_listing"

sha256sum "$archive"
ls -lh "$archive"
