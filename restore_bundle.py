#!/usr/bin/env python3
"""Rebuild stock_data_bundle.npz from its base64 parts in bundle/.

The bundle is a binary file (42.5 MB). The snapshot push used for this
repository carries text only, so the bundle travels as base64 text
parts. Run this script after cloning; it concatenates the parts,
decodes them, verifies the SHA-256 of the rebuilt file, and writes
stock_data_bundle.npz next to this script (the location the stock
config expects).

    python3 restore_bundle.py
"""
import base64
import glob
import hashlib
import os
import sys

EXPECTED_SHA256 = "ed500a0b327e757ea366d5e6ee3478128740f53c5c348565b576ad0ce9f48810"
EXPECTED_SIZE = 42521504


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    part_paths = sorted(glob.glob(os.path.join(here, "bundle", "stock_data_bundle.b64.part*")))
    if not part_paths:
        sys.exit("no bundle parts found under bundle/; run this from the repository root checkout")
    text = "".join(open(p).read().strip() for p in part_paths)
    raw = base64.b64decode(text, validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != EXPECTED_SIZE or digest != EXPECTED_SHA256:
        sys.exit(f"verification FAILED: size {len(raw)} (expected {EXPECTED_SIZE}), sha256 {digest}")
    out = os.path.join(here, "stock_data_bundle.npz")
    with open(out, "wb") as f:
        f.write(raw)
    print(f"OK: wrote {out} ({len(raw)} bytes, sha256 verified, {len(part_paths)} parts)")


if __name__ == "__main__":
    main()
