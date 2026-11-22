#!/usr/bin/env bash
set -euo pipefail
cd "$HOME"
grep -qm1 '^vendor_id[[:space:]]*:[[:space:]]*GenuineIntel$' /proc/cpuinfo
export DEBIAN_FRONTEND=noninteractive
sudo cloud-init status --wait >/dev/null || test "$?" -eq 2
sudo fallocate -l 12G /var/tmp/spt-swap
sudo chmod 600 /var/tmp/spt-swap
sudo mkswap /var/tmp/spt-swap
sudo swapon /var/tmp/spt-swap
sudo apt-get -q update
sudo env DEBIAN_FRONTEND=noninteractive apt-get -y -q -o DPkg::Lock::Timeout=120 install git python3-venv ffmpeg
python3 -m venv "$HOME/tools"
"$HOME/tools/bin/pip" install --disable-pip-version-check uv==0.13.4
export PATH="$HOME/tools/bin:$PATH"
git clone -q https://github.com/jasonjiang8866/spleeterpt.git source
git -C source checkout -q e63c8987fdf18a30c8d85785f1d680e08bc9430d
git clone -q https://github.com/deezer/spleeter.git upstream
git -C upstream checkout -q c8854001ac8acad34a9bc2bd15f28475541828b1
cd source
export TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
uv sync --locked --python 3.12.16 --group dev --group release
.venv/bin/python scripts/prepare_five_stem.py --work "$HOME/weights"
.venv/bin/python scripts/verify_native_five_stem.py --weights "$HOME/weights" \
  --upstream-root "$HOME/upstream" --output "$HOME/native.json"
