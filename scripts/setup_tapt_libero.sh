#!/bin/bash
set -euo pipefail
mkdir -p /workspace/tapt/evidence
cd /workspace/tapt
apt-get update -qq
apt-get install -y -qq libegl1 libgl1-mesa-glx libosmesa6-dev libglew-dev libglfw3-dev ffmpeg > evidence/apt.log 2>&1
pip install -q uv
git clone https://github.com/cxliu0314/openpi.git author
git -C author checkout f4eb160ba52b22c1e85fe432de59c24bbbac6187
git -C author submodule update --init --depth 1 third_party/libero
git clone https://github.com/Physical-Intelligence/openpi.git upstream
git -C upstream checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git -C upstream rev-parse HEAD > evidence/upstream-commit.txt
cd author
UV_CACHE_DIR=/workspace/tapt/uv-cache uv sync --frozen --no-dev > ../evidence/policy-install.log 2>&1
cd /workspace/tapt
uv pip install --python author/.venv/bin/python pytest==9.1.1
uv venv --python 3.8 sim-env
uv pip sync --python sim-env/bin/python author/examples/libero/requirements.txt author/third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match > evidence/sim-install.log 2>&1
uv pip install --python sim-env/bin/python -e author/packages/openpi-client -e author/third_party/libero >> evidence/sim-install.log 2>&1
mkdir -p /root/.libero /usr/share/glvnd/egl_vendor.d
cat > /root/.libero/config.yaml <<EOF
benchmark_root: /workspace/tapt/author/third_party/libero/libero/libero
bddl_files: /workspace/tapt/author/third_party/libero/libero/libero/bddl_files
init_states: /workspace/tapt/author/third_party/libero/libero/libero/init_files
datasets: /workspace/tapt/data
assets: /workspace/tapt/author/third_party/libero/libero/libero/assets
EOF
# NVIDIA container images already provide a vendor file, sometimes read-only.
test -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json
date -u > evidence/setup-ready.txt
