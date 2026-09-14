#!/usr/bin/env bash
set -euo pipefail
# Linux NVIDIA host. GPU driver must expose compute,utility,graphics to containers.
export BVI_WORKSPACE="${BVI_WORKSPACE:-/workspace/bvi}"
mkdir -p "$BVI_WORKSPACE"
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/bvi-runtime}"
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
vulkaninfo --summary
cd "$BVI_WORKSPACE"
for spec in 'mshab https://github.com/arth-shukla/mshab.git e9ff3d23496d38e4431c8d913e147ffa007f7f72' 'ManiSkill https://github.com/haosulab/ManiSkill.git 17121e3f96e3ee3ed0c03610b17f8bc2864617af'; do
  read -r name url commit <<< "$spec"
  if [ ! -d "$name/.git" ]; then
    git init "$name"
    git -C "$name" remote add origin "$url"
    git -C "$name" fetch --depth 1 origin "$commit"
    git -C "$name" checkout --detach FETCH_HEAD
  fi
  test "$(git -C "$name" rev-parse HEAD)" = "$commit"
done
if [ ! -d venv ]; then python3 -m venv --system-site-packages venv; fi
venv/bin/python -m pip install --no-cache-dir -e ManiSkill -e 'mshab[train]'
venv/bin/python -m pip freeze > requirements-observed.txt
cat > activate.sh <<'ACTIVATE'
export BVI_WORKSPACE="${BVI_WORKSPACE:-/workspace/bvi}"
source "$BVI_WORKSPACE/venv/bin/activate"
export MS_ASSET_DIR="$BVI_WORKSPACE/assets"
export MSHAB_CHECKPOINT_DIR="$BVI_WORKSPACE/mshab_checkpoints"
export SAPIEN_NO_DISPLAY=1
export XDG_RUNTIME_DIR=/tmp/bvi-runtime
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ACTIVATE
source activate.sh
python -c 'import torch, sapien, mshab.envs; assert torch.cuda.is_available(); print("BVI_IMPORT_OK",torch.__version__,sapien.__version__)'
