#!/usr/bin/env bash
# 云端 DPO 一键包 —— 本地 6GB 卡 OOM（见 paper_agent/reports/train_dpo_smoke.log），
# 这里把「换到一张 ≥24GB 的卡上就能跑」的全部前置与判据固定下来。
#
# 用法：
#   bash paper_agent/train/run_dpo_cloud.sh --dry-run          # 只打印将要执行的动作（本地可跑，无 GPU 也安全）
#   bash paper_agent/train/run_dpo_cloud.sh --smoke            # 10 步链路验证（云端）
#   bash paper_agent/train/run_dpo_cloud.sh                    # 正式 P2 DPO（云端，默认）
#   bash paper_agent/train/run_dpo_cloud.sh --min-vram-gb 40   # 抬高门槛（如要跑 bs=2）
#
# 设计取舍：
# - 不静默降级：显存不够、数据缺行、依赖装不上，一律硬失败并打印下一步该做什么。
# - 前置判据先跑再花钱：任何一步失败都在训练开跑之前退出，避免付费机上空转。
# - 环境按 reports/dpo_env_freeze.txt 重建（原 venv 在 /tmp，重启即丢）。
set -euo pipefail

MODE="p2"
DRY=0
MIN_VRAM_GB=24
WORKDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --smoke) MODE="smoke" ;;
    --min-vram-gb) shift; MIN_VRAM_GB="${1:?--min-vram-gb 需要数值}" ;;
    --workdir) shift; WORKDIR="${1:?--workdir 需要路径}" ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRAIN="$ROOT/paper_agent/train/train_dpo.py"
FREEZE="$ROOT/paper_agent/reports/dpo_env_freeze.txt"
DATA="$ROOT/paper_agent/data/dpo_p0v2.jsonl"

run() {
  if [ "$DRY" = "1" ]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi
}

echo "== 0. 路径自检 =="
for f in "$TRAIN" "$FREEZE" "$DATA"; do
  [ -f "$f" ] || { echo "缺文件：$f" >&2; exit 3; }
  echo "ok  $f"
done

echo "== 1. 显存判据（>= ${MIN_VRAM_GB} GB）=="
if command -v nvidia-smi >/dev/null 2>&1; then
  MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "检测到显存 ${MIB} MiB"
  if [ "$DRY" = "0" ] && [ "${MIB:-0}" -lt $((MIN_VRAM_GB * 1024)) ]; then
    echo "显存不足：需要 ${MIN_VRAM_GB} GB，实际 $((MIB / 1024)) GB。" >&2
    echo "本地 6GB 卡已实测 OOM（train_dpo_smoke.log：Tried to allocate 742 MiB，free 468 MiB）。" >&2
    echo "下一步：换 ≥${MIN_VRAM_GB}GB 的卡，或加 --min-vram-gb 降低门槛（会同时降 bs/grad_accum）。" >&2
    exit 4
  fi
else
  echo "没有 nvidia-smi：这台机器跑不了 DPO（本包只支持 CUDA 机型）" >&2
  [ "$DRY" = "1" ] || exit 5
fi

echo "== 2. 依赖（按冻结清单重建）=="
run python3 -m pip install -q -r "$FREEZE"

echo "== 3. 数据自检 =="
LINES=$(wc -l < "$DATA")
echo "偏好对行数：$LINES"
if [ "$DRY" = "0" ] && [ "$LINES" -lt 100 ]; then
  echo "偏好对太少（$LINES 行 < 100），DPO 会过拟合到个别样本；先补数据再花机时。" >&2
  exit 6
fi

echo "== 4. 起跑（mode=$MODE）=="
if [ "$MODE" = "smoke" ]; then
  run python3 "$TRAIN" --smoke
else
  run python3 "$TRAIN" --p2
fi

echo "== 5. 产物核对（训练脚本自己写 cfg_snapshot.json）=="
OUT="$ROOT/paper_agent/outputs/tmlr-qwen3-4b-p2"
if [ "$MODE" = "smoke" ]; then OUT="$ROOT/paper_agent/outputs/tmlr-qwen3-4b-p2smoke"; fi
run test -d "$OUT"
echo "产物目录：$OUT"
echo "完成。下一步：用 cost-per-correct 给这一档定价（正确率来源写清是离线评测还是人工），"
echo "并把 adapter 以 vLLM --enable-lora 挂起来跑 ttftkit 三闸（见 ttftkit/docs/FINDINGS.md F7）。"
