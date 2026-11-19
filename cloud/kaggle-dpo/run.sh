#!/usr/bin/env bash
# Kaggle 通道：把 DPO 推到云端 GPU 跑（本地 6GB 已实测 OOM）
#
# 背景（2026-11-19 现测）：
#   kaggle quota → GPU 剩 11.19h / 30h（2026-11-21 刷新）；T4 16GB 可跑 4B QLoRA DPO
#   （bs=1、max_len=384、ref_model=None 共享基座，脚本原本就是这套省显存配置）。
#   其它通道实测：ecs-hz 2核1GB 无 GPU、hpccube 2核251GB 无 GPU（sinfo 无分区）、
#   ecs-us 连不上、j1gpu 被对端断开 → 云端只剩 Kaggle 这一条能真跑。
#
# 用法：
#   bash paper_agent/cloud/kaggle-dpo/run.sh --dry-run   # 只打印要上传什么、跑什么（不花额度）
#   bash paper_agent/cloud/kaggle-dpo/run.sh --push      # 真上传数据集 + 提交 kernel（开始花 GPU 额度）
#   bash paper_agent/cloud/kaggle-dpo/run.sh --status     # 看 kernel 状态
#   bash paper_agent/cloud/kaggle-dpo/run.sh --fetch      # 拉回产物到 paper_agent/outputs/dpo-kaggle/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAGE="$ROOT/paper_agent/cloud/kaggle-dpo"
DS_SLUG="dsh-dpo-p0v2"
KERNEL_SLUG="dsh-dpo-train"
MODE="dry"

case "${1:---dry-run}" in
  --dry-run) MODE="dry" ;;
  --push) MODE="push" ;;
  --status) MODE="status" ;;
  --fetch) MODE="fetch" ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  *) echo "未知参数：$1" >&2; exit 2 ;;
esac

echo "== 将要上传的数据集内容 =="
for f in data/dpo_p0v2.jsonl train/train_dpo.py reports/dpo_env_freeze.txt; do
  [ -f "$ROOT/paper_agent/$f" ] || { echo "缺文件：paper_agent/$f" >&2; exit 3; }
  printf '  %-32s %s 行/字节\n' "$f" "$(wc -l < "$ROOT/paper_agent/$f")"
done

echo "== kernel 配置 =="
echo "  dataset: $DS_SLUG   kernel: $KERNEL_SLUG   GPU: T4 x1 (16GB)   预计墙钟 1–2h"

case "$MODE" in
  dry)
    echo "[dry-run] kaggle datasets create -p $STAGE/dataset"
    echo "[dry-run] kaggle kernels push -p $STAGE/kernel"
    echo "（真跑请加 --push；会消耗 Kaggle GPU 额度）"
    ;;
  push)
    echo "== 1. 数据集 =="
    STAGE_DIR="$STAGE/dataset"
    mkdir -p "$STAGE_DIR"
    cp "$ROOT/paper_agent/data/dpo_p0v2.jsonl" "$STAGE_DIR/"
    cp "$ROOT/paper_agent/train/train_dpo.py" "$STAGE_DIR/"
    cp "$ROOT/paper_agent/reports/dpo_env_freeze.txt" "$STAGE_DIR/"
    cat > "$STAGE_DIR/dataset-metadata.json" <<JSON
{"title": "dsh dpo p0v2", "id": "$(kaggle config view 2>/dev/null | awk -F': *' '/username/{print $2}')/$DS_SLUG",
 "licenses": [{"name": "other"}]}
JSON
    kaggle datasets create -p "$STAGE_DIR" --dir-mode zip || \
      kaggle datasets version -p "$STAGE_DIR" -m "update $(date +%F-%H%M)"
    echo "== 2. kernel =="
    kaggle kernels push -p "$STAGE/kernel"
    echo "已提交。用 --status 看进度，--fetch 拉产物。"
    ;;
  status)
    kaggle kernels status "$(kaggle config view 2>/dev/null | awk -F': *' '/username/{print $2}')/$KERNEL_SLUG"
    ;;
  fetch)
    OUT="$ROOT/paper_agent/outputs/dpo-kaggle"
    mkdir -p "$OUT"
    kaggle kernels output "$(kaggle config view 2>/dev/null | awk -F': *' '/username/{print $2}')/$KERNEL_SLUG" -p "$OUT"
    echo "产物落在 $OUT"
    ;;
esac
