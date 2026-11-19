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
# Kaggle 的「账号 slug」与「API 用户名」是**两个不同的东西**：kaggle.json 里
# username=liuzewen，但所有产物都落在 liuzewen123 下。dataset_sources 写错时
# Kaggle 只打一行 "not valid dataset sources" 却照样把 kernel 提交成功。
KAGGLE_ACCOUNT="${KAGGLE_ACCOUNT:-liuzewen123}"
FULL_DS="$KAGGLE_ACCOUNT/$DS_SLUG"
FULL_KERNEL="$KAGGLE_ACCOUNT/$KERNEL_SLUG"
MODE="dry"

# fail-closed：占位符没换就上传，Kaggle 会把 kernel 建到不存在的用户名下（或者更糟，
# 撞上别人的 slug）。上传前先拒。
if grep -q "PLACEHOLDER_USERNAME" "$STAGE/kernel/kernel-metadata.json" 2>/dev/null; then
  echo "kernel-metadata.json 仍含 PLACEHOLDER_USERNAME —— 先把用户名填成真实值" >&2
  exit 4
fi

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
{"title": "dsh dpo p0v2", "id": "$FULL_DS",
 "licenses": [{"name": "other"}]}
JSON
    # 「已存在」不是错误：新建会报 title in use 且**退出码仍是 0**，`create || version`
    # 这条经典写法因此静默不更新（2026-11-19 实测：kernel 真跑了旧脚本，130 秒后
    # 还是同一个 ValueError）。显式先问存在性。
    if kaggle datasets files "$FULL_DS" >/dev/null 2>&1; then
      echo "数据集已存在 → 发新版本"
      kaggle datasets version -p "$STAGE_DIR" -m "update $(date +%F-%H%M)"
    else
      kaggle datasets create -p "$STAGE_DIR" --dir-mode zip
    fi
    # 回读校验：不看退出码，看字节数对不对得上（静默未更新是这套工具链的常态）
    LOCAL_BYTES=$(wc -c < "$ROOT/paper_agent/train/train_dpo.py")
    for _ in $(seq 1 30); do
      REMOTE_BYTES=$(kaggle datasets files "$FULL_DS" 2>/dev/null | awk '$1=="train_dpo.py"{print $2}')
      [ "${REMOTE_BYTES:-0}" = "$LOCAL_BYTES" ] && break
      sleep 5
    done
    if [ "${REMOTE_BYTES:-0}" != "$LOCAL_BYTES" ]; then
      echo "❌ 数据集里的 train_dpo.py 不是本地这一版（远端 ${REMOTE_BYTES:-?} B vs 本地 ${LOCAL_BYTES} B）" >&2
      exit 6
    fi
    echo "✓ 数据集已更新（train_dpo.py ${LOCAL_BYTES} B）"
    echo "== 2. kernel =="
    PUSH_LOG="$(mktemp)"
    kaggle kernels push -p "$STAGE/kernel" 2>&1 | tee "$PUSH_LOG"
    # 「账号 slug ≠ API 用户名」是 Kaggle 的常态（本次实测：config 里 username=liuzewen，
    # 但所有产物都落在 liuzewen123 下）。dataset_sources 写错时 Kaggle 只打一行
    # "not valid dataset sources"，kernel 照样「成功」提交 —— 然后在云端因为没有
    # /kaggle/input 直接失败。所以这里必须自己判红。
    if grep -q "not valid dataset sources" "$PUSH_LOG"; then
      echo "❌ kernel 的 dataset_sources 无效（账号 slug 与 API 用户名不一致）" >&2
      echo "   用 kaggle datasets list -m 或页面 URL 读出真实 slug，写进 kernel-metadata.json" >&2
      rm -f "$PUSH_LOG"; exit 5
    fi
    rm -f "$PUSH_LOG"
    echo "已提交。用 --status 看进度，--fetch 拉产物。"
    ;;
  status)
    kaggle kernels status "$FULL_KERNEL"
    ;;
  fetch)
    OUT="$ROOT/paper_agent/outputs/dpo-kaggle"
    mkdir -p "$OUT"
    kaggle kernels output "$FULL_KERNEL" -p "$OUT"
    echo "产物落在 $OUT"
    ;;
esac
