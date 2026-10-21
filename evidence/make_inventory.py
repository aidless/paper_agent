#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 evidence/data_manifest.sha256 生成 ASSET_INVENTORY.md。

只读。复用价值按类别赋值, 关键条目单独覆盖。
用法: python3 paper_agent/evidence/make_inventory.py
"""
import os
from datetime import datetime, timezone

PA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(PA, "evidence")
OUT = os.path.join(PA, "ASSET_INVENTORY.md")


def load_rows():
    rows = []
    p = os.path.join(EV, "data_manifest.sha256")
    for line in open(p, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        h, s, sc, path = line.rstrip("\n").split("  ", 3)
        rows.append((h, int(s), sc, path))
    return rows


def mib(n):
    return "%.1f MiB" % (n / 1048576.0)


# ---- 复用价值: 类别默认值 + 逐项覆盖 ----
CAT_RULES = [
    ("outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/", "A-交付",
     "**唯一正式交付物**。LoRA r32/BF16, 504 张量。详见 DECISION_20261008.md 的三项复算警告。"),
    ("outputs/", "B-备选", "备选/事故存档。含 optimizer.pt(252MiB)等续训状态, 仅在需要复现或续训时复用。"),
    ("data/", "C-数据", "训练/评测语料。构造脚本与 jsonl 同目录, 可重跑。"),
    ("train/", "D-代码", "训练脚本。含 P2 续训事故(wrongbase)的正确姿势。"),
    ("eval/", "D-代码", "评测脚本。注意 --val 默认值与 p1v2 实际训练 val 不同(见 DECISION)。"),
    ("infer/", "D-代码", "推理脚本。"),
    ("retrieval/", "D-代码", "检索代码。四路融合, 最可复用的一块工程资产。"),
    ("reports/", "E-证据", "评测/训练/门禁报告。结论均可回溯到对应 json 与日志。"),
]

OVERRIDE = {
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/adapter_model.safetensors":
        "A-交付 | **交付权重本体**。实测 132,188,392 B = 126.1 MiB(BF16)。"
        "DELIVERY.md 写的『252MB』实为 p1v2 root 的 F32 文件大小, 与本文件无关。",
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/optimizer.pt":
        "B-备选 | 优化器状态 252.5 MiB。仅续训需要, 定档后可从分发包剔除。",
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/rng_state.pth":
        "B-备选 | 续训 RNG 状态。复现 bit-level 续训才需要。",
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/trainer_state.json":
        "A-交付 | **0.4469 的唯一出处**。best_metric=0.44691@step136, 4 个评测点中最低。",
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/adapter_config.json":
        "A-交付 | LoRA 超参(r32/alpha16/7 target modules)。加载必需。",
    "outputs/tmlr-qwen3-4b-p1v2/adapter_model.safetensors":
        "B-备选 | **与 checkpoint-136 逐比特相同**(实测 max abs diff = 0.000e+00), 只是 F32。"
        "评测里被当成第二个独立臂, 是『三方对比实为两方』的根因。",
    "outputs/tmlr-qwen3-4b-p1v2/checkpoint-270/adapter_model.safetensors":
        "B-备选 | p1v2 终点。eval_loss 0.4834 > 0.4469, 是过拟合侧的真对照臂。",
    "outputs/tmlr-qwen3-4b-p2_wrongbase/":
        "B-备选 | **事故存档**。从基座从零起步(缺 --from-adapter), val 3.56。保留作教训证据。",
    "data/dpo_p0v2.jsonl":
        "C-数据 | **400 对, 但 rejected 只有 3 个互异串**, chosen(min 272字) 与 rejected(max 44字) "
        "长度完全可分。真实 DPO 信号 = 0, 不能当偏好数据用。",
    "data/dpo_p0.jsonl":
        "C-数据 | 400 对, rejected 只有 **1** 个互异串。同上, 更退化。",
    "data/sft_p0v2_train.jsonl":
        "C-数据 | p1v2 实际训练集 2160 条, 6 阶段各 360。**decontam 报告未覆盖此文件**。",
    "data/sft_p0v2_val.jsonl":
        "C-数据 | p1v2 验证集 240 条。0.4469 就是在这上面算的。",
    "data/sft_p0_val.jsonl":
        "C-数据 | eval_3way.py 的**默认值**验证集。与 p0v2_val 同 40 个论文签名但回复不同。",
    "data/eval_fixed.jsonl":
        "C-数据 | 8 条检索评测(标题→pred_ok)。decontam 报告声称覆盖的是它。",
    "data/decontam_report.md":
        "E-证据 | **覆盖范围错误**: 声称训练集 = sft_fullprocess.jsonl(5条), "
        "而实际训练集是 2160 条的 sft_p0v2_train.jsonl。",
    "reports/dpo_env_freeze.txt":
        "E-证据 | **不可信**: 登记 8 个版本里 6 个与 venv 实测不符(torch 记 2.14.0 实为 2.5.1+cu121)。",
    "reports/eval_3way_v2.json":
        "E-证据 | ckpt136 与 ckpt_final 两键规范化后 sha256 相同(177db3828ad785ee), 非独立两臂。",
    "reports/eval_3way_v2.md":
        "E-证据 | 三方对比表。因 root≡ckpt-136, 实为两方。",
    "reports/P1_VERDICT.md":
        "E-证据 | P1 门禁史。文中『权重md5不同』的推断经复算不成立。",
    "reports/P2_VERDICT.md":
        "E-证据 | P2 无增益结论 + 6GB 卡四次 OOM 记录(硬件成本估算的依据)。",
    "reports/compliance.md":
        "E-证据 | 合规链自述。承认六扫脚本尚未绑定。",
    "requirements.txt":
        "F-文档 | **未钉版且被违反**: numpy<2.0, 实测 2.5.3。",
    "DELIVERY.md":
        "F-文档 | 交付说明。其『TMLR 级别不成立』的自述经本次复算**确认**, 另有两处数字不准。",
    "README.md":
        "F-文档 | 运行说明。venv 路径与实测一致。",
}


def classify(path):
    for pref, cat, note in CAT_RULES:
        if path.startswith(pref):
            return cat, note
    return "Z-其他", "—"


def reuse(path, scope):
    if scope == "external":
        return ("X-外部依赖",
                "工作区外, 训练/检索依赖。已对账 sha256, 换机复现必须先核对。")
    if path in OVERRIDE:
        return OVERRIDE[path].split(" | ", 1)
    return classify(path)


def main():
    rows = load_rows()
    internal = [r for r in rows if r[2] == "internal"]
    external = [r for r in rows if r[2] == "external"]

    order = ["A-交付", "B-备选", "C-数据", "D-代码", "E-证据", "F-文档",
             "X-外部依赖", "Z-其他"]
    buckets = {}
    for h, s, sc, p in rows:
        cat, note = reuse(p, sc)
        buckets.setdefault(cat, []).append((h, s, sc, p, note))

    total_bytes = sum(s for _, s, _, _ in internal)
    ext_bytes = sum(s for _, s, sc, _ in external if sc == "external" and s)

    L = []
    L.append("# ASSET_INVENTORY — paper_agent 资产清单（定档冻结用）\n")
    L.append("> 生成方式：`python3 paper_agent/evidence/make_inventory.py`"
             "（数据源 `paper_agent/evidence/data_manifest.sha256`，只读）  ")
    L.append("> 生成时间（UTC）：%s  " % datetime.now(timezone.utc).isoformat())
    L.append("> 复验方式：`python3 paper_agent/evidence/make_manifest.py --check`"
             "（内容改动/文件缺失均 exit≠0，已做三条反证）\n")
    L.append("**零丢失承诺**：本清单为**只读**生成，未删除、未移动、未改写任何原始产物。"
             "全部 %d 个内部条目 + %d 个外部依赖已逐项记录 sha256 与字节数。\n"
             % (len(internal), len(external)))
    L.append("| 统计 | 值 |")
    L.append("|---|---|")
    L.append("| 内部条目数 | %d |" % len(internal))
    L.append("| 外部依赖条目数 | %d |" % len(external))
    L.append("| 内部总体积 | %.2f GiB |" % (total_bytes / 1073741824.0))
    L.append("| 外部依赖总体积 | %.2f GiB |" % (ext_bytes / 1073741824.0))
    L.append("| 缺失条目 | %d |" % sum(1 for r in rows if r[0] == "MISSING"))
    L.append("")
    L.append("**复用价值分级**：A-交付（正式对外件）/ B-备选（存档，续训或取证用）/"
             "C-数据 / D-代码 / E-证据 / F-文档 / X-外部依赖。\n")

    for cat in order:
        items = buckets.get(cat)
        if not items:
            continue
        sub = sum(s for _, s, _, _, _ in items)
        L.append("---\n")
        L.append("## %s（%d 项，%.2f MiB）\n" % (cat, len(items), sub / 1048576.0))
        L.append("| sha256 | 大小 | 路径 | 复用价值 |")
        L.append("|---|---|---|---|")
        for h, s, sc, p, note in items:
            pdisp = p
            if sc == "external":
                pdisp = "`%s`" % p.split("  # ")[0]
            else:
                pdisp = "`%s`" % p
            L.append("| `%s` | %s | %s | %s |"
                     % (h, mib(s) if s >= 1048576 else "%d B" % s, pdisp, note))
        L.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[inventory] %d internal + %d external -> %s"
          % (len(internal), len(external), OUT))


if __name__ == "__main__":
    main()