# -*- coding: utf-8 -*-
"""DPO-v2: 与dpo_p0同400篇论文, 偏好对升级为实质段落(多模板轮换).
chosen: 带证据绑定的实质修改(>=200字); rejected: 去证据+绝对化表述.
诚实标签: 合成偏好(规则构造), 非真人标注; 仅作P2冷启动, 真人偏好后续升级.
"""
import json

import pandas as pd

CORPUS = "/data3/Research/corpus_hf/all-papers.parquet"
OUT = "paper_agent/data/dpo_p0v2.jsonl"
SRC = "paper_agent/data/dpo_p0.jsonl"
FLOOR = 200

CHO = [
    "修改后: 本文方法基于摘要所述方案(证据{pid}), 转述止于原文, 不发挥; "
    "局限一: 摘要未报告失败模式(属推断, 待验证); 局限二: 缺同venue对照(待检索补齐)。"
    "复现边界: 种子与配置以原文为准, 缺失处显式标注, 不脑补默认值。无证据的推测已删除。",
    "修改后: 保留有证据支撑的两点——方法要点与报告结果(证据{pid}); "
    "将“显著提升”降级为“摘要报告的数值差异, 显著性待检验”; 新增可证伪假设句, 跑完实验前不做结论。"
    "适用范围: 结论仅限原文数据集与指标, 外推需重新立假设并验证, 不得直接泛化。",
    "修改后: 每句实证陈述后附证据编号(证据{pid}); 动机解释标注为笔者推测; "
    "基线清单只列本文(已证实), 其余标待补齐, 不虚构对照数字。删去所有无来源形容词。"
    "数字一致性: 段中样本量与种子数与原文对账, 对不上宁可删除该句, 不留约数蒙混。",
]
REJ = [
    "修改后: 该方法显然成立, 效果显著优于现有方案, 得到广泛认同。 (无证据绑定)",
    "修改后: 结论A毋庸置疑; 结论B具有普遍意义, 可直接推广到所有场景。 (无证据绑定)",
    "修改后: 实验充分证明了优越性, 审稿人一定会接受。 (无证据绑定)",
]


def S(x):
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def main():
    df = pd.read_parquet(CORPUS, columns=["paper_id", "title", "abstract"])
    meta = {}
    for _, r in df.iterrows():
        meta[S(r["paper_id"])] = (S(r["title"]), S(r["abstract"])[:500])
    n = 0
    with open(SRC, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as fo:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            pid = json.loads(line)["paper_id"]
            title, abs_ = meta.get(pid, (pid, ""))
            prompt = "修改以下段落, 保留证据绑定, 区分假设与证据。对象: [%s] %s。摘要节选: %s" % (pid, title, abs_[:300])
            chosen = CHO[i % 3].format(pid=pid) + " 摘要依据: %s… 转述边界止于此。" % abs_[:120]
            assert len(chosen) >= FLOOR, (i, len(chosen))
            fo.write(json.dumps({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": REJ[i % 3],
                "paper_id": pid,
                "synthetic": "dpo-v2-multitemplate",
            }, ensure_ascii=False) + "\n")
            n += 1
    print("[dpo-v2] pairs=%d" % n)


if __name__ == "__main__":
    main()
