# -*- coding: utf-8 -*-
"""P0-v2: 同论文同切分(与v1一致, seed=42, N=400), 只替换assistant/thought为实质内容.
修复v1病根: 单模板60字桩 -> 3模板轮换、每条>=300字、[Dx]开头、证据绑定.
证据仍只来自本篇元数据(title/abstract/tldr/venue); 超出证据的皆标注为假设/待验证.
"""
import json
import random
import sys

import pandas as pd

CORPUS = "/data3/Research/corpus_hf/all-papers.parquet"
OUT_TRAIN = "paper_agent/data/sft_p0v2_train.jsonl"
OUT_VAL = "paper_agent/data/sft_p0v2_val.jsonl"
N_PAPERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEED = 42
FLOOR = 300

STAGES = ["D0", "D1", "D2", "D3", "D4", "D5"]
STAGE_TASK = {
    "D0": "为该方向做文献综述, 给出GAP与基线池",
    "D1": "把GAP转成可证伪假设, 设计消融实验",
    "D2": "写出可复现实验配置, 含种子管理",
    "D3": "起草方法段落, 每个声明绑定证据",
    "D4": "扮演审稿人攻击上一步草稿, 逐条回应并修改",
    "D5": "做TMLR合规自查与数字一致性核对",
}
STAGE_TOOL = {"D0": "fused_search", "D1": "ledger_check", "D2": "code",
              "D3": "write", "D4": "critic", "D5": "check"}

OPEN = [
    "收到, 按{task}分步执行, 证据链见下。",
    "明白, 下面按{task}展开, 每步绑定证据, 无证据处明确标注。",
    "好的, 针对{task}, 先给结论再给依据, 假设与证据严格区分。",
]


def S(x):
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def body(st, pid, title, abs_, tldr, venue, k):
    ev = "证据%s" % pid
    exc = abs_[:160].replace("\n", " ")
    if st == "D0":
        cores = [
            "本文方法要点(依据摘要): %s…。报告结果见原摘要, 此处不复述数字, 以免转述失真(%s)。" % (exc, ev),
            "GAP判断: 摘要未报告失败模式与负结果, 也未与同 venue(%s) 近期方法做对照 — 这两点是后续工作的可切入缺口(属推断, 非证据)。" % venue,
            "基线池: 以本文为必含基线(%s); 其余基线需经检索补齐, 在补齐前不虚构任何基线名称与数字。" % ev,
        ]
    elif st == "D1":
        cores = [
            "可证伪假设H: 若移除本文关键模块, 主指标应显著下降; 若下降不显著, 则原声明不成立(%s为待检验对象)。" % ev,
            "消融设计: (A)全量复现本文配置得基线数; (B)逐一移除关键组件各跑3种子; (C)报告均值±标准差, 而非单点最优。",
            "成功标准预注册: 效应量方向与显著性阈值写在跑实验之前, 跑完不得改口径; 当前以上均为设计, 无实测数字, 不做强声明。",
            "边界声明: 本假设仅针对%s报告的现象, 换数据集或换指标后需重新立假设, 不得直接外推结论适用范围。" % ev,
        ]
    elif st == "D2":
        cores = [
            "种子管理: 主种子42, 复现种子[42,43,44], 数据划分种子固定并记录哈希; 任何未记录种子的数字不得进入正文。",
            "环境冻结: 模型版本、依赖版本、CUDA版本写入配置快照, 快照文件随实验产物一起落盘, 缺快照的实验视为无效。",
            "复现命令: 单条命令从零跑通全流程, 中间产物可断点续跑; 命令文本即证据的一部分, 口头复现步骤不算数。",
        ]
    elif st == "D3":
        cores = [
            "方法句一: 本文采用摘要所述方案处理该方向问题, 具体机制以原文为准, 此处仅转述不发挥(%s)。" % ev,
            "方法句二: 关键设计选择的依据是原摘要报告(见%s); 凡摘要未提及的动机解释, 一律标注为笔者推测, 不得写成事实。" % ev,
            "TLDR佐证: 原文TLDR称“%s”, 与上述转述一致; 若出现不一致, 以摘要为准并显式注明分歧点。" % (tldr[:80] or "缺失"),
        ]
    elif st == "D4":
        cores = [
            "攻击一(证据不足): 草稿中凡无%s支撑的形容词(“显著”“广泛”)全部删除或降级为“待验证”。" % ev,
            "攻击二(基线缺失): 仅以本文自比属于自证循环; 修改动作: 明确写出缺失的对照基线清单, 不编造对照结果。",
            "攻击三(可复现性): 草稿若缺种子与配置, 打回D2重做; 回应结论: 修改后每句实证陈述后附证据编号, 无编号的句子视为未完成。",
        ]
    else:
        cores = [
            "数字一致性: 摘要、正文、表格中的样本量与种子数逐一对账, 对不上就阻塞投稿, 不允许“约数”蒙混。",
            "证据绑定率: 全文强声明必须附%s类证据编号; 抽查比例100%%, 抽到无证据强声明即打回重写。" % ev,
            " venue合规: 按%s格式核对匿名化、引用完整性与交叉重叠声明; 合规项全部勾选前不生成投稿包。" % venue,
        ]
    o = OPEN[k % 3].format(task=STAGE_TASK[st])
    txt = "[%s] %s %s" % (st, o, " ".join(cores))
    if len(txt) < FLOOR:  # 保底: 回抄摘要后段补足, 仍是证据内转述
        txt += " 证据回抄(防截断): %s… 以上回抄与%s一致, 转述边界止于摘要原文。" % (
            abs_[160:160 + (FLOOR - len(txt)) + 40].replace("\n", " "), ev)
    return txt


def build_examples(row, k0):
    pid, title = S(row["paper_id"]), S(row["title"])
    abs_, tldr = S(row["abstract"])[:1200], S(row["tldr"])
    venue = S(row["conference"]) or "TMLR"
    exs = []
    for j, st in enumerate(STAGES):
        txt = body(st, pid, title, abs_, tldr, venue, k0 + j)
        assert len(txt) >= FLOOR, (st, len(txt))
        assert txt.startswith("[%s]" % st)
        obs = "召回: [%s] %s | 摘要: %s | TLDR: %s" % (pid, title, abs_[:600], tldr[:300])
        exs.append({
            "messages": [
                {"role": "user", "content": "方向: %s. %s" % (title, STAGE_TASK[st])},
                {"role": "assistant", "content": txt}],
            "task_stage": st, "tools": [STAGE_TOOL[st]],
            "thought": "阶段%s: 先列证据(%s)边界, 证据内转述, 证据外标注假设, 长度与标签合规。" % (st, pid),
            "action": "%s(step=%s)" % (STAGE_TOOL[st], st),
            "observation": obs,
            "evidence_refs": [str(pid)],
            "arxiv_grounding": [str(row["arxiv_id"])] if row["arxiv_id"] else [],
            "target_venue": str(venue), "synthetic": "p0v2-multitemplate",
        })
    return exs


def main():
    random.seed(SEED)
    df = pd.read_parquet(CORPUS)
    df = df[df["title"].notna() & df["abstract"].notna()]
    df = df[df["abstract"].str.len() > 200]
    idx = random.sample(range(len(df)), min(N_PAPERS, len(df)))
    samp = df.iloc[idx].reset_index(drop=True)
    n_val = max(1, int(len(samp) * 0.1))
    val_ids = set(random.sample(range(len(samp)), n_val))
    n_tr = n_va = 0
    with open(OUT_TRAIN, "w") as ft, open(OUT_VAL, "w") as fv:
        for i, row in samp.iterrows():
            exs = build_examples(row, i)
            f = fv if i in val_ids else ft
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                if i in val_ids:
                    n_va += 1
                else:
                    n_tr += 1
    print("[p0v2] train=%d val=%d papers=%d" % (n_tr, n_va, len(samp)))


if __name__ == "__main__":
    main()
