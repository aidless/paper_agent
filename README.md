# 论文全流程 Agent（Qwen3-4B QLoRA + peS2o/arXiv 双库）

这是一个把「文献检索 → 论文草稿 → 评测」串成一条线的科研 agent 训练与评测管线。
核心资产是一个 Qwen3-4B 的 4bit NF4 + LoRA adapter，用于在论文写作任务上做
指令微调（SFT），并对检索到的事实做证据绑定。项目**当前定档为技术报告**，
不宣称 TMLR 级可用（见 `DELIVERY.md` / `DECISION_20261008.md` 的诚实自声明）。

---

## 这个项目解决什么问题

写论文（尤其是文献综述 / GAP 与基线分析）时，LLM 容易「编造引用、不绑定证据、
跳过任务阶段」。本项目想训练一个 agent 模型，让它：

1. **按阶段走完论文流程**（D0 文献综述 → D1 假设 → D2 配置 → … → D5 草稿）。
2. **每个数字都绑定证据**（`evidence_refs` / `arxiv_grounding` 字段强制存在）。
3. **从真实语料检索而非编造**：双库（peS2o 基座 54 万篇 + arXiv 增量库 1.7 万篇）。

## 为什么重要

- **证据绑定是结构性的**：训练数据（`build_sft.py`）强制每个样本带
  `evidence_refs` + `arxiv_grounding`，模型学到的不是「写漂亮话」而是「写有据
  可查的话」。
- **检索双库**：peS2o 基座（54 万篇，只读）+ arXiv 增量（每日 `--fetch`），
  用 `fuse.py` 四路融合（向量增量 + 关键词增量 + 向量基座 + 关键词基座）。
- **门禁不撒谎**：P1 曾因「训练 loss 降但验证 loss 四连升」判 **FAIL**（过拟合
  信号），P2 因「合成优选微调无增益」再判 **FAIL**——项目**不因为跑了训练就
  宣布成功**。

## 我用了什么方法

- **训练**：Qwen3-4B 基座，4bit NF4 量化 + LoRA（r=32），PEFT + TRL。
- **数据**：P0 模板合成轨迹（`build_sft.py` / `build_p0v2.py`）+ 400 对偏好
  （`build_dpo_v2.py`，后因 6GB 卡 OOM 改走 ReST 式单轮优选监督）。
- **评测**：60 条验证集同一掩码 NLL + 8 条贪心生成，三方对比
  （base / ckpt-68 / ckpt-136），见 `eval/eval_3way.py` + `reports/eval_3way_v2.*`。
- **检索**：`fuse.py` 四路融合 + `recency_boost` 时鲜旋钮（默认 0，诚实声明为
  拍脑袋常数，非学得权重）。

## 结果如何（实测数字，全部来自仓库落盘文件）

**最优模型 = ckpt-136**（`outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/`），
来自 `reports/eval_3way_v2.json`（逐样本明细在 `.json`）：

| 模型 | 60 条 val NLL | 阶段标签率 | 证据绑定率 | 平均长度 |
|---|---|---|---|---|
| base Qwen3-4B | **3.9718** | 0.00 | 0.125 | — |
| ckpt-136 | **0.3879** | 0.00 | **1.0（8/8）** | 418 |

- 验证集 NLL 从 3.9718 → 0.3879（约 **10 倍降低**），证据绑定率 0.125 → 1.0。
- 阶段标签率 0.00：判据走**内容相符**而非字面标签（`eval_3way_v2.md` 说明）。
- P1 门禁（`reports/P1_VERDICT.md`）：base 4.4904 → ckpt-68 1.1030，但验证 loss
  0.72→1.02 四连升 → **FAIL（过拟合）**。
- P2 门禁（`reports/P2_VERDICT.md`）：ckpt-136 验证 0.4469 维持交付物；E5 对决
  8 战全平（b=0 c=0 p=1.0）→ **FAIL（无增益）**。
- 检索融合（`reports/RETRIEVAL_UPGRADE.md`）：查询 "LLM agent evaluation" top-10
  里 9 条来自向量通道、1 条关键词；增量 17,263 条向量与基座 541,744 条同空间
  （MiniLM-384）。

## Fail-closed 声明

本项目评测与门禁**默认 fail-closed**：训练 loss 下降但验证 loss 上升（过拟合信号）
**不会**被当作成功——P1 门禁因验证 loss 四连升判 **FAIL**，P2 因「合成优选微调无增益」
再判 **FAIL**（`reports/P1_VERDICT.md` / `P2_VERDICT.md`）。评测数字缺失、来源无法
落盘、或复算不一致时，对应结论**不发布**，宁可判 FAIL 也不给出无依据的「成功」结论。
「跑了训练」本身不构成成功证据，唯一成功判据是验证集上的可复算指标。

## 边界（证明什么 / 不证明什么）

**证明什么：**
- 管线能在**合成数据**上把 Qwen3-4B 微调到「证据绑定率 8/8、验证 NLL 降到
  0.3879」——这是**合成分布内**的拟合能力，有 `eval_3way_v2.json` 逐样本明细。
- 检索双库能在真实语料上跑通四路融合，且缺索引时向量通道**优雅降级为空、
  不抛错**（`fuse.py`）。

**不证明什么（诚实边界）：**
- **不证明 TMLR 级可用**：缺真 DPO（6GB 卡四次 OOM）、缺真人偏好标注、缺
  括号阶段标签。`DELIVERY.md` 明确「TMLR 级别可用：不成立」。
- **不证明真实写作能力**：训练数据是 P0 模板合成轨迹，规律性强；P1 已实测
  「模型学到模板捷径而非写作能力」（阶段标签率 0.00、多条 15 字截断输出）。
- **不证明合成分布之外的泛化**：P2 显示合成数据天花板已现——SFT-v2 把证据
  绑定打到 8/8 后，继续在合成分布上打转不再涨分。
- **不证明 `recency_boost` 是最优**：默认 0，该常数未经标注集验证，README 与
  `RETRIEVAL_UPGRADE.md` 均声明「不得写成最优」。
- **不证明无泄漏**：数据去重/防污染未做完整证明（`data/decontam_report.md`
  仅记录了去污检查，未覆盖全部训练集）。
- **单 benchmark 评测**：只用 60 条验证 NLL + 8 条生成，未做多 benchmark
  统计检验（`DECISION_20261008.md` 自陈缺口）。

## 快速开始（安装 / 运行）

```bash
# 1. 环境（venv 在 /data1/paper_agent_venv，工作目录含冒号无法直接 venv）
/data1/paper_agent_venv/bin/pip install -r paper_agent/requirements.txt
/data1/paper_agent_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. 数据构建
python3 paper_agent/data/build_sft.py

# 3. 增量库拉取
python3 arxiv_incremental.py --stats
python3 arxiv_incremental.py --fetch --days 7 --max-results 100 --query "LLM agent evaluation"

# 4. 检索（四路融合）
python3 paper_agent/retrieval/fuse.py "LLM agent evaluation"

# 5. 评测 / 推理
python3 paper_agent/eval/eval_agent.py
python3 paper_agent/infer/infer_agent.py --prompt "LLM agent evaluation GAP与基线"

# 6. 训练（先 smoke 再全量）
/data1/paper_agent_venv/bin/python paper_agent/train/train_agent.py --smoke
/data1/paper_agent_venv/bin/python paper_agent/train/train_agent.py
```

## 测试与复现

本项目**没有正式测试目录**（研究管线，非库）。可复现性靠两类证据：

**1. 静态校验（全部可重跑，不依赖 GPU/大模型）：**

```bash
# 一键冒烟测试（tests/test_smoke.py：静态解析 + 检索降级 + 评测数字一致性，4 项）
/data1/paper_agent_venv/bin/python -m unittest tests.test_smoke -v

# 全部 .py 语法解析 + 全部 .json 可解析（14 py + 58 json，实测 0 失败）
/data1/paper_agent_venv/bin/python -c "
import ast, json, glob
for f in glob.glob('**/*.py', recursive=True):
    ast.parse(open(f, encoding='utf-8').read())
for f in glob.glob('**/*.json', recursive=True):
    json.load(open(f, encoding='utf-8'))
print('all parse OK')"

# 检索冒烟（强制向量通道降级 → 纯关键词回退，验证优雅降级）
/data1/paper_agent_venv/bin/python -c "
import sys; sys.path.insert(0, 'retrieval')
from fuse import fused_search
import fuse
fuse.search_pes2o_vector.cache = None
fuse.search_incr_vector.cache = None
fuse._st_model.cache = None
print(len(fused_search('LLM agent evaluation', top_k=3)), 'results (keyword fallback)')"

# 评测 JSON 数字一致性（防手抄漂移）
/data1/paper_agent_venv/bin/python -c "
import json
d = json.load(open('reports/eval_3way_v2.json'))
assert abs(d['base']['val_nll_60']-3.9718)<1e-4
assert abs(d['ckpt136']['val_nll_60']-0.3879)<1e-4
assert d['ckpt136']['gen']['evd']==1.0
print('eval numbers OK')"
```

**2. 数字溯源（每个报告数字的落盘来源 + 重算命令）：**

| 数字 | 落盘来源 | 重算 |
|---|---|---|
| base 3.9718 / ckpt-136 0.3879 | `reports/eval_3way_v2.json` | `eval/eval_3way.py`（60 条 val 同一掩码） |
| 证据绑定 8/8 | `reports/eval_3way_v2.json` 的 `gen.evd` | 同上（8 条贪心生成） |
| 验证 0.4469 | `reports/P2_VERDICT.md` | `train_agent.py` 训练日志 |
| 17,263 增量 / 541,744 基座 | `RETRIEVAL_UPGRADE.md` + `fuse.py:541744` | `build_incr_index.py` / `kb_health` |
| 4.4904→1.1030（P1） | `reports/P1_VERDICT.md` | `eval_3way.py`（ckpt-68） |

> 训练全量需 GPU（6GB 分片，4bit NF4）；本机复现只验证静态/检索/评测数字，
> 不重训模型（重训需云端 GPU，见 `DECISION_20261008.md`）。

## 目录结构

```
data/             # 数据构建脚本 + 训练/评测集（sft/dpo jsonl）
retrieval/        # fuse.py 四路融合 + build_incr_index.py 增量索引
train/            # train_agent.py（SFT）+ train_dpo.py（DPO，OOM 后封存）
eval/             # eval_3way.py 三方评测 + eval_agent.py + gen_probe.py
infer/            # infer_agent.py 推理
tests/            # tests/test_smoke.py 冒烟测试（静态解析 + 检索降级 + 数字一致）
outputs/          # 训练输出（ckpt-10/68/136/270 + p1/p1v2/p2 系列）
reports/          # 评测 JSON/MD + P1/P2 门禁裁决 + 检索升级记录
evidence/         # 环境指纹 + manifest sha256
DELIVERY.md       # 交付包（诚实自声明：TMLR 级不成立）
DECISION_20261008.md  # 定档：技术报告（封存 + 可复现证据包）
ASSET_INVENTORY.md    # 资产清单（逐项 sha256 + 复用价值）
```

## 参考

- `DELIVERY.md`：交付包与未达项诚实声明
- `DECISION_20261008.md`：定档决策（技术报告 / 升级到一区的最小路径）
- `ASSET_INVENTORY.md`：资产清单（sha256 逐项）
- `evidence/env_fingerprint.json`：环境指纹

## 一键运行（需 GPU）

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 data/build_sft.py
python3 retrieval/build_incr_index.py
python3 retrieval/fuse.py "LLM agent evaluation"
python3 eval/eval_agent.py
python3 infer/infer_agent.py --prompt "LLM agent evaluation GAP与基线"
.venv/bin/python train/train_agent.py --smoke
.venv/bin/python train/train_agent.py
```

## 约定
- 基座语料 `$PE_S2O_KB_FAISS`（外部只读 FAISS 索引，541745 篇；不随仓库分发，需自行准备）
- 增量语料目录 `$ARXIV_INCREMENTAL`，独立于基座，每日 `--fetch` 增量更新
- 训练输出 `outputs/tmlr-qwen3-4b-agent` + `cfg_snapshot.json`
- TMLR: claim ledger必填, Abstract最后写, G1/G3/G4/G9/G10硬线

## 仓库结构

```
paper_agent/
├── data/       # SFT/DPO 数据构建与产物
│   │             build_sft.py / build_p0v2.py / build_dpo_v2.py → sft_*.jsonl, dpo_*.jsonl
├── retrieval/  # 双库检索融合：peS2o 基座（外部只读 FAISS）+ arXiv 增量
│   │             build_incr_index.py（建/更新增量索引）、fuse.py（融合查询）
├── train/      # QLoRA 训练：train_agent.py（SFT，含 --smoke）、train_dpo.py（DPO）
├── infer/      # 推理入口：infer_agent.py
├── eval/       # 三方对比评估：eval_3way.py / eval_agent.py / gen_probe.py
├── reports/    # 结论与判定：P1_VERDICT.md / P2_VERDICT.md / eval_3way*.json+md
└── evidence/   # 证据链：verify_tracked.py（免 GPU 离线校验）、MANIFEST.sha256、
                  env_fingerprint.json、make_manifest.py / make_inventory.py
```

数据流向：`data/`（构建语料）→ `retrieval/`（检索融合）→ `train/`（SFT→DPO）→
`infer/` + `eval/`（推理与三方评估）→ `reports/`（判定落盘）→ `evidence/`（全链哈希校验）。

## Reproduce offline

```bash

# Fresh-clone check: tracked files, committed JSON, and evidence hashes
python3 evidence/verify_tracked.py
```

The fresh-clone verifier deliberately does not validate
`evidence/data_manifest.sha256`. That dated snapshot includes local model
checkpoints and mutable external `/data1` corpora that are not redistributed.
On the original machine, `python3 evidence/make_manifest.py --check` can audit
that historical inventory, but missing or changed external files should be
reported as drift rather than silently replacing the snapshot.

## 验收标准与复算

验收判据：本仓判为合格的条件是 `python3 eval/run_cases.py` 通过率必须为 100%（20/20），且 `make eval` 退出码必须为 0；任何一条失败即判为不合格。题库里每条 case 的 `expected` 都是从本仓真实 API / 落盘数据现算出来的值，不是抄文档——改了实现或数据而不同步改期望值，这一节就会判红。

`eval/cases.jsonl` 的 20 条 case 按 axis 分工（七键 `id/axis/question/expected/mode/verify/source`，全部是 `expr`，真调本仓函数或真解析本仓数据文件）：

| axis | 条数 | 测什么 | 代表 case |
|---|---|---|---|
| retrieval/tokens | 3 | `retrieval/fuse.py` 的 `_tokens` 分词、短词回退、4 token 截断 | PA-01/02/03 |
| retrieval/fusion | 3 | `fused_search` 四路候选去重、按分数降序取 top_k、`recency_boost` 加分重排 | PA-04/05/06 |
| data/sft | 3 | `data/sft_p0v2_val.jsonl` 条数与阶段分布、`data/sft_fullprocess.jsonl` 证据绑定字段 | PA-07/08/09 |
| data/dpo | 2 | `data/dpo_p0v2.jsonl` 对数与 chosen 最短长度（构建地板 200） | PA-10/11 |
| reports/eval | 3 | `reports/eval_3way_v2.json` 的 NLL、由 `gen_detail` 逐条重算证据绑定率与平均长度 | PA-12/13/14 |
| data/fixture | 2 | `data/eval_fixed.jsonl` 条数与 `pred_ok` 计数 | PA-15/16 |
| evidence/manifest | 2 | `evidence/MANIFEST.sha256` 哈希实测复算、记录行数 | PA-17/18 |
| build/consts | 2 | `data/build_sft.py` 的 SEEDS 阶段序列、`data/build_p0v2.py` 的 FLOOR | PA-19/20 |

三条复算命令（在仓库根目录跑，不需要 GPU / 大模型 / 训练产物）：

```bash
python3 eval/run_cases.py              # 20/20 才算通过
python3 eval/run_cases.py --selftest   # 19/20 且 exit 0：第 1 条期望值被篡改成不可能值，必须变红
python3 eval/run_cases.py --save       # 另把本次结果落盘到 eval/run-report.json
```

复算说明：

- `eval/run_cases.py` 与题库分离：脚本只做「执行 + 比较 + 归因」，期望值全部写在 `eval/cases.jsonl` 的 `expected` 字段，出处见同条 `source`（`path:line`）。
- `make eval` = `python3 eval/run_cases.py` + `python3 -m pytest tests -q`；前者先跑，任何一条 case 失败就中断，随后才是 `tests/` 那 4 条静态解析 / 检索降级 / 评测数字一致性测试。
- `--selftest` 证明这套题不是装饰：它把 PA-01 的期望值换成 `__selftest_impossible_value__`，必须出现 1 条失败，否则脚本自己 exit 3。
- 结果文件 `eval/run-report.json` 由 `--save` 生成，含 `total/passed/failed/failed_by_axis/results`，与 `python3 eval/run_cases.py --json` 的输出逐字相同；要复算某个数字，重跑对应 case 的 `verify` 代码即可。
- PA-20 在 import `data/build_p0v2.py` 前把 `sys.argv` 收缩成只含脚本名——该模块顶层用 `sys.argv[1]` 当 `N_PAPERS`，不隔离就会把 `--selftest` 当整数解析。

失败归因（按 axis 汇总，不是只报一个 pass_rate）：`--selftest` 的实测输出长这样——

```
失败归因（按 axis）：
  retrieval/tokens: 1 条 —— PA-01
SELFTEST: 期望变红；若上面仍全绿则这套题无效
```

机器可读版本在 `eval/run-report.json` 的 `failed_by_axis` 字段（axis → 失败 case id 列表）。真实回归失败时按这张表定位到函数/数据文件，而不是去猜哪一步坏了。
