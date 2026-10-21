# 交付包（目标三件套：adapter + eval报告 + 增量库）

## 1. adapter（最优=ckpt-136）
- 路径：`paper_agent/outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/`
- 基座：Qwen3-4B，4bit NF4 + LoRA r32；权重252MB。
- 用法：venv用`/data1/paper_agent_venv`，推理关思考（`enable_thinking=False`），示例见`paper_agent/eval/gen_probe.py`。
- 性能：验证0.4469；60条NLL 0.3879（基座3.97）；证据绑定8/8；括号标签0（内容相符走第二判据）。

## 2. eval报告
- `paper_agent/reports/eval_3way_v2.json/md`（三方对比，键已订正：ckpt136/ckpt_final）
- `paper_agent/reports/P1_VERDICT.md`（P1门禁史：v1过拟合→v2最优136）
- `paper_agent/reports/P2_VERDICT.md`（P2无增益，136维持交付物；wrongbase事故封存）
- `paper_agent/reports/eval_3way_v2_thinkblocked.*`（思考污染版，provenance）
- `paper_agent/reports/gen_probe_136.json`（真实生成样本）

## 3. 增量库 + 检索
- `/data1/arxiv_incremental/arxiv.db`（17263行）+ `arxiv.index`（同数向量，MiniLM-384，与基座同空间）
- 基座：`/data1/peS2o_kb_faiss`（541745，只读）
- 融合：`paper_agent/retrieval/fuse.py`（四路+时鲜旋钮默认0），升级记录`RETRIEVAL_UPGRADE.md`

## 未达项（诚实）
- TMLR级别可用：不成立。缺真DPO（6GB装不下）、真人偏好、括号标签。
- 95分/v6冻结是进化Agent机制侧的自评，与本交付包模型质量无关。
