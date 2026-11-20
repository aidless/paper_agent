# DPO 产物评测：为什么在 AMD 上跑、判据是什么（2026-11-20）

## 起点
AMD 48GB 通道训完 DPO（p2，50 步，train_loss 0.01387，rewards/accuracies 0.8→1.0）。
但 `rewards/accuracies` 是**训练内**信号，不能当结论：它衡量的是 chosen/rejected 的相对序，
不是生成质量。训练完必须用独立验证集做三方对比（base / SFT / DPO），否则「DPO 有用」是未验证的断言。

## 为什么不在本机跑
本机 `~/.cache/huggingface/hub/models--Qwen--Qwen3-4B` 只有 12K 骨架（resolve 请求僵死留下的），
没有 fp16 权重；本机只有 AWQ 版本，拿它当 DPO 的 base 会二次量化，口径更脏。

AMD 实例上有训练时用的同一份 fp16 基座（`/workspace/models/Qwen3-4B`）+ 两个 adapter（p1v2/p2），
用 `torch_dtype=bfloat16` 全量加载即可与训练口径完全对齐。代价是那边没有 bitsandbytes，
所以必须能让脚本关掉 4bit —— 这就是 `eval/eval_config.py` 的由来。

## 改动（TDD）
1. `paper_agent/tests/test_eval_config.py` 先写 → RED（`ModuleNotFoundError: No module named 'eval_config'`）
2. `paper_agent/eval/eval_config.py` 后写 → GREEN（14 passed）
3. 期间修正一处自相矛盾的测试契约：我第一版把 `""` 同时放进 `_FALSEY` 又用 `or "1"` 兜底，
   空串到底算「关」还是「算未设」说不清。钉死为**只有明确 false 词才算关，空值一律当未设**
   （空值最可能来自「变量没传」，不是「我想关」）。
4. `eval/eval_3way.py` 改为调用这两个函数：`BASE = resolve_base_model(os.environ)`、
   `LOAD_4BIT = use_4bit(os.environ)`；`BitsAndBytesConfig` 只在 `LOAD_4BIT` 为真时构造
   （条件表达式只求值被选中的分支，关掉时不会碰未定义的 `bnb`）。
5. 验收：`pytest tests/ -q` → **48 passed**；`py_compile` OK；`ruff --select F,E9` clean。

## 判据先声明（跑前锁定，结果未出）

**关键口径**：历史 `eval_3way_v2.md` 的读数是 bnb 4bit 跑出来的，本轮是 bf16 全量。
绝对值**不可跨轮直接比**（4bit 量化误差会系统性抬高 NLL）。判据只建立在**本轮内同 substrate 的相对关系**上。

| 轴 | 判据 | 若不满足怎么读 |
| --- | --- | --- |
| val NLL | p2 ≤ p1v2 | p2 > p1v2 → DPO 在验证分布上拟合退化，不能声称正收益 |
| 证据绑定率 evd | p2 ≥ p1v2 | 下降 → DPO 把格式约束学坏了，属有害 |
| distinct-2 | p2 ≥ p1v2 | 下降 → 输出重复度上升 |
| 循环复读 loop | p2 ≤ p1v2 | 增加 → 复读变严重 |
| 平均长度 / 阶段标签率 | 仅记录，不作判据 | `tag` 在历史上两轮都是 0.00，该 metric 本身不敏感 |

四条全部满足才算「DPO 在 SFT 之上有正收益」；任何一条退化都必须如实写进报告，不许用另外三条的好看抵消。

## 运行命令（在 AMD 上）
```bash
EVAL_BASE_MODEL=/workspace/models/Qwen3-4B EVAL_LOAD_4BIT=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 eval/eval_3way.py outputs/tmlr-qwen3-4b-p1v2/checkpoint-136 outputs/tmlr-qwen3-4b-p2 \
  data/sft_p0_val.jsonl reports/eval_3way_p2.json
```

## 验收
- `pytest tests/ -q` → 48 passed
- `py_compile eval/eval_3way.py eval/eval_config.py` → OK
- `ruff check --select F,E9 eval/eval_3way.py eval/eval_config.py` → All checks passed
- 评测结果待回填（本轮结果出来后在报告节补读数，判据不改）
