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

## 读数（AMD 48GB / bf16 / 2026-11-20 08:35 远端时钟）

| 模型 | val NLL(60) | tag | stage | evd | distinct-2 | loop | 平均长度 |
|---|---|---|---|---|---|---|---|
| base | 4.3935 | 0.00 | 0.875 | 0.25 | 0.5318 | 6 | 603 |
| ckpt136（SFT/p1v2） | 2.8648 | 0.00 | 0.75 | 0.75 | 0.5092 | 6 | 282 |
| tmlr-qwen3-4b-p2（DPO） | **2.7079** | 0.00 | 0.75 | 0.75 | **0.5592** | **3** | 303 |

## 判定（机械执行，不是口头对照）

`eval/judge_p2.py reports/eval_3way_p2.json` → **exit 0，4/4**：

```
PASS  val NLL 不退化        候选=2.7079  基线=2.8648 (le)
PASS  证据绑定率不退化      候选=0.75    基线=0.75   (ge)
PASS  distinct-2 不下降    候选=0.5592  基线=0.5092 (ge)
PASS  循环复读不增加        候选=3       基线=6      (le)
```

判决写入 `reports/eval_3way_p2.verdict.json`。

**结论（已被下方 40 条复核推翻，保留作历史记录）：** 当时判 4/4、NLL 降 5.5%、distinct-2 升 9.8%、loop 6→3。**不要引用这一段下结论，往下看「40 条复核」。**

## 必须一起读的三条边界

1. **生成样本只有 8 条**，evd/d2/loop 都是这 8 条上的计数。evd 0.75 = 6/8、loop 3 = 3/8，**一个样本翻转就是 12.5 个百分点**。这三条只能当方向性信号，不能当「d2 提升 9.8%」这种精确断言引用。只有 NLL 是 60 条样本。
2. **NLL 绝对值不可与历史 `eval_3way_v2.md` 比**（那是 bnb 4bit 读数，本轮 bf16 全量）。本轮内三方同 substrate，相对关系可用。
3. **stage 从 base 的 0.875 掉到两个微调版的 0.75** —— 这条不在四条判据里，但也不能不提：两个 adapter 都牺牲了一点内容阶段符合率，且 DPO 没有把它补回来。base 的平均长度 603 vs 282/303 说明 base 输出更长，可能因此更容易命中阶段关键词，这个对比本身不干净。

## 验收
- `pytest tests/ -q` → **63 passed**（15 条 judge_p2 + 14 条 eval_config + 34 条既有）
- 九条变异反证（全部 RED）：M1 判据恒过 → 3 failed；M2 方向反了 → 7 failed；M3 缺字段静默为 0 → 1 failed；M5 结构错当通过 → RED；M6 错误文案丢字段路径 → 1 failed；M7 布尔检查剥离 → 1 failed；M8 缺模型段不校验 → 1 failed；M9 少一条判据 → 7 failed。**M4 曾假绿**（把 JSON 解析错误当通过），补 `test_cli_缺字段返回2` 后 M5 才真红 —— 记录在此：只测「坏 JSON」测不出「坏结构」。
- `ruff check --select F,E9` → All checks passed
- `python3 eval/judge_p2.py reports/eval_3way_p2.json` → exit 0，4/4（**注意：这是 8 条样本的结果，已被下方 40 条复核推翻**）

---

## 40 条复核（2026-11-20 02:18）——本轮最重要的发现

8 条样本下 evd=6/8、d2 领先 0.05，一个样本翻转就是 12.5pp。为检验 4/4 是不是小数定律，把 `N_GEN` 改成可配置（默认仍 8/42，`EVAL_N_GEN=40` 覆盖，commit `f1ad80d`），在同一 AMD 实例、同一 fp16 基座、同一 seed 下重跑，生成样本 8→40。

`$ judge_p2.py reports/eval_3way_p2_ng40.json` → **exit 1，2/4**：

| 判据 | 8 条 p2/p1 | 8 条判 | 40 条 p2/p1 | 40 条判 |
|---|---|---|---|---|
| NLL (le) | 2.7079 / 2.8648 | PASS | 2.7074 / 2.8653 | PASS |
| evd (ge) | 0.75 / 0.75 | PASS | **0.85 / 0.90** | **FAIL** |
| d2 (ge) | 0.5592 / 0.5092 | PASS | **0.6213 / 0.6268** | **FAIL** |
| loop (le) | 3 / 6 | PASS | 20 / 23 | PASS |

判决写入 `reports/eval_3way_p2_ng40.verdict.json`。

**结论反转：8 条的「4/4 全过、DPO 有正收益」是小数定律假象。40 条上只有 NLL 和 loop 两条稳定站住，evd 和 d2 都翻转——DPO 在这两条上不优于 SFT，甚至略差。**

- **evd**：8 条 DPO 与 SFT 恰好持平（都 6/8）。40 条上 SFT 36/40 > DPO 34/40，SFT 反超 2 条。8 条的「持平」是碰巧。
- **d2**：8 条 DPO 领先 0.05（0.5592 vs 0.5092），40 条 SFT 反超 0.0055（0.6268 vs 0.6213）。8 条的领先完全没有泛化。
- **稳定项**：NLL 两次几乎不变（2.7079→2.7074）——它是 60 条固定子集算的，本就不受 N_GEN 影响；loop 两次都是 DPO 更少（3/6 → 20/23），方向稳健。

## 修正后的最终结论（以 40 条为准）

**DPO 相对 SFT 只有一条稳健收益：循环复读更少（40 条上 20 vs 23，8 条上 3 vs 6，方向一致）。NLL 也略低但本就不随生成样本数变。此前宣称的「evd 持平、d2 提升 9.8%」在 40 条上不成立，撤回。**

这不是「DPO 有害」——evd 0.85 vs 0.90、d2 0.6213 vs 0.6268 的差距很小，且 40 条仍是小样本。准确说法是：**扩样本后证据不足以支持「DPO 在证据绑定和多样性上优于 SFT」，只支持「DPO 更少复读」。**

## 这对成本账的影响（必须一起改）

`cost-per-correct/examples.dpo-vs-sft-2026-11-20.json` 的三合一合规率在 40 条上变成：base 0/40=0.0、SFT 17/40=0.425、DPO 20/40=0.500。DPO 仍领先，但幅度从 0.625 vs 0.25 缩到 0.50 vs 0.425——**8 条把优势放大了**。成本账案例以 40 条为准重算。

---

## 跨 seed 复核（2026-11-20 02:53）——第二段反转，连「loop 稳健」也站不住

40 条复核后我说「loop 更少复读是唯一稳健收益」。为检验这句话，在 40 条、两个不同 seed（42 / 123）下各跑一次（seed 只影响生成子集，`EVAL_SEED=123`；NLL 子集固定不变）。两次结果：

| 判据 | seed42 DPO/SFT | 判 | seed123 DPO/SFT | 判 | 跨 seed |
|---|---|---|---|---|---|
| NLL (le) | 2.7074 / 2.8653 | PASS | 2.6638 / 2.8118 | PASS | **稳健 DPO 优** |
| evd (ge) | 0.85 / 0.90 | FAIL | 0.85 / 0.875 | FAIL | **稳健 SFT 优** |
| d2 (ge) | 0.6213 / 0.6268 | FAIL | 0.6747 / 0.667 | PASS | 噪声（跨 seed 反转） |
| loop (le) | 20 / 23 | PASS | 20 / 16 | FAIL | 噪声（跨 seed 反转） |

两次都是 **2/4，但失败的判据不同**。判决分别落 `reports/eval_3way_p2_ng40.verdict.json` 与 `reports/eval_3way_p2_ng40_s123.verdict.json`。

**这推翻了 40 条复核时的「loop 稳健」结论**：seed42 上 DPO 复读 20<23（少），seed123 上 DPO 20>16（多）——DPO 自己的复读数两次都是 20，是 SFT 从 23 变成 16 造成了反转。也就是说连「更少复读」都不是 DPO 的稳定属性，而是相对 SFT 在某一抽样下的表现。

三合一合规率也反转：seed42 SFT 0.425 < DPO 0.500，seed123 SFT 0.550 > DPO 0.475。

### 最终结论（以两个 seed 的交集为准）

- **稳健成立的只有两条**：NLL DPO 更低（两次都过，且本就不随抽样变）；evd **SFT 更高**（两次都过——注意这次是 SFT 方向，与 8 条时「持平」相反）。
- **d2 与 loop 是噪声**：跨 seed 反转，没有稳定方向。
- **没有任何一条支持「DPO 在生成质量上优于 SFT」**。此前 8 条的「4/4 全面正收益」、40 条的「loop 稳健」、成本账的「DPO 赢 1.17×」——三次结论逐层被推翻。
- 准确说法：**DPO 只稳定降低了验证集 NLL；生成质量维度上，证据绑定 SFT 稳定略优，多样性与复读在噪声范围内，无法判谁更好。** 差距都小、样本都只有 40 条，这不是「DPO 有害」的证据。

### 对成本账的二次影响（winner 反转）

按两个 seed 各自的三合一合规率算每正确成本（同推理单价 273.72）：

| seed | SFT 合规/成本每正确 | DPO 合规/成本每正确 | winner |
|---|---|---|---|
| 42 | 0.425 / 0.198366 | 0.500 / 0.170254 | DPO 1.17× |
| 123 | 0.550 / 0.152786 | 0.475 / 0.183825 | **SFT 1.20×** |

**成本账 winner 跨 seed 反转** —— seed42 案例里的「DPO 便宜 1.17×」不稳健，必须标注为单 seed 读数。成本账文件与 README 第四案例据此更正（另仓）。
