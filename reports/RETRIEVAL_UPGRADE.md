# 检索升级记录（2026-09-10）

## 做了什么

1. 增量库向量索引：`/data1/arxiv_incremental/arxiv.index`（17263向量，384维，
   与peS2o基座同一MiniLM空间，分数可比）+ `arxiv_ids.json`
   构建脚本：`paper_agent/retrieval/build_incr_index.py`，日志：`paper_agent/reports/build_incr_index.log`
2. `paper_agent/retrieval/fuse.py`：新增 `search_incr_vector`（cosine向量检索），
   `fused_search` = 向量增量 + 关键词增量 + 关键词基座，去重后按分排序。
   缺索引文件时向量通道自动降级为空，不抛错。

## 实测（查询 "LLM agent evaluation"，top-10）

- 10条中9条来自向量通道，仅1条来自关键词；10条全部2026年 fresh 论文。
- 关键词整句LIKE几乎零命中（长查询分词OR兜底也只中1条），向量通道补上语义相关。
- 证据：`paper_agent/reports/retrieval_check.json`

## 环境教训

- `NO_PROXY` 含 `[::1]` 会使venv内旧版httpx构造client直接炸；以后凡加载HF模型，
  先清全部代理变量 + `HF_HUB_OFFLINE=1` + 直载本地snapshot。
- 训练/评估/检索统一用 `/data1/paper_agent_venv`（torch/st/faiss/peft/bnb齐全），
  不再用 `/tmp/sota-venv-*` 临时环境。

## 待办

- 增量索引随每日拉取重建（目前静态快照17263，新论文需重跑build脚本）。

## 基座向量臂（本轮追加）

- `search_pes2o_vector`：直读 `/data1/peS2o_kb_faiss/papers.index`（541744 vectors）
  + `paper_ids.txt` 映射 + `papers.db`，CPU运行，与训练抢不到显存。
- 不再复用 `kb_search`（其 `KB_DIR` 硬编码 Windows 路径 `E:/`，且同样染代理病）。
- 分数可比性修正：两路向量同为MiniLM-384余弦，取消自创的0.7折扣；
  关键词臂保留0.8/0.5旧权重（与余弦不可比，历史行为保留）。
- 实测 "LLM agent evaluation" top-10：基座向量6 + 增量向量3 + 关键词1。
- 诚实记录的trade-off：纯余弦排序下基座（0.76）压过增量新论文（0.59），
  新鲜度只靠日期字段与关键词臂；时鲜加权是下一步，不在本轮硬凑。
- 证据：`paper_agent/reports/retrieval_check4.json`

## 时鲜旋钮测量（fused_search新增recency_boost参数，默认0不改变行为）

- 同一查询 "LLM agent evaluation" top-10：boost=0.0时新鲜(2026+)4条；
  boost=0.1无变化；boost=0.2时增量7+基座2，新鲜8条。
- 默认保持0.0（纯相关性），下游按需传参；该常数未经标注集验证，不得写成“最优”。
