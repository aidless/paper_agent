# P2 结论（2026-09-11 实测）

## 路线
真DPO在6GB卡上四次OOM（单模型2.94G，双路前向峰值超限；预计算/缩长度/卸载均无效），
改道优选答案监督微调（ReST式单轮），吃同一400对偏好中的chosen。

## 事故
- 第一版P2从基座从零起步（脚本缺--from-adapter），验证3.56，封存为`tmlr-qwen3-4b-p1v2_wrongbase`。
- 第二版从ckpt-136接续（lr 5e-5，1轮，25步）：验证0.4895 vs 起点0.4469。

## E5对决（P2-final vs ckpt-136，内容判据，E2同8条）
- b=0 c=0 p=1.0，8战全平 → hold。
- 结论：合成优选微调未带来可测增益，ckpt-136仍是交付物。

## 门禁：FAIL（无增益）
- 合成数据天花板已现：SFT-v2把证据绑定打到8/8后，继续在合成分布上打转不再涨分。
- 下一增益需：真DPO（大显存）或真人偏好数据或真实轨迹数据。三者皆缺时，不再烧轮次。

## 交付物（当前最优）
- adapter：`paper_agent/outputs/tmlr-qwen3-4b-p1v2/checkpoint-136/`（验证0.4469，NLL 0.3879，证据8/8）
- 评估：eval_3way_v2.json/md + P1_VERDICT.md + 本文件
- 增量库：/data1/arxiv_incremental（17263行+向量索引）+ 四路融合检索
