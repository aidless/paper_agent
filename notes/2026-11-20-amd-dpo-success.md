# AMD Radeon Cloud 通道跑通 DPO（2026-11-20）

## 背景
Kaggle T4 16GB 上 DPO 连续 v8/v9/v10 OOM：占用 14.19 / 14.21 / 14.23 GiB，缺口仅 20–48 MiB，precompute/use_reentrant/enable_input_require_grads/empty_cache 全部削峰无效——因为占用几乎不变，说明不是激活峰值而是 fp16 权重 + DPO 框架常驻把 16GB 压线。周配额耗尽等 11-21。用户给 AMD Radeon Cloud 实例作替代通道。

## 实例事实（现测）
- GPU gfx1100（RDNA3），VRAM 51522830336 B ≈ 48GB，capability (11,0)，48 CU
- torch 2.9.1+rocm6.4，bf16/fp16 matmul smoke 均通过
- HF 直连超时；hf-mirror.com 可达但 **huggingface_hub 的 resolve HEAD 请求僵死**（实测 16h 缓存只落 28K）；curl 带集群代理下载 6.4MB/s
- pip 需 `--break-system-packages`；装 transformers 5.23.0 / trl 1.18.2 / peft 0.22.0 / datasets 5.2.0

## 关键适配
1. **模型来源**：curl 带代理从 hf-mirror 下 Qwen3-4B（7.6G，398 权重条目）到 `/workspace/models/Qwen3-4B`；train_dpo.py 加 `resolve_base_model(env)` 支持 `DPO_BASE_MODEL` 指本地路径（commit ab9d3b7，TDD 5 条先红后绿 + 变异必红）。
2. **显式 --no-4bit**：capability(11,0)>=8 会让 load_plan 默认走 4bit+bf16，但 bnb 未装且 ROCm 上不可靠 → 强制非量化 bf16。
3. **--gpu-mem-mib 36000**：脚本默认 4600MiB 是本地 6GB 卡的保守值，48GB 卡必须调大否则权重被挤 CPU。
4. SFT adapter checkpoint-136 上传 → mode=resume_sft（比 fresh_lora 起点好）。

## 真跑读数
- smoke 10 步：loss 0.1387 → 1.18e-7，rewards/accuracies 0.8→1.0，adapter 落盘
- 正式 --p2：400 条 / 1 epoch / 50 步，train_runtime 261s，train_loss 0.01387，rewards/accuracies 0.8→1.0
- **`[mem] before train: allocated=8.029 GiB reserved=8.467 GiB`** —— 比 Kaggle 14.23 GiB 低 6.2 GiB

## 结论
OOM 死结是**卡容量**不是代码：同一份脚本在 48GB 上只用 8GB 就跑完。之前所有削峰尝试方向都对但治不了根——16GB 装不下这个配置的常驻占用。

## 产物
`paper_agent/outputs/tmlr-qwen3-4b-p2-amd/`（gitignored，本地留存）：adapter_model.safetensors 253M / 504 tensors（r=32 alpha=16 LORA CAUSAL_LM）、cfg_snapshot.json（base_model=/workspace/models/Qwen3-4B）、p2-train.log。sha256 前 24 位 e954a8ae1289ccb5690e2065，与远端逐字节一致。

## 验收
- paper_agent 全套件 34 passed；ruff F/E9 clean
- adapter_config + safetensors 结构核对通过（504 tensors = 训练日志 trainable tensors）
- 台账 L-ops-813
