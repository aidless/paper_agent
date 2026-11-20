# DPO 模型来源可覆盖：DPO_BASE_MODEL（2026-11-20）

## 症状
AMD Radeon 云实例上要用本地已 curl 下载的 Qwen3-4B，但 `train_dpo.py` 把模型来源写死成 HF repo id `Qwen/Qwen3-4B`，`from_pretrained` 会重新走 huggingface_hub 联网。而该实例上 hub 的 `resolve` HEAD 请求会卡死（实测僵 16 小时，缓存只落 28K），训练根本起不来。

## 根因
huggingface_hub 走集群代理时对 `resolve/.../config.json` 的 HEAD 请求超时重试无限循环；同一链路用 curl 带代理下载能到 6.4MB/s。所以模型权重必须用 curl 落到本地目录，训练脚本要能读本地路径。

## 改法
- 新增 `resolve_base_model(env) -> str`：`DPO_BASE_MODEL` 环境变量非空则用它（strip），否则回落 `CFG["base_model"]`。
- `main()` 里 `base_model = resolve_base_model(os.environ)`，两处 `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained` 都改读 `base_model`。
- `cfg_snapshot.json` 记 `base_model` 实际值，产物可追溯。

## 验收
- RED：`tests/test_dpo_base_model.py` 5 条先红（`resolve_base_model` 不存在 + main 仍裸写 `CFG["base_model"]`）。
- GREEN：实现后 5 条全绿；paper_agent 全套件 `34 passed`。
- 变异：把 tokenizer 的 `from_pretrained(base_model)` 改回 `from_pretrained(CFG["base_model"])` → `test_main的tokenizer与模型加载都走resolver` 必红。
- ruff `--select F,E9`：All checks passed。

## 真跑证据
AMD gfx1100 / 48GB / ROCm 6.4：`DPO_BASE_MODEL=/workspace/models/Qwen3-4B HF_HUB_OFFLINE=1 python3 train_dpo.py --no-4bit --gpu-mem-mib 36000 --smoke` → 10/10 步跑完，`[mem] before train: allocated=8.029 GiB`（Kaggle T4 是 14.23 GiB 且 OOM），loss 0.1387 → 1.18e-7，adapter 落盘。
