# paper_agent 资产清单 — 2026-10-08

总大小：4.4G（含全部 checkpoint）。以下为核心可复用资产。

## 模型

| 文件 | 大小 | sha256（前16） | 复用价值 |
|---|---|---|---|
| `outputs/tmlr-qwen3-4b-agent/adapter_model.safetensors` | 264MB | `078ce7e645b624f8` | 最优 adapter（ckpt-136 等价） |
| `outputs/tmlr-qwen3-4b-agent/adapter_config.json` | 1.1KB | `e7acf4e032f5536f` | LoRA 配置 |

## 数据

| 文件 | 大小 | sha256（前16） | 复用价值 |
|---|---|---|---|
| `data/sft_p0_train.jsonl` | 3.2MB | `adedf1e22c151198` | SFT 训练集 |
| `data/sft_p0_val.jsonl` | 358KB | `2eb4fe7e59bd6fa3` | SFT 验证集 |
| `data/dpo_p0v2.jsonl` | 510KB | `520be7f85d599668` | DPO 偏好对（已构造） |
| `data/eval_fixed.jsonl` | 775B | `a428dfb4771305cd` | 评测集 |

## 脚本

| 文件 | 大小 | sha256（前16） | 复用价值 |
|---|---|---|---|
| `data/build_sft.py` | 3.1KB | `82dfc588ac262dff` | SFT 数据构建 |
| `data/build_p0v2.py` | 7.5KB | `6de9737867ab8125` | P0 v2 数据构建 |
| `data/build_dpo_v2.py` | 3.2KB | `302429b7ee50061b` | DPO 数据构建 |
| `eval/eval_3way.py` | 7.1KB | `140922d7b0d830b4` | 三方评测 |

## 报告

| 文件 | 大小 | sha256（前16） | 复用价值 |
|---|---|---|---|
| `reports/eval_3way_v2.json` | 2.9KB | `1015cf050aacb42f` | 最优评测结果 |
| `reports/eval_3way_v2.md` | 805B | `191d8fcba32e67fe` | 评测报告 |
| `reports/P1_VERDICT.md` | 1.6KB | `02e41ed0ef923996` | P1 判决 |
| `reports/P2_VERDICT.md` | 1.2KB | `5763223a6457e828` | P2 判决 |
| `DELIVERY.md` | 1.5KB | `bdbee94d957c55f8` | 交付文档 |
| `evidence/env_fingerprint.json` | 4.4KB | `c1fc835fa7e2ad47` | 环境指纹 |

## 封存声明

全部原始文件保留不删。上述哈希为封存时快照，任何后续修改需重新计算并更新。
