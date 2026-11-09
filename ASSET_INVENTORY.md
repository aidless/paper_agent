# paper_agent 资产清单 — 2026-10-08

总大小：4.4G（含全部 checkpoint）。以下为核心可复用资产。

Adapter 两行指向 gitignore 的权重，本 clone 里没有这些文件，那两枚前 16 位哈希本次没有复算。表里改过路径或表述的文件已按当前字节重算；没改的文件仍与 2026-10-08 的前 16 位一致。这些哈希是文件身份，不是模型分数。P1 门禁仍是 FAIL。

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
| `data/build_p0v2.py` | 7656 B | `ed2de7d9301a22fb` | P0 v2 数据构建（路径修正后重算，不是新训练） |
| `data/build_dpo_v2.py` | 3344 B | `2389664883917bb9` | DPO 数据构建（路径修正后重算，不是新训练） |
| `eval/eval_3way.py` | 7267 B | `f168ac4449d151ce` | 三方评测（路径修正后重算，不是新评测） |

## 报告

| 文件 | 大小 | sha256（前16） | 复用价值 |
|---|---|---|---|
| `reports/eval_3way_v2.json` | 2.9KB | `1015cf050aacb42f` | 最优评测结果 |
| `reports/eval_3way_v2.md` | 1125 B | `bae321fa5e35835e` | 评测报告（行名与 0.125 对齐 JSON，数字未新测） |
| `reports/P1_VERDICT.md` | 1970 B | `19d7b2dd83cca46a` | P1 判决（仍是 FAIL） |
| `reports/P2_VERDICT.md` | 1379 B | `47aacba3ba025d69` | P2 判决（仍是 FAIL） |
| `DELIVERY.md` | 1730 B | `55c7d18f3232fc0e` | 交付文档（标明不推翻 P1 FAIL） |
| `evidence/env_fingerprint.json` | 4.4KB | `c1fc835fa7e2ad47` | 环境指纹 |

## 封存声明

全部原始文件保留不删。上述哈希为封存时快照，任何后续修改需重新计算并更新。
