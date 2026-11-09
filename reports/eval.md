# Eval报告

`success` / `task_success` below is the mean of `pred_ok` in `data/eval_fixed.jsonl` (5 of 8). It is not a model score. `grounded_rate` 0.875 needs the external sqlite files and is not recomputed offline. The JSON has no per-sample rows, so the old line "mean(d)由逐样本重算" does not describe this file.

n=8, success=0.625, grounded=0.875

Holm校正待多配置时启用, bootstrap CI待n>=20
