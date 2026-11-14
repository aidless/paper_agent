# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: tests/
# 命令逐字取自该来源，未增删任何参数。
.PHONY: eval

# 可执行验收题库（20 条七键 case；--selftest 证明它会红）
eval:
	python3 eval/run_cases.py
	python3 -m pytest tests -q
