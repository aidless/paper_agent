# P5 合规链 (paper-writing-agent Phase 1/3/5植入)
- Agent system约束: claim ledger必填, 每个数字绑定evidence_refs; Abstract最后写; Methods/Results由证据值直写
- 工具门: verify_claim_ledger / scan_number / scan_stats / check_tmlr / verify_taint / compile_gate, FAIL阻断下一阶段
- TMLR硬线: G1模板匿名(\usepackage{tmlr}无选项) / G3-G4匿名化 / G9原创性重叠 / G10双盲隔离 / G11哈希对账
- 新鲜度门: check_literature_freshness --max-age-days 30, 过期先重跑D0增量
- 封存: evidence_manifest.json SHA-256 + prepare_submission, 上传保持MANUAL
- 当前: infer.py已含ledger提示词, 六扫脚本待paper-writing-agent skill全量接入后绑定
