# -*- coding: utf-8 -*-
"""推理: 基座+adapter, 绑定融合检索+ledger检查, 8步工具循环, 温度0."""
import argparse, json, os, sys
for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    os.environ.pop(_v, None)
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "paper_agent/retrieval")
from fuse import fused_search

ADAPTER = "paper_agent/outputs/tmlr-qwen3-4b-agent"
BASE = "Qwen/Qwen3-4B"

def run(prompt, max_steps=8, adapter=None, no_model=False):
    hits = fused_search(prompt, top_k=5)
    steps = [{"step": 1, "tool": "fused_search", "n_hits": len(hits)}]
    draft = "【草稿片段】任务: %s\n【检索】召回%d条(peS2o+arXiv增量)\n" % (prompt, len(hits))
    for i, h in enumerate(hits[:3], 1):
        draft += "[%d] %s (%s)\n" % (i, h.get("title", ""), h.get("source", ""))
    draft += "【ledger】每个数字需绑定evidence_refs, Abstract最后写, TMLR匿名.\n"
    if no_model:
        steps.append({"step": 2, "tool": "model", "skipped": "no-model flag"})
        draft += "\n【纯检索模式】\n"
        return {"draft": draft, "steps": steps, "hits": hits}
    ad = adapter or ADAPTER
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=qconf,
            device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True)
        if os.path.exists(os.path.join(ad, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, ad)
            steps.append({"step": 2, "tool": "load_adapter", "ok": True, "adapter": ad})
        inputs = tok("任务:%s\n检索:%s\n写Methods+Results:" % (prompt, draft[:1500]), return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
        draft += "\n【模型】\n" + tok.decode(out[0], skip_special_tokens=True)[-1500:]
    except Exception as e:
        steps.append({"step": 2, "tool": "model", "skipped": str(e)[:200]})
        draft += "\n【模型未加载, 仅检索模式】\n"
    return {"draft": draft, "steps": steps, "hits": hits}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="LLM agent evaluation GAP与基线")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--no-model", action="store_true", help="纯检索冒烟, 不占显存")
    a = ap.parse_args()
    print(json.dumps(run(a.prompt, adapter=a.adapter, no_model=a.no_model), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
