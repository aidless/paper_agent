# -*- coding: utf-8 -*-
"""ckpt-136生成探针: 看实际输出文本, 诊断tag=0与循环复读."""
import json
import os
import random
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 200

for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    os.environ.pop(_v, None)
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AD = "paper_agent/outputs/tmlr-qwen3-4b-p1v2/checkpoint-136"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", quantization_config=bnb,
                                         device_map="auto", trust_remote_code=True)
m = PeftModel.from_pretrained(m, AD)
m.eval()
val = [json.loads(l) for l in open("paper_agent/data/sft_p0v2_val.jsonl", encoding="utf-8")]
random.seed(7)
out = []
for i in random.sample(range(len(val)), N):
    msgs = val[i]["messages"]
    pre = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    inp = tok(pre, return_tensors="pt").to(m.device)
    with torch.no_grad():
        gen = m.generate(**inp, max_new_tokens=MAXTOK, do_sample=False)
    txt = tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    out.append({"prompt": msgs[0]["content"][:80], "stage": val[i]["task_stage"], "text": txt})
    print("=== STAGE", val[i]["task_stage"], "PROMPT:", msgs[0]["content"][:60])
    print(txt[:700])
    print()
json.dump(out, open("paper_agent/reports/gen_probe_136.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
