# -*- coding: utf-8 -*-
"""P1三方对比: base vs ckpt-68 vs ckpt-270.
指标: val子集NLL(同一样本同一mask, 可比) + 生成质量(阶段标签/证据绑定/distinct-2/循环复读).
输出: paper_agent/reports/eval_3way.json / .md
"""
import json, os, re, gc, sys, torch
for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    os.environ.pop(_v, None)
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen3-4B"
AD68 = sys.argv[1] if len(sys.argv) > 1 else "paper_agent/outputs/tmlr-qwen3-4b-p1/checkpoint-68"
AD270 = sys.argv[2] if len(sys.argv) > 2 else "paper_agent/outputs/tmlr-qwen3-4b-p1"
VAL = sys.argv[3] if len(sys.argv) > 3 else "paper_agent/data/sft_p0_val.jsonl"
OUT_JSON = sys.argv[4] if len(sys.argv) > 4 else "paper_agent/reports/eval_3way.json"
OUT_MD = OUT_JSON[:-5] + ".md" if OUT_JSON.endswith(".json") else OUT_JSON + ".md"
N_NLL = 60
N_GEN = 8
SEED = 42

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

val = [json.loads(l) for l in open(VAL, encoding="utf-8") if l.strip()]
rng = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(val), generator=rng).tolist()
nll_set = [val[i] for i in perm[:N_NLL]]
gen_set = [val[i] for i in perm[N_NLL:N_NLL + N_GEN]]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

def build_inputs(msgs):
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(full, return_tensors="pt").input_ids[0]
    pre = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)  # 关思考：与训练格式对齐，直出正文
    pre_len = len(tok(pre).input_ids)
    labels = ids.clone()
    labels[:pre_len] = -100
    if (labels != -100).sum() == 0:
        labels[:] = ids
    return ids, labels

def load_model(kind):
    m = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    if kind == "ckpt68":
        m = PeftModel.from_pretrained(m, AD68)
    elif kind == "ckpt270":
        m = PeftModel.from_pretrained(m, AD270)
    m.eval()
    return m

def val_nll(model):
    tot_ll, tot_t = 0.0, 0
    with torch.no_grad():
        for ex in nll_set:
            ids, labels = build_inputs(ex["messages"])
            ids = ids.unsqueeze(0).to(model.device)
            labels = labels.unsqueeze(0).to(model.device)
            out = model(input_ids=ids, labels=labels)
            n_valid = (labels != -100).sum().item()
            tot_ll += out.loss.item() * n_valid
            tot_t += n_valid
    return tot_ll / max(tot_t, 1)

TAG = re.compile(r"^\s*\[[A-Za-z0-9\u4e00-\u9fa5]+\]")
EVD = re.compile(r"证据|arxiv|/\w+_\w+|20\d{2}|cvpr|icml|nips|iclr|acl|绑定|无证据", re.IGNORECASE)
# 内容型阶段符合度（括号标签之外的第二判据：harness本来就知道自己在跑哪阶段，
# 模型输出内容与阶段相符即算数；括号只是便利标记，不代表能力）
STAGE_KEYS = {"D0": ["综述", "GAP", "基线"], "D1": ["假设", "消融", "证伪"],
              "D2": ["种子", "配置", "复现"], "D3": ["方法", "证据"],
              "D4": ["攻击", "审稿", "修改"], "D5": ["核查", "一致", "合规"]}

def gen_metrics(model):
    res = []
    for ex in gen_set:
        prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)  # 关思考直出
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        toks = list(txt)
        bigrams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
        d2 = len(set(bigrams)) / max(len(bigrams), 1)
        grams8 = ["".join(toks[i:i + 8]) for i in range(len(toks) - 7)]
        loop = any(grams8.count(g) >= 3 for g in set(grams8)) if grams8 else False
        st = ex.get("task_stage", "")
        keys = STAGE_KEYS.get(st, [])
        stage_ok = any(k in txt for k in keys) if keys else False
        res.append({"tag": bool(TAG.search(txt)), "evd": bool(EVD.search(txt)),
                    "stage": bool(stage_ok),
                    "d2": round(d2, 4), "loop": loop, "len": len(txt)})
    agg = {k: (round(sum(r[k] for r in res) / len(res), 4) if k != "loop" else sum(r[k] for r in res))
           for k in ("tag", "evd", "stage", "d2", "loop", "len")}
    agg["loop"] = int(agg["loop"])
    agg["len"] = int(agg["len"])
    return agg, res

results = {}
# 键名从适配器路径派生（防误标：曾把136步标成68步）：取路径末段，去掉checkpoint-前缀
def _key(path, fallback):
    b = os.path.basename(path.rstrip("/")) or fallback
    return b.replace("checkpoint-", "ckpt")
ADKEY68, ADKEY270 = _key(AD68, "ckpt68"), _key(AD270, "ckpt270")
for kind in ("base", "ckpt68", "ckpt270"):
    print("=== %s ===" % kind, flush=True)
    model = load_model(kind)
    nll = val_nll(model)
    agg, det = gen_metrics(model)
    results[kind] = {"val_nll_%d" % N_NLL: round(nll, 4), "gen": agg, "gen_detail": det}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(results[kind]["gen"], ensure_ascii=False), flush=True)

os.makedirs("paper_agent/reports", exist_ok=True)
results = {"base": results["base"], ADKEY68: results["ckpt68"], ADKEY270: results["ckpt270"],
           "_meta": {"AD68": AD68, "AD270": AD270, "VAL": VAL,
                     "note": "键名由适配器路径派生；NLL同一样本同掩码可比"}}
json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
b, c68, c270 = results["base"], results[ADKEY68], results[ADKEY270]
md = ("# P1三方对比评估\n\nNLL样本=%d(同一子集同一mask), 生成样本=%d(贪心, 200token上限)\n\n"
      "| 模型 | val NLL | 括号标签率 | 内容阶段符合率 | 证据绑定率 | distinct-2 | 循环复读数 | 平均长度 |\n"
      "|---|---|---|---|---|---|---|---|\n" % (N_NLL, N_GEN))
for name, r in (("base", b), (ADKEY68, c68), (ADKEY270, c270)):
    g = r["gen"]
    md += "| %s | %.4f | %.2f | %.2f | %.2f | %.4f | %d | %d |\n" % (
        name, r["val_nll_%d" % N_NLL], g["tag"], g.get("stage", -1), g["evd"], g["d2"], g["loop"], g["len"])
md += ("\n判读: NLL越低拟合验证分布越好; 若ckpt-270的NLL高于ckpt-68, 则过拟合成立, "
       "以ckpt-68为P1交付物并做早停重训。\n")
open(OUT_MD, "w", encoding="utf-8").write(md)
print(md, flush=True)
