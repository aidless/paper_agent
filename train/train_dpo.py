# -*- coding: utf-8 -*-
"""P2 DPO: SFT adapter + 合成偏好对(dpo_p0v2), 去模板腔、学证据绑定.
配置即证据: 超参内嵌并输出cfg_snapshot.json (FM-29).
venv: /tmp/sota-venv-ap17 (torch2.14+trl1.13; paper_agent_venv的trl1.12与torch2.5不兼容, DPOTrainer无法import)
  冻结清单: paper_agent/reports/dpo_env_freeze.txt (/tmp易失, 重建用)
smoke: /tmp/sota-venv-ap17/bin/python train_dpo.py --smoke   # 10步链路验证
p2:    /tmp/sota-venv-ap17/bin/python train_dpo.py --p2       # P2正式DPO (默认)
"""
import argparse, json, os, random
import numpy as np

CFG = {
  "base_model": "Qwen/Qwen3-4B",
  "sft_adapter": "paper_agent/outputs/tmlr-qwen3-4b-p1v2/checkpoint-136",
  "dpo_train_file": "paper_agent/data/dpo_p0v2.jsonl",
  "p2_output_dir": "paper_agent/outputs/tmlr-qwen3-4b-p2",
  "smoke_output_dir": "paper_agent/outputs/tmlr-qwen3-4b-p2smoke",
  "beta": 0.1,
  "lr": 5e-5,
  "epochs": 1,
  "per_device_bs": 1,
  "grad_accum": 8,
  "max_len": 1024,
  "seed": 42,
  "warmup_steps": 5,
  "logging_steps": 5,
  "save_steps": 25,
  "save_total_limit": 1,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "bnb_4bit_use_double_quant": True,
}

def set_seed(s):
    random.seed(s); np.random.seed(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    smoke = a.smoke
    out_dir = CFG["smoke_output_dir"] if smoke else CFG["p2_output_dir"]
    train_file = CFG["dpo_train_file"]
    sft_ad = CFG["sft_adapter"]
    if a.out_dir: out_dir = a.out_dir
    if a.train_file: train_file = a.train_file
    if a.sft_adapter: sft_ad = a.sft_adapter
    set_seed(CFG["seed"])
    os.makedirs(out_dir, exist_ok=True)
    snap = dict(CFG); snap.update({"output_dir": out_dir, "train_file": train_file,
        "sft_adapter": sft_ad, "mode": "smoke" if smoke else "p2"})
    with open(os.path.join(out_dir, "cfg_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print("[cfg] snapshot -> %s/cfg_snapshot.json" % out_dir)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from trl import DPOTrainer, DPOConfig
        from datasets import load_dataset
    except Exception as e:
        print("[error] 缺依赖: %s" % e)
        return
    for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        os.environ.pop(_v, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("[model] load %s + SFT %s (4bit)" % (CFG["base_model"], sft_ad))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=CFG["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        max_memory={0: "4600MiB", "cpu": "12GiB"})  # P2：6GB卡DPO峰值超限，溢出几层到CPU（慢但不断）
    model.gradient_checkpointing_enable()
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    if os.path.exists(os.path.join(sft_ad, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, sft_ad, is_trainable=True)
        # 量化基座+载入adapter后，显式解冻LoRA参数（否则DPOTrainer报无可训练参数）
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        print("[sft] adapter attached+unfrozen: %s" % sft_ad)
    else:
        print("[warn] SFT adapter缺失, 从基座直做DPO: %s" % sft_ad)
    ds = load_dataset("json", data_files=train_file)["train"]
    print("[data] pairs=%d" % len(ds))
    args = DPOConfig(
        output_dir=out_dir, beta=CFG["beta"],
        per_device_train_batch_size=CFG["per_device_bs"],
        gradient_accumulation_steps=CFG["grad_accum"], learning_rate=CFG["lr"],
        num_train_epochs=1, max_steps=10 if smoke else -1,
        max_length=384,  # P2省显存：偏好对实测最长363token（2026-09复核），384零截断
        precompute_ref_log_probs=False,  # P2省显存：True会另载一份ref模型，双倍权重必爆；False走adapter-disable共享基座
        logging_steps=CFG["logging_steps"], save_steps=CFG["save_steps"],
        seed=CFG["seed"], report_to="none",
        save_total_limit=CFG["save_total_limit"], gradient_checkpointing=True,
    )
    try:
        trainer = DPOTrainer(model=model, ref_model=None, args=args,
                             train_dataset=ds, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(model=model, ref_model=None, args=args,
                             train_dataset=ds, tokenizer=tok)
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("[done] dpo adapter -> %s" % out_dir)

if __name__ == "__main__":
    main()
