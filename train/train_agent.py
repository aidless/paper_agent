# -*- coding: utf-8 -*-
"""Qwen3-4B QLoRA训练 (6GB RTX3060 Laptop可用).
配置即证据: 全部超参内嵌并输出cfg_snapshot.json (FM-29).
venv: /data1/paper_agent_venv
smoke: python train_agent.py --smoke          # 10步链路验证 (旧5条数据)
p1:    python train_agent.py --p1             # P1正式SFT (P0 2160训/240验)
full:  python train_agent.py                  # = --p1
"""
import argparse, json, os, random
import numpy as np
for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    os.environ.pop(_v, None)
os.environ["HF_HUB_OFFLINE"] = "1"

CFG = {
  "base_model": "Qwen/Qwen3-4B",
  "quant": "bnb_4bit_NF4",
  "lora_r": 64, "lora_alpha": 16, "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "seq_len": 2048, "fallback_seq_len": 2048, "fallback_r": 32,
  "per_device_bs": 1, "grad_accum": 16, "lr": 2e-4,
  "scheduler": "cosine", "warmup_ratio": 0.03, "epochs": 2,
  "seed": 42,
  "warmup_steps": 8,  # ~3% of 270 P1 steps (2160样/bs1/accum16/ep2); trl1.12无warmup_ratio
  "smoke_output_dir": "paper_agent/outputs/tmlr-qwen3-4b-agent",
  "p1_output_dir": "paper_agent/outputs/tmlr-qwen3-4b-p1",
  "smoke_train_file": "paper_agent/data/sft_fullprocess.jsonl",
  "p1_train_file": "paper_agent/data/sft_p0_train.jsonl",
  "p1_val_file": "paper_agent/data/sft_p0_val.jsonl",
  "bnb_4bit_use_double_quant": True, "gradient_checkpointing": True,
  "eval_steps": 68, "logging_steps": 10, "save_total_limit": 2,
}

def set_seed(s):
    random.seed(s); np.random.seed(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑10步验证链路")
    ap.add_argument("--p1", action="store_true", help="P1正式SFT (默认)")
    ap.add_argument("--train-file", default=None, help="覆盖训练数据路径")
    ap.add_argument("--val-file", default=None, help="覆盖验证数据路径")
    ap.add_argument("--out-dir", default=None, help="覆盖输出目录")
    ap.add_argument("--from-adapter", default=None, help="续训起点SFT adapter（P2用）")
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率（P2续训用小lr）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖轮数")
    a = ap.parse_args()
    smoke = a.smoke
    out_dir = CFG["smoke_output_dir"] if smoke else CFG["p1_output_dir"]
    train_file = CFG["smoke_train_file"] if smoke else CFG["p1_train_file"]
    val_file = None if smoke else CFG["p1_val_file"]
    if a.out_dir: out_dir = a.out_dir
    if a.train_file: train_file = a.train_file
    if a.val_file: val_file = a.val_file
    set_seed(CFG["seed"])
    os.makedirs(out_dir, exist_ok=True)
    snap = dict(CFG); snap["output_dir"] = out_dir; snap["train_file"] = train_file
    snap["val_file"] = val_file; snap["mode"] = "smoke" if smoke else "p1"
    if not smoke:
        snap["lora_r"] = CFG["fallback_r"]; snap["seq_len"] = 1024  # P1降级: OOM纠正
    with open(os.path.join(out_dir, "cfg_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print("[cfg] snapshot -> %s/cfg_snapshot.json" % out_dir)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except Exception as e:
        print("[error] 缺依赖, 先安装requirements: %s" % e)
        print("  /data1/paper_agent_venv/bin/pip install -r paper_agent/requirements.txt")
        return
    print("[model] load %s (4bit)" % CFG["base_model"])
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=CFG["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], quantization_config=bnb,
        device_map="auto", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_r = CFG["fallback_r"] if not smoke else CFG["lora_r"]  # P1用r32降显存
    seq_len = 1024 if not smoke else CFG["seq_len"]  # P1样本短(<1k token),1024省一半激活
    if a.from_adapter and os.path.exists(os.path.join(a.from_adapter, "adapter_config.json")):
        from peft import PeftModel  # 续训：从SFT adapter接着练（P2正确姿势），而非从零新建
        model = PeftModel.from_pretrained(model, a.from_adapter, is_trainable=True)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        print("[resume] SFT adapter attached+unfrozen: %s" % a.from_adapter)
    else:
        lora = LoraConfig(r=lora_r, lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
            target_modules=CFG["target_modules"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    def fmt(ex):
        ms = ex["messages"]
        t = "\n".join("%s: %s" % (m["role"], m["content"]) for m in ms)
        return {"text": t}
    train_ds = load_dataset("json", data_files=train_file)["train"]
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    eval_ds = None
    if val_file:
        eval_ds = load_dataset("json", data_files=val_file)["train"]
        eval_ds = eval_ds.map(fmt, remove_columns=eval_ds.column_names)
    print("[data] train=%d val=%s" % (len(train_ds), len(eval_ds) if eval_ds else None))
    lr = a.lr or CFG["lr"]
    epochs = a.epochs if a.epochs else (1 if smoke else CFG["epochs"])
    base_kwargs = dict(
        output_dir=out_dir, per_device_train_batch_size=CFG["per_device_bs"],
        per_device_eval_batch_size=CFG["per_device_bs"],
        gradient_accumulation_steps=CFG["grad_accum"], learning_rate=lr,
        lr_scheduler_type=CFG["scheduler"], warmup_steps=CFG["warmup_steps"],
        num_train_epochs=epochs,
        max_steps=10 if smoke else -1, max_length=CFG["seq_len"] if smoke else seq_len,
        logging_steps=5 if smoke else CFG["logging_steps"],
        save_steps=50 if smoke else CFG["eval_steps"],
        seed=CFG["seed"], report_to="none",
        save_total_limit=1 if smoke else CFG["save_total_limit"],
    )
    if not smoke:
        base_kwargs.update(dict(
            eval_steps=CFG["eval_steps"], save_strategy="steps",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False,
        ))
        try:
            args = SFTConfig(eval_strategy="steps", **base_kwargs)
        except TypeError:
            args = SFTConfig(evaluation_strategy="steps", **base_kwargs)
    else:
        args = SFTConfig(**base_kwargs)
    trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=args)
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("[done] adapter -> %s" % out_dir)

if __name__ == "__main__":
    main()
