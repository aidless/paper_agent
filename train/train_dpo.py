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


def attach_trainable_adapter(model, sft_ad, *, peft_mod, lora_config_cls,
                             lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """保证「模型上确实挂着可训练 adapter」，返回 (model, mode)。

    为什么必须有这一步（2026-11-19 云端实测）：
        4bit 量化基座自己没有任何可训练参数。旧实现在 SFT adapter 缺失时只打一行
        `[warn] SFT adapter缺失, 从基座直做DPO` 就继续，于是 DPOTrainer 在跑起来
        130 秒后抛 `ValueError: You cannot perform fine-tuning on purely quantized
        models. Please attach trainable adapters on top of the quantized model ...`。
        「知道条件不满足还继续」= 假绿，和 `audit` 把「无法核对」当「不一致」是同一类病。

    两条路径：
        - 有 SFT adapter（目录里有 adapter_config.json）→ 续训它（resume_sft），
          并显式解冻 lora_ 参数（量化基座 + adapter 载入后默认仍是冻结的）。
        - 没有 → 现建一个 LoRA（fresh_lora），target_modules 覆盖 7 个投影层。
    """
    ad_cfg = os.path.join(sft_ad, "adapter_config.json") if sft_ad else ""
    if ad_cfg and os.path.exists(ad_cfg):
        model = peft_mod.PeftModel.from_pretrained(model, sft_ad, is_trainable=True)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        return model, "resume_sft"
    lora_cfg = lora_config_cls(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                               target_modules=list(CFG["target_modules"]),
                               task_type="CAUSAL_LM")
    return peft_mod.get_peft_model(model, lora_cfg), "fresh_lora"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gpu-mem-mib", type=int, default=None,
                    help="显存上限（MiB）。默认 4600 是本地 6GB 卡的保守值；云端 >=24GB 卡请给 22000 左右，"
                         "否则权重仍会被挤到 CPU，慢十倍（2026-11-19 加，云端 DPO 用）")
    a = ap.parse_args()
    smoke = a.smoke
    out_dir = CFG["smoke_output_dir"] if smoke else CFG["p2_output_dir"]
    train_file = CFG["dpo_train_file"]
    sft_ad = CFG["sft_adapter"]
    if a.out_dir: out_dir = a.out_dir
    if a.train_file: train_file = a.train_file
    if a.sft_adapter: sft_ad = a.sft_adapter
    set_seed(CFG["seed"])
    # 云端（DPO_HF_ONLINE=1）要现下模型；本地默认断网隔离。先算，snapshot 要记这一项。
    online = os.environ.get("DPO_HF_ONLINE") == "1"
    os.makedirs(out_dir, exist_ok=True)
    snap = dict(CFG); snap.update({"output_dir": out_dir, "train_file": train_file,
        "sft_adapter": sft_ad, "mode": "smoke" if smoke else "p2",
        "gpu_mem_mib": a.gpu_mem_mib or 4600, "hf_online": online})
    with open(os.path.join(out_dir, "cfg_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print("[cfg] snapshot -> %s/cfg_snapshot.json" % out_dir)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import peft
        from trl import DPOTrainer, DPOConfig
        from datasets import load_dataset
    except Exception as e:
        print("[error] 缺依赖: %s" % e)
        return
    # 本地跑：断网 + 去掉代理，避免半路去连 HF。云端（Kaggle 等）要现下模型，
    # 用 DPO_HF_ONLINE=1 关掉这套隔离，否则会遇到 "offline mode is enabled" 直接死。
    if online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        print("[net] DPO_HF_ONLINE=1 → 允许 HF 下载（云端模式）")
    else:
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
        max_memory={0: "%dMiB" % (a.gpu_mem_mib or 4600), "cpu": "12GiB"})  # 默认值对齐 6GB 卡（溢出到 CPU，慢但不断）
    model.gradient_checkpointing_enable()
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    model, mode = attach_trainable_adapter(model, sft_ad, peft_mod=peft, lora_config_cls=peft.LoraConfig)
    print("[adapter] mode=%s (sft_adapter=%s)" % (mode, sft_ad))
    n_train = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("挂完 adapter 仍无任何可训练参数 —— 量化基座下 DPOTrainer 一定拒跑")
    print("[adapter] trainable tensors=%d" % n_train)
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
