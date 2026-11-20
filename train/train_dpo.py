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


def pick_compute_dtype(capability, torch_mod):
    """按 GPU 计算能力选 4bit 计算的累积精度，返回 torch dtype。

    为什么不能写死 bf16（2026-11-19 云端实测）：
        Kaggle T4 是 Turing（(7,5)），**硬件不支持 bfloat16**。脚本原来写死
        `bnb_4bit_compute_dtype=torch.bfloat16`，训练第 2 步就
        `CUDA error: an illegal memory access was encountered` —— 不是慢，是崩。
        bf16 从 Ampere（8.0）才有；更老的卡保守用 fp16（V100/T4 都支持）。
    capability=None（查询失败）时同样退 fp16：宁可慢一点，不要非法访存。
    """
    if capability and capability[0] >= 8:
        return torch_mod.bfloat16
    # 2026-11-19 云端第二轮实测：Turing 上退 fp16 也不够。4bit gemm 走
    # `_dequant_linear_fallback` → `F.linear` → `cublasGemmEx` 时
    # `RuntimeError: CUBLAS_STATUS_NOT_SUPPORTED ... CUDA_R_16F ... compute_type`
    # —— T4/V100 上 fp16 的累加组合不被 cuBLAS 接受。宁可慢，必须 fp32。
    return torch_mod.float32


def load_plan(capability, force_quant=None):
    """决定「要不要 4bit 量化」以及用什么 dtype，返回 dict。

    为什么老卡必须绕开 bnb（2026-11-19 云端三次真跑）：
        Kaggle T4（Turing, (7,5)）上 4bit 量化 + LoRA 反传三次崩，而且三个 dtype 全崩：
          - bnb_4bit_compute_dtype=bfloat16 → `CUDA error: an illegal memory access`
          - 改 float16 → `CUBLAS_STATUS_NOT_SUPPORTED ... cublasGemmEx`
            （栈在 bitsandbytes/backends/cuda/ops.py 的 `_dequant_linear_fallback`）
          - 改 float32 → 仍然是 `illegal memory access`，这次崩在
            transformers/integrations/sdpa_attention.py 的 scaled_dot_product_attention
        共同项是 bnb 的 4bit 反传路径。所以 capability < 8 时**不量化**：
        4B 模型 fp16 权重 8GB，T4 有 16GB，LoRA + bs=1 + max_len=384 装得下。

    force_quant=True/False 供排查用；None = 按能力自动。
    """
    if force_quant is True:
        return {"use_4bit": True,
                "dtype": "bfloat16" if capability and capability[0] >= 8 else "float16",
                "reason": "显式要求 4bit"}
    if force_quant is False:
        return {"use_4bit": False,
                "dtype": "bfloat16" if capability and capability[0] >= 8 else "float16",
                "reason": "显式关闭 4bit"}
    if capability and capability[0] >= 8:
        return {"use_4bit": True, "dtype": "bfloat16",
                "reason": "Ampere+ 支持 bf16，4bit 省显存且已验证可用"}
    return {"use_4bit": False, "dtype": "float16",
            "reason": "老卡（capability<8）上 4bit 反传实测非法访存，改用非量化 fp16 权重"}


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
    ap.add_argument("--no-4bit", action="store_true",
                    help="强制不做 4bit 量化（老卡排查用；默认按 capability 自动决定）")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="强制 4bit 量化（默认按 capability 自动决定）")
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
    try:
        _cap = torch.cuda.get_device_capability()
    except Exception:
        _cap = None
    _force = None
    if a.no_4bit:
        _force = False
    elif a.four_bit:
        _force = True
    plan = load_plan(_cap, _force)
    _cdtype = pick_compute_dtype(_cap, torch)
    print("[gpu] capability=%s compute_dtype=%s" % (_cap, _cdtype))
    print("[plan] use_4bit=%s dtype=%s reason=%s" % (plan["use_4bit"], plan["dtype"], plan["reason"]))
    print("[model] load %s + SFT %s (%s)" % (CFG["base_model"], sft_ad,
          "4bit" if plan["use_4bit"] else plan["dtype"]))
    tok = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _max_mem = {0: "%dMiB" % (a.gpu_mem_mib or 4600), "cpu": "12GiB"}
    if plan["use_4bit"]:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=CFG["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=_cdtype)
        model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], quantization_config=bnb,
            device_map="auto", trust_remote_code=True, max_memory=_max_mem)
        model.gradient_checkpointing_enable()
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        # 非量化路径：fp16/bf16 权重直接上卡（老卡 4bit 反传不可用，见 load_plan 注释）
        _wdt = torch.bfloat16 if plan["dtype"] == "bfloat16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], torch_dtype=_wdt,
            device_map="auto", trust_remote_code=True, max_memory=_max_mem)
        model.gradient_checkpointing_enable()
        # PEFT + gradient_checkpointing 的硬要求：frozen 基座的输出不 requires_grad 时，
        # checkpointed block 反向不重算、检查点**静默失效**（v9 只改 use_reentrant 一点没省）。
        # 必须让输入 requires_grad，激活才会真正分段重算。
        model.enable_input_require_grads()
        model.config.use_cache = False
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
        # PEFT 下 ref_model=None（TRL dpo_trainer.py:920），不会另载 ref 权重；
        # True 让 ref logps 在训练前 no_grad 缓存进 dataset（:1141-1215），
        # 训练步直接读（:1402），不再叠第二组 chosen/rejected 前向的激活。
        # 旧注释「True 会另载一份 ref 模型，双倍权重必爆」是错的（v8/v9 OOM 根因）。
        precompute_ref_log_probs=True,
        logging_steps=CFG["logging_steps"], save_steps=CFG["save_steps"],
        seed=CFG["seed"], report_to="none",
        save_total_limit=CFG["save_total_limit"], gradient_checkpointing=True,
        # 非重入检查点：重入版在反向时重算，会把每层输入整段留在 autograd 图里；
        # T4 非量化路径 DPO 四次前向的激活本来就把 14.5GB 塞满（v8 实测 OOM）。
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
