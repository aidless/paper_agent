# Kaggle kernel：4B QLoRA DPO（SFT adapter 之上继续 DPO）
# 由 paper_agent/cloud/kaggle-dpo/run.sh --push 提交；本文件也可在 Kaggle 网页里直接粘贴运行。
#
# 为什么是这套参数（都有实测出处，不是拍的）：
# - bs=1 / max_len=384 / ref_model=None：偏好对实测最长 363 token，384 零截断；
#   ref_model=None 时 TRL 走「关掉 adapter 共享基座」而不是再载一份权重 —— 双份权重在
#   6GB 上必爆（paper_agent/reports/train_dpo_smoke.log）。
# - --gpu-mem-mib 14000：T4 16GB 留 2GB 给 CUDA context 与 tokenizer/数据搬运；
#   脚本默认 4600 是本地 6GB 卡的保守值，不覆盖就会把权重挤到 CPU、慢十倍。
# - DPO_HF_ONLINE=1：云端要现下 Qwen3-4B，脚本默认断网隔离会直接死。
import json
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
DS = "/kaggle/input/dsh-dpo-p0v2"  # run.sh 上传的数据集挂载点
os.makedirs(WORK, exist_ok=True)

print("== 环境 ==")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)

print("== 依赖 ==")
# 冻结清单里有本地 CUDA 版本号，云端按同名安装；失败就让 pip 自己解
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers==4.57.6", "trl==1.13.0", "peft==0.20.0",
                "bitsandbytes==0.50.2", "datasets==4.8.5", "accelerate==1.15.0"], check=True)

print("== 数据与脚本 ==")
for name in ("dpo_p0v2.jsonl", "train_dpo.py"):
    src = os.path.join(DS, name)
    if not os.path.exists(src):
        raise SystemExit("数据集里缺 %s（检查 run.sh 的 --push 是否成功）" % name)
    shutil.copy(src, WORK)
data = os.path.join(WORK, "dpo_p0v2.jsonl")
print("偏好对行数：", sum(1 for _ in open(data, encoding="utf-8")))

print("== 训练 ==")
env = dict(os.environ, DPO_HF_ONLINE="1", TOKENIZERS_PARALLELISM="false")
cmd = [sys.executable, os.path.join(WORK, "train_dpo.py"),
       "--train-file", data,
       "--out-dir", os.path.join(WORK, "tmlr-qwen3-4b-p2"),
       "--gpu-mem-mib", "14000"]
print("$", " ".join(cmd))
rc = subprocess.run(cmd, env=env).returncode
print("== 训练退出码 ==", rc)

# 产物清单：让拉回后能一眼看出跑没跑成（退出码 0 不等于产物齐）
out = os.path.join(WORK, "tmlr-qwen3-4b-p2")
manifest = {"exit_code": rc, "out_dir": out, "files": []}
if os.path.isdir(out):
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        manifest["files"].append({"name": f, "bytes": os.path.getsize(p)})
with open(os.path.join(WORK, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=1)
print(json.dumps(manifest, ensure_ascii=False, indent=1))
