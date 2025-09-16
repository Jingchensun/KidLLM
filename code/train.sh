#!/usr/bin/env bash
# set -euo pipefail

# #!/usr/bin/env bash
# export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
# export TORCH_EXTENSIONS_DIR="/tmp/$USER/torch_ext"
# mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# —— 指定算力架构：A100 用 8.0；若是 3090/4090 改成 "8.6" / "8.9" ——
# export TORCH_CUDA_ARCH_LIST="8.0"

# （可选）提高文件句柄上限，防止 DataLoader 打开大量文件时报错
#ulimit -n 65536 || true

# —— 启动 DeepSpeed ——
deepspeed \
  --include localhost:0,1,2,3 \
  --master_addr 127.0.0.1 \
  --master_port 28458 \
  train_sft.py \
  --model openllama_peft \
  --stage 1 \
  --data_path ../dataset/output/merged_train.json \
  --test_data_path ../dataset/output/merged_val.json \
  --image_root_path ../data \
  --vicuna_hf_repo jsun39/kidspeak_vicuna \
  --whisper_pretrained small \
  --max_tgt_len 256 \
  --save_path ../checkpoints/kidspeak_small/check/ \
  --log_path ../checkpoints/kidspeak_small/log/
