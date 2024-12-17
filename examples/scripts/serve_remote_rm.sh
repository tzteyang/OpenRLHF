set -x

CUDA_VISIBLE_DEVICES=0 python -m openrlhf.cli.serve_rm \
    --reward_pretrain /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --port 6006 \
    --bf16 \
    --flash_attn \
    --max_len 8192 \
    --batch_size 4