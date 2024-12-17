set -x

DATASET=$1
POLICY_MODEL_NAME=Skywork-o1-Open-Llama-3.1-8B
REWARD_MODEL_NAME=Skywork-o1-Open-PRM-Qwen-2.5-7B
SAVE_NAME=MODEL_NAME-${DATASET}

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /root/autodl-tmp/models/${POLICY_MODEL_NAME} \
   --reward_pretrain /root/autodl-tmp/models/${REWARD_MODEL_NAME} \
   --save_path /root/autodl-tmp/models/${SAVE_NAME} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --num_episodes 3 \
   --max_epochs 3 \
   --prompt_max_len 512 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --lora_rank 8 \
   --target_modules all-linear \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /root/autodl-tmp/datasets/L3_MCQ_16_task_dataset/dataset/${DATASET} \
   --input_key input_content \
   --apply_chat_template \
   --max_samples 1000 \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb 2e27e60a3903753a3f996ae442f6c69c02f22b6c \
   --remote_rm_url http://localhost:5000/get_reward
EOF

    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
