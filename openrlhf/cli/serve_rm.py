import argparse
import re

import torch
import uvicorn
import numpy as np
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

from openrlhf.utils.skywork_o1_model_utils.prm_model import PRM_MODEL
from openrlhf.utils.skywork_o1_model_utils.io_utils import (
    prepare_input,
    prepare_batch_input_for_model,
    derive_step_rewards
)

logger = init_logger(__name__)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


CHAT_TEMPLATE = {
    "Llama": {
        "one_turn": r"<\|start_header_id\|>user<\|end_header_id\|>\n(.*?)<\|eot_id\|>.*?<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>",
        "query_role": "user",
        "response_role": "assistant",
        "end_of_turn": "<|eot_id|>",
    },
    "Qwen": {
        "oue_turn": r"<\|im_start\|>user\n(.*?)<\|im_end\|>.*?<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
        "query_role": "user",
        "response_role": "assistant",
        "end_of_turn": "<|im_end|>",
    }
}

@dataclass
class RewardStrategy(Enum):
    MEAN = "mean"
    MIN = "min"
    LAST = "last"

class SkyworkO1PRM(RewardModelProxy):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True)
        self.reward_model = PRM_MODEL.from_pretrained(
            args.reward_pretrain,
            _attn_implementation="flash_attention_2" if args.flash_attn else "eager",
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            device_map="auto"
        ).eval()

        self.max_length = args.max_len
        self.batch_size = args.batch_size
        
        self.reward_assign_strategy = RewardStrategy(args.reward_assign_strategy)

    def assign_rewards(self, step_rewards: List[List[int]], strategy: RewardStrategy) -> List[int]:
        if strategy is RewardStrategy.MEAN:
            outcome_rewards = [np.mean(rewards) for rewards in step_rewards]
        elif strategy is RewardStrategy.MIN:
            outcome_rewards = [np.min(rewards) for rewards in step_rewards]
        elif strategy is RewardStrategy.LAST:
            outcome_rewards = [rewards[-1] for rewards in step_rewards]
        else:
            raise ValueError("Invalid reward strategy")
        return outcome_rewards

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        chat_style = None
        for template_name, template_style in CHAT_TEMPLATE.items():
            chat_pattern = re.compile(template_style["one_turn"], re.DOTALL)
            chat_matches = chat_pattern.findall(queries[0])
            if chat_matches:
                chat_style = template_name
                break
        
        if chat_style is None:
            raise ValueError("Chat style not found in the query")
        format_samples = []
        for raw_text in queries:
            chat_pattern = re.compile(CHAT_TEMPLATE[chat_style]["one_turn"], re.DOTALL)
            query, response = chat_pattern.findall(raw_text)[0]
            format_samples.append((query, response))

        processed_data = [prepare_input(sample[0], sample[1], tokenizer=self.tokenizer, step_token="\n\n") for sample in format_samples]

        scores = []
        device = self.reward_model.v_head.summary.weight.device
        with torch.no_grad():
            for i in range(0, len(processed_data), batch_size):
                input_ids, steps, reward_flags = zip(*processed_data[i: i+batch_size])
                input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, self.tokenizer.pad_token_id)
                _, _, rewards = self.reward_model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    return_probs=True
                )
                step_rewards = derive_step_rewards(rewards, reward_flags)
                outcome_rewards = self.assign_rewards(step_rewards, strategy=self.reward_assign_strategy)

                scores.extend(outcome_rewards)
        
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    # Process-Supervised Reward Model
    parser.add_argument("--reward_assign_strategy", choices=["mean", "min", "last"], default="mean")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    reward_model = SkyworkO1PRM(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries=queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
