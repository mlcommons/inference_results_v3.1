import argparse
import re
import time
import json
import os
import pathlib
import torch
import types
from pathlib import Path
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
import transformers
import intel_extension_for_pytorch as ipex
from neural_compressor import PostTrainingQuantConfig, quantization

import numpy as np
from itertools import chain


from dataset import Dataset

calib_size = 1

torch._C._jit_set_texpr_fuser_enabled(False)
def quantize(args):

    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        return_dict=False
    )
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    user_model = ipex._optimize_transformers(
        user_model.eval(), dtype=torch.int8, inplace=True
    )

    calib_dataset = Dataset(dataset_path=args.cal_data_path,model_checkpoint_path=args.model,total_sample_count=args.calib_iters, pad_inputs=args.pad_inputs)
    calib_dataset.loadDataset()
    example_batch = calib_dataset[0]
    input_ids, past_key_values, position_ids, attention_mask = calib_dataset.collate_batch([example_batch])[0]
    example_inputs = (input_ids, attention_mask, position_ids, past_key_values)

    calib_dataloader=DataLoader(calib_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=calib_dataset.collate_batch
    )

    def calib_func(prepared_model):
        for i, (
            (input_ids, past_key_values, position_ids, attention_mask),
            last_ind,
        ) in enumerate(calib_dataloader):
            if i >= args.calib_iters:
                break
            prepared_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

    from neural_compressor import PostTrainingQuantConfig, quantization

    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["kl"],
            },
        },
    }

    excluded_precisions = []
    if args.sq:
        args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
        sq_recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha, "folding": True}}
        #sq_recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 'auto', 'folding': False }}
        conf = PostTrainingQuantConfig(
            backend="ipex",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            recipes=sq_recipes,
            example_inputs=example_inputs,
        )
    else:
        conf = PostTrainingQuantConfig(
            backend="ipex",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            example_inputs=example_inputs,
        )

    # save config
    user_model.config.save_pretrained(args.output_dir)
    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    q_model.save(args.output_dir)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--model", nargs="?", default="EleutherAI/gpt-j-6B", const="EleutherAI/gpt-j-6B"
	)
    parser.add_argument(
		"--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
	)
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument(
		"--max-new-tokens", default=32, type=int, help="output max new tokens"
	)
    parser.add_argument("--output_dir", nargs="?", default="./saved_results")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--sq", action="store_true")
    parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
    parser.add_argument(
		"--pad_max_length", default=512, type=int, help="Pad input ids to max length."
	)
    parser.add_argument("--calib_iters", default=512, type=int, help="calibration iters.")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument(
		"--int8_bf16_mixed",
		action="store_true",
		help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
	)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iters", default=100, type=int, help="num iter")
    parser.add_argument("--num_warmup", default=3, type=int, help="num warmup")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--cal-data-path", help="Path to calibration json file")
    parser.add_argument("--val-data-path", help="Path to validation json file")
    parser.add_argument("--beams", type=int, help="Number of beams for decoder", default=4)
    parser.add_argument("--warmup", action="store_true", help="Do warmup")
    parser.add_argument("--pad-inputs", action="store_true", help="Whether to pad input sequence")

    args = parser.parse_args()

    quantize(args)


if __name__=="__main__":

    main()
