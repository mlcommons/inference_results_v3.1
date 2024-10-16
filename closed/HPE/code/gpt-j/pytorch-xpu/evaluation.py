import argparse
import evaluate
import json
import nltk
import numpy as np
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import Dataset
from utils import logger

mlperf_rouge = {
    "rouge1" : 42.9865,
    "rouge2" : 20.1235,
    "rougeL" : 29.9881,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf_accuracy_file", type=str, default="./logs/Offline/mlperf_log_accuracy.json",
                        help="Path to mlperf_log_accuracy.json")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--dataset_path", type=str, default="./data/cnn_eval.json")
    parser.add_argument("--output_path", type=str, default="./prompts", help="output directory for picked prompts")
    parser.add_argument("--pick_index", type=str, default=None,
                        help="index of prompts for picking up, e.g.(3 samples): --pick_index 1,2,3 or --pick_index 0-2")
    parser.add_argument("--skip_pick_result", action="store_true", help="Set to skip picking results by pick_index")
    parser.add_argument("--dump_input", action="store_true", help="Set to dump with input prompts to get a sub-dataset")
    parser.add_argument("--evaluate", action="store_true", help="Set to do Rouge evaluation")
    parser.add_argument("--sort_results", action="store_true", help="Sort results by index in raw dataset")
    parser.add_argument("--verbose", action="store_true", help="Set true to compare accuracy & ratio w/ A100 FP16")
    parser.add_argument("--ref_log_path", type=str, default="./data/out_prompts_a100_fp16.json",
                        help="reference prompts")
    parser.add_argument("--dtype", default="int64",
                        help="dtype of the accuracy log", choices=["int32", "int64"])
    args = parser.parse_args()
    return args

def bytes2ndarray(bt, dtype=np.int64):
    arr = np.frombuffer(bytes.fromhex(bt), dtype)
    return arr

def bytes2tensor(bt, dtype=np.int64):
    arr = bytes2ndarray(bt, dtype)
    tensor = torch.from_numpy(arr)
    return tensor

def bytes2prompt(bt, tokenizer, dtype=np.int64):
    arr = bytes2ndarray(bt, dtype)
    prompt = ndarray2prompt(arr, tokenizer)
    return prompt

def ndarray2prompt(arr, tokenizer, skip_special_tokens=True):
    try:
        prompt = tokenizer.decode(arr, skip_special_tokens=skip_special_tokens)
    except Exception as e:
        logger.error(f"Notice: {str(e)}!")
        return None
    return prompt

def process_prompt(prompt):
    # rougeLSum expects newline after each sentence
    prompt_processed = "\n".join(nltk.sent_tokenize(prompt.strip()))
    return prompt_processed

def batch_process_prompts(prompt_dict, filter_idx=[]):
    # rougeLSum expects newline after each sentence
    prompt_processed = []
    for idx, prompt in prompt_dict.items():
        if idx not in filter_idx:
            if prompt != None:
                prompt_processed.append(process_prompt(prompt))
            else:
                filter_idx.append(idx)
    return prompt_processed, filter_idx

def dump_prompts(output_path, out_prompts, in_prompts=None):
    prompts = []
    for idx in out_prompts.keys():
        prompt = {}
        if in_prompts != None:
            prompt["input"] = in_prompts[idx]
            prompt["instruction"] = "Summarize the following news article:"
        prompt["output"] = out_prompts[idx]
        prompts.append(prompt)
    res = json.dumps(prompts, indent=4)
    if output_path != None:
        with open(output_path, "w") as f:
            f.write(res)

def evaluation(out_prompts, ref_prompts, msg="", filter_idx=[]):
    preds, filter_idx = batch_process_prompts(out_prompts, filter_idx)
    refs, _ = batch_process_prompts(ref_prompts, filter_idx)
    metric = evaluate.load("rouge")
    pred_rouge = metric.compute(predictions=preds, references=refs, use_stemmer=True, use_aggregator=False)
    pred_rouge = {k: round(np.mean(v) * 100, 4) for k, v in pred_rouge.items()}
    pred_lens = [len(pred) for pred in preds]
    pred_rouge["gen_len"] = np.sum(pred_lens)
    pred_rouge["gen_num"] = len(preds)
    print(f"{msg} Accuracy\n{pred_rouge}")
    ratio = {f"{k} ratio": round(float(pred_rouge[k])/float(mlperf_rouge[k]) * 100, 2) for k in mlperf_rouge.keys()}
    print(f"{msg} / MLPerf Accuracy Ratios(%)\n{ratio}")
    if len(filter_idx) != 0:
        logger.error(f"NOTICE: evaluate accuracy w/o samples {filter_idx} due to finding NoneType in these predict prompts!")
    return filter_idx

def main():
    args = get_args()
    print(args)
    eval_dtype = np.int64
    if args.dtype == "int32":
        eval_dtype = np.int32

    nltk.download("punkt")  # one-shot

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # Deduplicate the results loaded from the json
    dedup_results = []
    seen = set()
    for result in results:
        item = result['qsl_idx']
        if item not in seen:
            seen.add(item)
            dedup_results.append(result)
    results = dedup_results

    if args.sort_results:
        results = sorted(results, key=lambda s: s.__getitem__("qsl_idx"), reverse=False)

    if args.pick_index != None:
        if "-" in args.pick_index:  # pick by range
            pick_range = args.pick_index.split("-")
            pick_index = [*range(int(pick_range[0]), int(pick_range[1])+1, 1)]
        else:  # pick by #
            pick_index = [int(idx) for idx in args.pick_index.split(",")]
    else:  # pick all
        pick_index = [result["qsl_idx"] for result in results]

    dataset = Dataset(args.model_path, args.dataset_path, pick_index=args.pick_index)
    tgt_prompts = {idx : dataset.targets[idx] for idx in pick_index}

    pred_prompts = {result["qsl_idx"] : bytes2prompt(result["data"], dataset.tokenizer, eval_dtype) for result in results}
    if not args.skip_pick_result:
        logger.debug(f"==> Picking {len(pick_index)} prompts from predict result by idx: {pick_index}")
        pred_prompts = {idx : pred_prompts[idx] for idx in pick_index}

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        if args.dump_input:
            in_prompts = {idx : dataset.list_data_dict[idx]["input"] for idx in pick_index}
        else:
            in_prompts = None

        dump_path = f"{args.output_path}/prediction.json"
        dump_prompts(dump_path, pred_prompts, in_prompts)
        logger.debug(f"==> Saved prediction prompts to {dump_path}")

        dump_path = f"{args.output_path}/reference.json"
        dump_prompts(dump_path, tgt_prompts, in_prompts)
        logger.debug(f"==> Saved target prompts to {dump_path}")

    if args.evaluate:
        print(f"MLPerf Accuracy\n{mlperf_rouge}")
        filter_idx = []
        evaluation(pred_prompts, tgt_prompts, msg="Predict", filter_idx=filter_idx)

        if args.verbose:
            with open(args.ref_log_path, "r") as f:
                ref_prompts = json.load(f)
            logger.debug(f"==> Picking {len(pick_index)} prompts from ref result by idx: {pick_index}")
            ref_prompts = {idx: ref_prompts[idx]["output"] for idx in pick_index}
            evaluation(ref_prompts, tgt_prompts, msg="Reference", filter_idx=filter_idx)

if __name__ == "__main__":
    main()
