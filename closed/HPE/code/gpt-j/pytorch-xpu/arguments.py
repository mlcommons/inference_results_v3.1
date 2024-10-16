import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", type=str, choices=["SingleStream", "Offline", "Server"],
                        default="Offline", help="Scenario")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--dataset_path", type=str, default="./data/cnn_eval.json")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--mlperf_conf", default="./configs/mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="./configs/user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--audit_conf", default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy evaluation")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "int4"],
                        help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--device", type=str, choices=["cpu", "xpu", "cuda"],
                        help="cpu, xpu or cuda", default="cpu")
    parser.add_argument("--warmup", action="store_true", help="Enable warmup")
    parser.add_argument("--warmup_path", type=str, default="./data/cnn_eval_warmup.json")
    parser.add_argument("--profile", action="store_true", help="Enable profile")
    parser.add_argument("--dynamic_batching", action="store_true", help="Set true to enable dynamic batching")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=4, help="beam width for BeamSearch")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--start_rank", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to inference (-1 for full dataset, full mlperf dataset contains 13368 samples)")
    parser.add_argument("--optimize_transformers", action="store_true")
    parser.add_argument("--sort", action="store_true", help="Enable sorting by input token length")
    parser.add_argument("--verbose", action="store_true", help="Set true to dump i/o")
    parser.add_argument("--pick_index", type=str, default=None,
                        help="index of prompts for picking up, e.g.(3 samples): --pick_index 1,2,3 or --pick_index 0-2")
    parser.add_argument("--repeat", type=int, default=1, help="times of repeating picked samples")
    parser.add_argument("--ref_log_path", type=str, default="./data/out_prompts_a100_fp16.json",
                        help="reference prompts")
    parser.add_argument("--padding_side", default="left", choices=["left", "right"],
                        help="Set padding side, left or right")
    args = parser.parse_args()
    print(args)
    return args
