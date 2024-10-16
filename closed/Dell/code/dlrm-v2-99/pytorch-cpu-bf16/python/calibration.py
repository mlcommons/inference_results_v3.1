"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from multiprocessing import JoinableQueue

import mlperf_loadgen as lg
import numpy as np
import torch

import dataset
import multihot_criteo
from backend_pytorch_native import get_backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "debug":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
    "multihot-criteo-sample":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
    "multihot-criteo":
        (multihot_criteo.MultihotCriteo, multihot_criteo.pre_process_criteo_dlrm, multihot_criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-debug-pytorch": {
        "dataset": "debug",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-multihot-sample-pytorch": {
        "dataset": "multihot-criteo-sample",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-multihot-pytorch": {
        "dataset": "multihot-criteo",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    }
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument("--samples-per-query-multistream", default=8, type=int, help="query length for multi-stream scenario (in terms of aggregated samples)")
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument("--samples-per-query-offline", type=int, default=2048, help="query length for offline scenario (in terms of aggregated samples)")
    parser.add_argument("--samples-to-aggregate-fix", type=int, help="number of samples to be treated as one")
    parser.add_argument("--samples-to-aggregate-min", type=int, help="min number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-max", type=int, help="max number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-quantile-file", type=str, help="distribution quantile used to generate number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-trace-file", type=str, default="dlrm_trace_of_aggregated_samples.txt")
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--calibration", action="store_true", help="Whether calibration only for this run.")
    parser.add_argument("--int8-configure-dir", type=str, default="./int8_configure.json", help="int8 recipe location")
    parser.add_argument("--use-int8", action="store_true", default=False)
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def main():
    args = get_args()

    backend = get_backend(args.backend, args.dataset, args.use_gpu, debug=args.debug)
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]

    # --count-samples can be used to limit the number of samples used for testing
    ds = wanted_dataset(num_embeddings_per_feature=[40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,3],
                        data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=pre_proc,  # currently an identity function
                        count=args.count_samples,
                        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
                        samples_to_aggregate_min=args.samples_to_aggregate_min,
                        samples_to_aggregate_max=args.samples_to_aggregate_max,
                        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
                        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
                        max_ind_range=args.max_ind_range,
                        **kwargs)
    # load model to backend
    model = backend.load(args, ds)
    print(model)
    print('model load done')
    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    if args.calibration:
        print("Calibration done")
        # return
    # #
    # # make one pass over the dataset to validate accuracy
    # #
    print('load dataset')
    count = ds.get_item_count()
    # # warmup
    import intel_extension_for_pytorch as ipex
    results = np.zeros(count).astype(np.float32)
    targets = np.zeros(count).astype(np.float32)
    for i in range(0, count, args.max_batchsize):
        sample_s = i
        sample_e = min(i + args.max_batchsize, count)
        ds.load_query_samples(range(sample_s, sample_e))
        batch = ds.test_data.load_batch(range(sample_s, sample_e))
        r = backend.predict(batch)
        results[sample_s:sample_e] = r.detach().cpu().numpy()
        targets[sample_s:sample_e] = batch.labels.detach().cpu().float().numpy()
        # results.append(r[0].detach().cpu().numpy())
        # targets.append(batch.labels.detach().cpu().float().numpy())
        res_np = results[0:sample_e].copy()
        tgt_np = targets[0:sample_e].copy()
        roc_auc = ipex._C.roc_auc_score(torch.tensor(tgt_np).reshape(-1), torch.tensor(res_np).reshape(-1))
        print(f'roc_auc of {i}', roc_auc)

if __name__ == "__main__":
    main()
