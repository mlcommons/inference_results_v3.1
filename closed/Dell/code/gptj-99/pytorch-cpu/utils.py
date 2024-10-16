from argparse import ArgumentParser


def getArgs():

    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--workload-name", help="Name of workload", required=True)
    parser.add_argument("--scenario", choices=["Offline", "Server", "SingleStream"], help="MLPerf scenario to run", default="Offline")
    parser.add_argument("--mlperf-conf", help="Path to mlperf.conf file")
    parser.add_argument("--user-conf", help="Path to user.conf file containing overridden workload params")
    parser.add_argument("--mode", choices=["Accuracy", "Performance"], help="MLPerf mode to run", default="Performance")
    parser.add_argument("--num-proc", type=int, help="Number of instances/consumers", default=2)
    parser.add_argument("--cpus-per-proc", type=int, help="Number of cores per instance", default=8)
    parser.add_argument("--warmup", action="store_true", help="Whether to do warmup")
    parser.add_argument("--precision", choices=["int8", "bf16", "fp32", "mix", "int4_bf16_mixed"], help="Model precision to run", default="fp32")
    parser.add_argument("--workers-per-instance", type=int, help="Number of workers per each instance/consumer", default = 1)
    parser.add_argument("--cores-offset", type=int, help="Cpus to offset on 1st socket", default=0)
    parser.add_argument("--dataset-path", type=str, help="Path to dataset", required=True)
    parser.add_argument("--model-checkpoint-path", type=str, help="Path to fp32 checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument("--workers-per-proc", type=int, help="Number of worker threads per process", default=1)
    parser.add_argument("--total-sample-count", type=int, help="Total number of samples to sample from", default=1000)
    parser.add_argument("--output-dir", type=str, help="Output directory for mlperf logs", default="output_logs")
    #parser.add_argument("--precision", choices=["int8", "fp32"], default="fp32", help="Precision to run")
    parser.add_argument("--quantized-model", help="Path to quantized model")
    parser.add_argument("--pad-inputs", action="store_true", help="Whether to pad inputs")
    args = parser.parse_args()
    
    return args
