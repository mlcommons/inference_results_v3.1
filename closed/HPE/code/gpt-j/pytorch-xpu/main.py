import arguments
import logging
import mlperf_loadgen as lg
import os
import sys

from pytorch_sut import OfflineSUT, ServerSUT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GPT-J")

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = arguments.parse_args()
    if args.scenario == "Server":
        sut = ServerSUT(
            args.model_path, args.dataset_path, args.dtype, args.device,
            args.num_workers, args.num_beams, args.scenario, args
        )
    else:
        sut = OfflineSUT(
            args.model_path, args.dataset_path, args.dtype, args.device,
            args.num_workers, args.num_beams, args.scenario, args
        )
    # set cfg
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)
    settings.mode = (
        lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    )
    # set log
    os.makedirs(args.log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True
    # run benchmark
    print("==> Running loadgen test")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings, args.audit_conf)
    print("Done!")


if __name__ == "__main__":
    main()
