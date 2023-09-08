import os

import mlperf_loadgen as lg

import logging
from SUTOffline import SUT as SUTOffline
from SUTServer import SUT as SUTServer

from utils import getArgs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GPT-J")

SCENARIO_MAP = {
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

PAD_VALUE=1
PAD_MAX=196

def response_loadgen(out_queue):

    while True:
        next_task = out_queue.get()
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result
        array_type_code = next_task.array_type_code

        batch_size = len(query_id_list)

        for id, out in zip(query_id_list, result):
            response_array = array.array(array_type_code, out)
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
            lg.QuerySamplesComplete(responses)

def main():
    args = getArgs()

    settings = lg.TestSettings()
    scenario = args.scenario

    settings.scenario = SCENARIO_MAP[args.scenario.lower()]
    settings.FromConfig(args.mlperf_conf, args.workload_name, args.scenario)
    settings.FromConfig(args.user_conf, args.workload_name, args.scenario)

    settings.mode = lg.TestMode.AccuracyOnly if args.mode.lower()=="accuracy" else lg.TestMode.PerformanceOnly

    if args.scenario.lower() == "server" and args.use_dynamic_batching:
        sut = SUTServer(args.num_proc, args.cpus_per_proc, args.model_checkpoint_path, initial_core=args.cores_offset, batch_size=args.batch_size, dataset_path=args.dataset_path, 
            workers_per_proc=args.workers_per_proc, warmup=args.warmup, precision=args.precision, quantized_model=args.quantized_model, total_sample_count=args.total_sample_count, pad_inputs=args.pad_inputs, hp_threshold=args.hp_threshold, max_dynamic_batch_size=args.max_dynamic_batch_size, numa_offset=args.numa_offset)
    else:
        sut = SUTOffline(args.num_proc, args.cpus_per_proc, args.model_checkpoint_path, initial_core=args.cores_offset, batch_size=args.batch_size, dataset_path=args.dataset_path, 
            workers_per_proc=args.workers_per_proc, warmup=args.warmup, precision=args.precision, quantized_model=args.quantized_model, total_sample_count=args.total_sample_count, pad_inputs=args.pad_inputs)

    # Start SUT
    sut.startSUT()

    # Create SUT, QSL Trampoline
    lg_sut = lg.ConstructSUT(sut.issueQueries, sut.flushQueries)
    lg_qsl = lg.ConstructQSL(args.total_sample_count, args.total_sample_count, sut.loadSamplesToRam, sut.unloadSamplesFromRam)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    # Start loadgen test
    log.info("Starting {}-{} Test".format(args.scenario, args.mode))
    lg.StartTestWithLogSettings(lg_sut, lg_qsl, settings, log_settings)

    log.info("Test completed")
    # Stop SUT
    sut.stopSUT()

    lg.DestroyQSL(lg_qsl)
    lg.DestroySUT(lg_sut)

if __name__=="__main__":
    main()
