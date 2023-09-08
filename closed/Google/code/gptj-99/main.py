r"""mlperf inference GPT-J benchmark.

Needs to be built with non-prod
blaze build -c opt //buildenv/target:non_prod \
    //third_party/mlperf/inference/gptj:main

blaze-bin/third_party/mlperf/inference/gptj/main --accuracy --logtostderr \
    --model_path=/sax/mrasquinha/gptj8bs32-int8wt --log_path=/tmp/logs/ \
    --gfs_user=tpu-perf-team \
    --census_enabled=false --max_examples 122 \
    --perf_examples=122 --batch_sz 8 &> /tmp/log
"""

import gc
import logging
import os

import time
from absl import app
from absl import flags
from google3.pyglib import gfile
from google3.pyglib import resources

from google3.third_party.mlperf.inference.gptj import backend
from google3.third_party.mlperf.inference.loadgen.bindings import mlperf_loadgen as lg

_DATASET_PATH = flags.DEFINE_string(
    "dataset_path",
    default="/cns/ik-d/home/mrasquinha/llm_6B/data/cnn_eval.json",
    help="path to the dataset")
_SCENARIO = flags.DEFINE_enum(
    "scenario", default="Offline",
    enum_values=["Server", "Offline"],
    help="benchmark scenario.")
_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    default="/sax/mrasquinha/test1",
    help="path to the sax admin server model.")
_NUM_CLIENT_THREADS = flags.DEFINE_integer(
    "num_client_threads", default=200, help="Number of client threads to use."
)
_ACCURACY = flags.DEFINE_bool(
    "accuracy", default=False, help="enable accuracy pass.")
_LOG_PATH = flags.DEFINE_string(
    "log_path", default="/tmp/data", help="path to the dataset.")
_LOG_INTERVAL = flags.DEFINE_integer(
    "log_interval", default=10, help="interval for logging.")
_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", default=13368,
    help="Max examples to run. For a full run this needs to be 24K in offline")
_PERF_EXAMPLES = flags.DEFINE_integer(
    "perf_examples", default=13368, help="target qps estimate")
_USER_CONF_OVERRIDE_PATH = flags.DEFINE_string(
    "user_conf_override_path",
    default="",
    help="When given overrides the default user.conf path.",
)
_SHORT_NAME = flags.DEFINE_string(
    "short_name", default="test", help="Experiment identifier.")
_SAVE_RESULT = flags.DEFINE_bool(
    "save_result", default=False, help="Preserve logs on cns.")

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}

_MLPERF_CONF_PATH = "google3/third_party/mlperf/inference/gptj/configs/mlperf.conf"
_USER_CONF = "google3/third_party/mlperf/inference/gptj/configs/user.conf"


def main(argv):
  del argv
  settings = lg.TestSettings()
  settings.scenario = scenario_map[_SCENARIO.value]

  mlperf_conf = resources.GetResourceFilename(_MLPERF_CONF_PATH)
  if _USER_CONF_OVERRIDE_PATH.value:
    user_conf = _USER_CONF_OVERRIDE_PATH.value
  else:
    user_conf = resources.GetResourceFilename(_USER_CONF)

  logging.info("Mlperf config: %s", mlperf_conf)
  logging.info("User config: %s", user_conf)

  settings.FromConfig(mlperf_conf, "gptj", _SCENARIO.value)
  settings.FromConfig(user_conf, "gptj", _SCENARIO.value)

  cfg = _SCENARIO.value
  if _ACCURACY.value:
    cfg = cfg + "_accuracy"
    settings.mode = lg.TestMode.AccuracyOnly
  else:
    cfg = cfg + "_performance"
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  if _SCENARIO.value == "Server":
    logging.info("Server mode run. See issue threads > 1")

  log_path = os.path.join(_LOG_PATH.value, cfg)
  if not gfile.Exists(log_path):
    gfile.MakeDirs(log_path)

  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = log_path
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  # Some thread is not terminating even after all the outputs are written
  log_settings.enable_trace = True

  saxml_output_log = os.path.join(_LOG_PATH.value, "sax_accuracy.json")
  sut = backend.get_sut(
      scenario=_SCENARIO.value,
      model_path=_MODEL_PATH.value,
      dataset_path=_DATASET_PATH.value,
      num_client_threads=_NUM_CLIENT_THREADS.value,
      max_examples=_MAX_EXAMPLES.value,
      perf_examples=_PERF_EXAMPLES.value,
      log_interval=_LOG_INTERVAL.value,
      log_path=saxml_output_log,
  )

  logging.info("Start Testing!")
  lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)
  logging.info("Test Done!")

  logging.info("Destroying SUT...")
  lg.DestroySUT(sut.sut)

  logging.info("Destroying QSL...")
  lg.DestroyQSL(sut.qsl)

  if _SAVE_RESULT.value:
    opath = os.path.join("/cns/rs-d/home/tpu-perf-team/gptj/logs/",
                         _SHORT_NAME.value + "_" +
                         time.strftime("%m%d_%H%M%S", time.gmtime()))
    gfile.RecursivelyCopyDir(_LOG_PATH.value, opath)
    logging.info("Output at %s", opath)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
