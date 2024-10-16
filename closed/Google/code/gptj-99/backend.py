"""mlperf loadgen interface for GPTJ."""
import array
import concurrent.futures
import dataclasses
import json
import logging
from operator import itemgetter  # pylint: disable=g-importing-member
import time
from typing import List

import numpy as np
from saxml.client.python import sax

from google3.pyglib import gfile
from google3.third_party.mlperf.inference.gptj import dataset
from google3.third_party.mlperf.inference.loadgen.bindings import mlperf_loadgen as lg


@dataclasses.dataclass
class WarmupSample:
  id: int
  index: int


class ThreadedLMClient:
  """Holds a thread pool and a sax client for LM inference."""

  _thread_pool: concurrent.futures.ThreadPoolExecutor
  _sax_model: sax.Model
  _sax_language_model: sax.LanguageModel
  _dataset: dataset.Dataset
  _futures = List[concurrent.futures.Future]

  def __init__(
      self,
      num_threads: int,
      model_path: str,
      dataset_object: dataset.Dataset,
  ):
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(num_threads)
    self._sax_model = sax.Model(model_path)
    self._sax_language_model = self._sax_model.LM()
    self._dataset = dataset_object
    self._futures = []
    self.pred_outputs = {}
    self._resp_cnt = 0

  def _process_sample(self, sample, warmup):
    """Processes a single sample."""
    response = self._sax_language_model.Generate(
        self._dataset.inputs_str[sample.index]
    )
    if not warmup:
      pred_output_str = response[0][0]
      pred_output = np.fromstring(pred_output_str, dtype=int, sep=",")
      response_array = array.array("B", pred_output.tobytes())
      buffer_info = response_array.buffer_info()
      response = lg.QuerySampleResponse(
          sample.id, buffer_info[0], buffer_info[1]
      )
      lg.QuerySamplesComplete([response])
      self.pred_outputs[sample.index] = pred_output_str
      self._resp_cnt += 1

      if self._resp_cnt % 100 == 0:
        logging.info("Completed %d queries", self._resp_cnt)

  def process_single_sample_async(self, query_sample, warmup):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
        self._process_sample, query_sample, warmup
    )
    self._futures.append(future)

  def flush(self):
    concurrent.futures.wait(self._futures)
    self._futures = []


class SutBase:
  """Base SUT."""

  def __init__(
      self,
      model_path,
      dataset_path,
      num_client_threads,
      max_examples=None,
      perf_examples=None,
      log_interval=100,
      batch_size_exp=5,
      log_path=None,
  ):
    self._model_path = model_path
    self._dataset_path = dataset_path
    self._max_examples = max_examples
    self._perf_examples = perf_examples
    self._log_interval = log_interval
    self._batch_size_exp = batch_size_exp

    print("Loading Dataset ... ")
    self.dataset = dataset.Dataset(
        dataset_path=self._dataset_path,
        total_count_override=self._max_examples,
        perf_count_override=self._perf_examples,
    )

    print("Loading model ...")
    self._client = ThreadedLMClient(
        num_threads=num_client_threads,
        model_path=model_path,
        dataset_object=self.dataset,
    )

    self.qsl = lg.ConstructQSL(
        self.dataset.count,
        self.dataset.perf_count,
        self.dataset.LoadSamplesToRam,
        self.dataset.UnloadSamplesFromRam,
    )

    # We need to add some warmup to improve throughput estimation
    logging.info("Starting warmup....")
    # Warm up with exponentially increasing batch sizes up to 32.
    for batch_size_exp in range(self._batch_size_exp + 1):
      batch_size = 2**batch_size_exp
      for warmup_id, warmup_idx in enumerate(range(batch_size)):
        warmup_sample = WarmupSample(id=warmup_id, index=warmup_idx)
        self._client.process_single_sample_async(warmup_sample, True)
      self._client.flush()

    logging.info("Warmup done....")
    time.sleep(30)
    self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    # Its small enough to hold all responses in memory
    # Write out our own accuracy log in case loadgen has a corrupt version.
    if log_path is not None:
      self.accuracy_log = gfile.Open(log_path, "w")
    else:
      self.accuracy_log = None

  def issue_queries(self, query_samples):
    """Issue queries."""
    # With bucketization
    # # logging.info(">>>> Initial query order...")
    # # for x in query_samples:
    # #   in_tokens = self.dataset.inputs_str[x.index].split(",")
    # #   logging.info("%d -> %d", x.index, len(in_tokens))
    # # In offline mode we can sort the queries for input bucketization.
    if len(query_samples) > 1:
      sorted_samples = []
      for q in query_samples:
        sorted_samples.append(
            (len(self.dataset.inputs_str[q.index].split(",")), q))
      query_samples = [x[1] for x in sorted(sorted_samples, key=itemgetter(0))]
    # # logging.info(">>>> After reorder...")
    # # for x in query_samples:
    # #   in_tokens = self.dataset.inputs_str[x.index].split(",")
    # #   logging.info("%d -> %d", x.index, len(in_tokens))
    for query_sample in query_samples:
      self._client.process_single_sample_async(query_sample, False)

  def flush_queries(self):
    """Flush queries."""
    logging.info("Loadgen has completed issuing queries... ")
    self._client.flush()

    results = []
    for idx, x in self._client.pred_outputs.items():
      results.append({"qsl_idx": idx,
                      "data": x})
    if self.accuracy_log is not None and results:
      self.accuracy_log.write(json.dumps(results))
      self.accuracy_log.flush()
      self.accuracy_log.close()

  def __del__(self):
    print("Finished destroying SUT.")


def get_sut(
    scenario: str,
    model_path: str,
    dataset_path: str,
    num_client_threads: int,
    max_examples: int,
    perf_examples: int,
    log_interval: int,
    log_path: str = None,
):
  """Get SUT."""
  logging.info("Starting %s SUT.", scenario)
  return SutBase(
      model_path=model_path,
      dataset_path=dataset_path,
      num_client_threads=num_client_threads,
      max_examples=max_examples,
      perf_examples=perf_examples,
      log_interval=log_interval,
      log_path=log_path,
  )
