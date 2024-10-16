"""Tests backend threaded implementation."""

import dataclasses
import logging
import time
from unittest import mock
from google3.testing.pybase import googletest
from google3.third_party.mlperf.inference.gptj import backend


def mock_client_call(input_str: str):
  time.sleep(0.5)
  return [(input_str, 0.5)]


class MockDataset:
  inputs_str: dict[int, str] = {
      index: ",".join([str(index)] * (index + 1)) for index in range(10)
  }
  count: int = 10
  perf_count: int = 10
  # pylint: disable=invalid-name
  LoadSamplesToRam = None
  UnloadSamplesFromRam = None
  # pylint: enable=invalid-name


@dataclasses.dataclass
class MockSample:
  index: int
  id: int


TestSamples = [
    MockSample(id=mock_id, index=mock_idx)
    for mock_id, mock_idx in enumerate(range(10))
]


class BackendTest(googletest.TestCase):

  @mock.patch.object(backend.sax, "Model")
  @mock.patch.object(backend.lg, "QuerySamplesComplete")
  def test_threaded_sax_client(self, lg_mock, sax_model_mock):
    sax_model_mock.return_value.LM.return_value.Generate.side_effect = (
        mock_client_call
    )
    mock_dataset = MockDataset()
    client = backend.ThreadedLMClient(2, "/test/model/path", mock_dataset)
    # start-end should take about 2.5 seconds for 10 samples
    logging.info("start")
    for i in TestSamples:
      client.process_single_sample_async(i, False)
    client.flush()
    logging.info("end")
    self.assertEqual(10, lg_mock.call_count)

  @mock.patch.object(backend.sax, "Model")
  @mock.patch.object(backend.lg, "QuerySamplesComplete")
  @mock.patch.object(backend.dataset, "Dataset")
  def test_construct_sutbase(self, ds_mock, lg_mock, sax_model_mock):
    sax_model_mock.return_value.LM.return_value.Generate.side_effect = (
        mock_client_call
    )
    ds_mock.return_value = MockDataset()

    batch_size_exp = 3
    backend.SutBase(
        model_path="/test/model/path",
        dataset_path="/test/dataset/path",
        num_client_threads=2,
        batch_size_exp=batch_size_exp,
    )
    self.assertEqual(0, lg_mock.call_count)


if __name__ == "__main__":
  googletest.main()
