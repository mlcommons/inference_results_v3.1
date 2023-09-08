"""GPT-J dataset processing."""

import json
import os

from tensorflow.io import gfile
import transformers

from google3.pyglib import resources


_TOKENIZER_PATH = "google3/third_party/mlperf/inference/gptj/tokenizer"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input"
        " that provides further context. Write a response that appropriately"
        " completes the request.\n\n### Instruction:\n{instruction}\n\n###"
        " Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def jload(f, mode="r"):
  """Load a .json file into a dictionary."""
  f = gfile.GFile(f, mode)
  jdict = json.load(f)
  f.close()
  return jdict


class Dataset:
  """GPT-J dataset processing."""

  def __init__(
      self,
      dataset_path,
      batch_size=1,
      pad_val=1,
      pad_max=196,
      total_count_override=None,
      perf_count_override=None,
      encode_targets=True
  ):
    print("Constructing QSL")

    self.dataset = "cnn_dailymail"
    self.dataset_path = dataset_path
    self.batch_size = batch_size
    self.pad_val = pad_val
    self.pad_max = pad_max

    main_extracted_dir = resources.GetARootDirWithAllResources()
    tokenizer_path = os.path.join(main_extracted_dir, _TOKENIZER_PATH)
    self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        tokenizer_path, model_max_length=2048, padding_side="left",
        use_fast=False,)
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.list_data_dict = jload(self.dataset_path)

    prompt_input = PROMPT_DICT["prompt_input"]
    self.sources = [
        prompt_input.format_map(example) for example in self.list_data_dict
    ]
    self.targets = [f"{example['output']}" for example in self.list_data_dict]

    self.source_token_ids, self.source_attn_masks = self.encode_samples(
        self.sources
    )
    if encode_targets:
      self.target_token_ids, self.target_attn_masks = self.encode_samples(
          self.targets, sample_type="Target"
      )
    self.inputs_str = [
        ",".join([str(i) for i in source_token_ids])
        for source_token_ids in self.source_token_ids
    ]

    self.count = total_count_override or len(self.sources)
    self.perf_count = perf_count_override or self.count

  def encode_samples(self, samples, sample_type="Source"):
    """Encode samples."""
    print(f"Encoding {sample_type} Samples")

    total_samples = len(samples)

    token_ids = []
    attn_masks = []

    for i in range(total_samples):
      sample_encoded = self.tokenizer(samples[i])
      token_ids.append(sample_encoded.input_ids)
      attn_masks.append(sample_encoded.attention_mask)

    return token_ids, attn_masks

  def LoadSamplesToRam(self, sample_list):  # pylint:disable=invalid-name
    """Load samples to RAM."""

  def UnloadSamplesFromRam(self, sample_list):  # pylint:disable=invalid-name
    """Unload samples from RAM."""

  def __del__(self):
    print("Finished destroying QSL.")
