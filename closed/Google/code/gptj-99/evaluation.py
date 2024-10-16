"""mlperf inference GPT-J model output evaluation script."""
import json
import os
from absl import app
from absl import flags
import numpy as np
import transformers
from google3.pyglib import gfile
from google3.pyglib import resources

from google3.third_party.google_research.google_research.rouge import rouge_scorer
from google3.third_party.mlperf.inference.gptj import dataset


_MLPERF_ACC_LOG = flags.DEFINE_string(
    'mlperf_acc_log',
    None,
    'SAX cell of the admin server.',
)
_DATASET_PATH = flags.DEFINE_string(
    'dataset_path',
    '/cns/ik-d/home/mrasquinha/llm_6B/data/cnn_eval.json',
    'SAX cell of the admin server.',
)
_EVAL_LOG_DIR = flags.DEFINE_string(
    'eval_log_dir',
    None,
    'SAX cell of the admin server.',
)

_DTYPE = 'int64'
_TOKENIZER_PATH = 'google3/third_party/mlperf/inference/gptj/tokenizer/tokenizer_config.json'


def compute_rogue_scores(targets, predictions):
  """Compute rogue scores."""
  assert len(targets) == len(predictions)
  scorer = rouge_scorer.RougeScorer(
      ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True
  )
  print('Compute rouge for {} samples'.format(len(targets)))

  rogue_scores, r1, r2, rl, rlsum = dict(), list(), list(), list(), list()
  result_log = gfile.Open(
      os.path.join(_EVAL_LOG_DIR.value, 'eval_log.json'), 'ab'
  )
  for idx, (target, prediction) in enumerate(zip(targets, predictions)):
    scores = scorer.score(target, prediction)

    # append some low scoring results to the eval log.
    if(scores['rouge1'].fmeasure < 0.2):
      res = {}
      res['scores'] = scores
      res['target'] = target
      res['prediction'] = prediction
      result_log.write(json.dumps(res))

    r1.append(scores['rouge1'])
    r2.append(scores['rouge2'])
    rl.append(scores['rougeL'])
    rlsum.append(scores['rougeLsum'])

    if idx % 100 == 0:
      tmp_r1 = np.mean(scores['rouge1'], axis=0)
      tmp_r2 = np.mean(scores['rouge2'], axis=0)
      tmp_rl = np.mean(scores['rougeL'], axis=0)
      print(f'Computed {idx} samples so far... {tmp_r1} / {tmp_r2} / {tmp_rl}')

  rogue_scores['r1_mean'] = np.mean(r1)
  rogue_scores['r2_mean'] = np.mean(r2)
  rogue_scores['rl_mean'] = np.mean(rl)
  rogue_scores['rlsum_mean'] = np.mean(rlsum)

  rogue_scores['gen_len'] = sum([len(prediction) for prediction in predictions])
  rogue_scores['gen_num'] = len(predictions)

  return rogue_scores


def main(argv):
  del argv
  print('Loading Dataset ... ')
  tk_path = os.path.dirname(resources.GetResourceFilename(_TOKENIZER_PATH))
  tokenizer = transformers.GPT2Tokenizer.from_pretrained(
      tk_path,
      model_max_length=2048,
      padding_side='left',
      use_fast=False,
  )
  tokenizer.pad_token = tokenizer.eos_token

  data_object = dataset.Dataset(dataset_path=_DATASET_PATH.value)

  targets = data_object.targets
  with gfile.Open(_MLPERF_ACC_LOG.value, 'r') as f:
    results = json.load(f)

  target_required = []
  preds_token_ids = []

  eval_dtype = np.int64
  if _DTYPE == 'int32':
    eval_dtype = np.int32

  for pred in results:
    qsl_idx = pred['qsl_idx']
    target = targets[qsl_idx]
    target_required.append(target)
    preds_token_ids.append(
        np.frombuffer(bytes.fromhex(pred['data']), eval_dtype)
    )

  preds_decoded_text = tokenizer.batch_decode(
      preds_token_ids, skip_special_tokens=True
  )

  preds, targets = (preds_decoded_text, target_required)

  result = compute_rogue_scores(targets, preds)
  result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  result['gen_len'] = np.sum(prediction_lens)
  result['gen_num'] = len(preds)
  print('\nResults\n')
  print(result)


if __name__ == '__main__':
  app.run(main)
