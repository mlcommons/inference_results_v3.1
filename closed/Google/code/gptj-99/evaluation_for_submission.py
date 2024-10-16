"""Evaluate lm model output."""
import argparse
import json
from dataset import Dataset
import evaluate
import nltk
import numpy as np
from transformers import GPT2Tokenizer  # pylint: disable=g-importing-member


def get_args():
  """Parse commandline."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--mlperf-accuracy-file",
      required=True,
      help="path to mlperf_log_accuracy.json",
  )
  parser.add_argument(
      "--dataset-file", required=True, help="path to cnn_eval.json"
  )
  parser.add_argument("--verbose", action="store_true", help="verbose messages")
  parser.add_argument(
      "--dtype",
      default="int64",
      help="dtype of the accuracy log",
      choices=["int32", "int64"],
  )
  parser.add_argument(
      "--encoded", action="store_true", help="Accuracy log is in bytes"
  )
  args = parser.parse_args()
  return args


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]

  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def main():
  args = get_args()
  dataset_path = args.dataset_file
  metric = evaluate.load("rouge")
  nltk.download("punkt")

  tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer/")

  tokenizer.pad_token = tokenizer.eos_token

  data_object = Dataset(dataset_path)

  targets = data_object.targets

  with open(args.mlperf_accuracy_file, "r") as f:
    results = json.load(f)

  # Deduplicate the results loaded from the json
  dedup_results = []
  seen = set()
  for result in results:
    item = result["qsl_idx"]
    if item not in seen:
      seen.add(item)
      dedup_results.append(result)
  results = dedup_results

  target_required = []
  preds_decoded_text = []

  eval_dtype = np.int32

  for pred in results:
    qsl_idx = pred["qsl_idx"]
    target = targets[qsl_idx]
    target_required.append(target)
    if args.encoded:
      tks = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)
      tks = list(tks)
    else:
      tks = [int(x) for x in pred["data"].split(",")]
    try:
      end_here = tks.index(50256)  # End at endotext token
    except ValueError:
      end_here = len(tks)

    postprocessed_tks = tks[: end_here + 1]
    decoded_pre = tokenizer.decode(tks, skip_special_tokens=True)
    decoded = tokenizer.decode(postprocessed_tks, skip_special_tokens=True)

    preds_decoded_text.append(decoded)

    s = metric.compute(
        references=[target],
        predictions=[decoded],
        use_stemmer=True,
        use_aggregator=False,
    )
    if args.verbose and s["rouge2"][0] < 0.5:
      print("\n\n>>> Decoding .. ", tks, "\nto\n   ", decoded_pre)
      print(
          f"[{qsl_idx}] score:",
          s,
          "\n  pred:",
          decoded.replace("\n", ""),
          "\n  target:",
          target.replace("\n", ""),
      )

  preds, targets = postprocess_text(preds_decoded_text, target_required)

  result = metric.compute(
      predictions=preds,
      references=targets,
      use_stemmer=True,
      use_aggregator=False,
  )
  result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  result["gen_len"] = np.sum(prediction_lens)
  result["gen_num"] = len(preds)
  print("\nResults\n")
  print(result)


if __name__ == "__main__":
  main()
