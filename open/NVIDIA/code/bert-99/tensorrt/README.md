# BERT Benchmark Setup and Usage

This benchmark performs language processing using BERT network.

:warning: **IMPORTANT**: Please use [open/NVIDIA](/open/NVIDIA) as the working directory when running the below commands. :warning:

## Dataset

Please refer to the [dataset section in our closed division submission](/closed/NVIDIA/code/bert/tensorrt/README.md#dataset) for information on setting up the SQuAD v1.1 validation dataset.

## Model

### Downloading / obtaining the model

Our optimized bert model is available at `/opt/bert_open/` in our NGC container for the closed division submission.

## Optimizations

In addition to the [optimizations in our closed division submission](/closed/NVIDIA/code/bert/tensorrt/README.md#optimizations), we perform the following model optimizations for our open division submission to achieve this speedup:

### Pre-training BERT with Whole Word Masking

We use the popular Whole Word Masking technique where we mask all the tokens corresponding to a word at once. This is in contrast to the default masking strategy where we mask each token independently. This technique leads to better starting accuracy.

Here we skip this step and reuse the [HuggingFace QA checkpoint](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) instead of retraining from scratch.

### Pruning transformer layers, attention heads and linear layers

We use our proprietary automatic structured pruning tool that uses a gradient-based sensitivity analysis to prune the model to the given target FLOPs.

We prune the number of transformer layers, attention heads and linear layer dimension in all the tranformer layers in the model and keep the embedding dimension unchanged.

The pruned model searched by our tool reduces the number of parameters by 4x and the number of FLOPs by 5.6x. This model has varying number of heads and linear layer dimension in each layer for optimal accuracy.

### Fine-tuning with distillation

We fine-tune the above pruned model with distillation from the Whole Word Masking QA teacher obtained in Step 1.

### INT8 Quantization

We quantize the fine-tuned model to INT8 precision using the same quantization tool used in our submission for the closed division.

We use a combination of post-training quantization and quantization-aware training (with distillation) to achieve the best accuracy (above 99% of the required target).

## Results

```
Offline:
  Samples per second: 4609.0

Server:
  Completed samples per second    : 4264.74

SingleStream:
  50.00 percentile latency (ns)   : 750092
  90.00 percentile latency (ns)   : 816040
  95.00 percentile latency (ns)   : 883759
  97.00 percentile latency (ns)   : 927099
  99.00 percentile latency (ns)   : 946718
  99.90 percentile latency (ns)   : 973139
```


## Instructions for Auditors

### Run Inference through LoadGen

Run the following commands from within the container to run inference through LoadGen:

```bash
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=default --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=default --test_mode=AccuracyOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run RUN_ARGS="--benchmarks=bert --scenarios=<SCENARIO> --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

The performance and the accuracy results will be printed to stdout, and the LoadGen logs can be found in `build/logs`.
