This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_armnn,_neon,_accuracy-only \
	--adr.compiler.tags=gcc \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--results_dir=/home/ubuntu/mobilenet_results
```