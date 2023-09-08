This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission,_full \
	--model=resnet50 \
	--implementation=tflite-cpp \
	--device=cpu \
	--scenario=SingleStream \
	--category=edge \
	--division=closed \
	--quiet \
	--adr.compiler.tags=gcc \
	--execution_mode=valid \
	--results_dir=/mnt/workspace/mlperf_results \
	--env.IMAGENET_PATH=/mnt/workspace/imagenet-2012/val \
	--env.CM_MLPERF_USE_MAX_DURATION=no \
	--singlestream_target_latency=900 \
	--env.CM_MLPERF_INFERENCE_MIN_DURATION=900000
```