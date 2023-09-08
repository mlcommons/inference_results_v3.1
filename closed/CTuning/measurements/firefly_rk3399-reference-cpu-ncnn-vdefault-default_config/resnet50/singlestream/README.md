This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission,_full \
	--quiet \
	--submitter=Collabora \
	--hw_name=default \
	--model=resnet50 \
	--implementation=reference \
	--backend=ncnn \
	--device=cpu \
	--scenario=SingleStream \
	--adr.mlperf-inference-implementation.num_threads=1 \
	--adr.mlperf-inference-implementation.tags=_batch_size.1 \
	--clean \
	--env.IMAGENET_PATH=/mnt/workspace/imagenet-2012/val \
	--execution_mode=valid \
	--division=closed \
	--preprocess_submission \
	--result_dir=/mnt/workspace/mlperf_results
```