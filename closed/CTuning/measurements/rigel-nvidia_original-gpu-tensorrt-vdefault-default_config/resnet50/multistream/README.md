This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _submission _all-scenarios \
	--model=resnet50 \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--execution-mode=valid \
	--target_qps=160000 \
	--rerun \
	--gpu_name=a100 \
	--adr.nvidia-harness.tags=_sxm \
	--env.CM_MLPERF_PERFORMANCE_SAMPLE_COUNT=2048 \
	--results_dir=/home/cmuser/results_dir
```