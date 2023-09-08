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
	--adr.nvidia-harnesss.power_setting=MaxQ \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4940 \
	--offline_target_qps=3000 \
	--execution-mode=valid \
	--adr.nvidia-harness.tags=_maxq \
	--gpu_name=orin \
	--env.CM_MLPERF_PERFORMANCE_SAMPLE_COUNT=2048 \
	--results_dir=/home/arjun/results_dir \
	--singlestream_target_latency=1 \
	--multistream_target_latency=2
```