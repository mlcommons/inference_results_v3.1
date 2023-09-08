This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _submission _all-scenarios \
	--model=bert-99 \
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
	--offline_target_qps=270 \
	--execution-mode=valid \
	--adr.nvidia-harness.tags=_maxq \
	--gpu_name=orin \
	--results_dir=/home/arjun/results_dir \
	--singlestream_target_latency=12 \
	--skip_submission_generation=yes
```