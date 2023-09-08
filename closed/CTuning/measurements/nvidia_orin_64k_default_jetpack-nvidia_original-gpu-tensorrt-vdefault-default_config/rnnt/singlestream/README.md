This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=rnnt \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4940 \
	--execution-mode=valid \
	--adr.nvidia-harness.tags=_maxn \
	--gpu_name=orin \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes \
	--offline_target_qps=1600 \
	--singlestream_target_latency=90 \
	--multistream_target_latency=105
```