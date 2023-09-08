This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_performance-only \
	--model=bert-99 \
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
	--offline_target_qps=540 \
	--singlestream_target_latency=0.6 \
	--multistream_target_latency=2 \
	--scenario=Offline \
	--rerun \
	--adr.cuda.version=11.4 \
	--env.CM_RUN_PREFIX0=sleep 300 && 
```