This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=rnnt \
	--execution-mode=valid \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--results_dir=/home/cmuser/results_dir \
	--category=edge \
	--division=closed \
	--skip_submission_generation=yes \
	--quiet \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--offline_target_qps=15200 \
	--server_target_qps=14150 \
	--singlestream_target_latency=23
```