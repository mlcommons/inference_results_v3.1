This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission,_all-scenarios \
	--model=resnet50 \
	--execution-mode=valid \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--results_dir=/home/cmuser/results_dir \
	--category=datacenter-edge \
	--division=open \
	--skip_submission_generation=yes \
	--quiet \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--offline_target_qps=45000 \
	--server_target_qps=38000 \
	--singlestream_target_latency=0.2 \
	--multistream_target_latency=0.4
```