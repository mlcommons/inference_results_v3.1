This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_submission \
	--scenario=Offline \
	--model=resnet50 \
	--implementation=reference \
	--device=cuda \
	--backend=tf \
	--category=edge \
	--division=closed \
	--quiet \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes \
	--execution-mode=valid \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4950
```