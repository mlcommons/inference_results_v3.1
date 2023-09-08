This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _submission _all-scenarios \
	--model=bert-99.9 \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=datacenter \
	--division=closed \
	--quiet \
	--gpu_name=custom \
	--adr.cuda.version=12.2 \
	--offline_target_qps=3300 \
	--server_target_qps=2800 \
	--execution_mode=valid
```