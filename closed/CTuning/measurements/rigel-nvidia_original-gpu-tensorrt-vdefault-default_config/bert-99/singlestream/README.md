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
	--scenario=Offline \
	--execution-mode=valid \
	--target_qps=15000 \
	--rerun \
	--gpu_name=a100 \
	--adr.nvidia-harness.tags=_sxm \
	--results_dir=/home/cmuser/results_dir
```