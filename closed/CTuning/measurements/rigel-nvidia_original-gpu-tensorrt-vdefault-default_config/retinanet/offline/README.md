This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _submission _all-scenarios \
	--model=retinanet \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--execution-mode=valid \
	--target_qps=2800 \
	--rerun \
	--gpu_name=a100 \
	--adr.nvidia-harness.tags=_sxm \
	--results_dir=/home/cmuser/results_dir
```