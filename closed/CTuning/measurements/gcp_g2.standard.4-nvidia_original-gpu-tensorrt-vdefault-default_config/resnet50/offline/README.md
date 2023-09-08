This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _submission \
	--offline_target_qps=12000 \
	--model=resnet50 \
	--device=cuda \
	--implementation=nvidia-original \
	--backend=tensorrt \
	--execution-mode=valid \
	--results_dir=/home/cmuser/results_dir \
	--category=edge \
	--division=closed \
	--quiet \
	--skip_submission_generation=yes \
	--gpu_name=l4 \
	--scenario=Offline
```