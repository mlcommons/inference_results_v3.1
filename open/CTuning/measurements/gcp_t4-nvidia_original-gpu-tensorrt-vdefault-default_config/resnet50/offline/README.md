This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _performance-only \
	--model=resnet50 \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=open \
	--quiet \
	--execution_mode=valid \
	--preprocess_submission \
	--results_dir=/home/cmuser/results_dir \
	--scenario=Offline \
	--offline_target_qps=5000 \
	--rerun
```