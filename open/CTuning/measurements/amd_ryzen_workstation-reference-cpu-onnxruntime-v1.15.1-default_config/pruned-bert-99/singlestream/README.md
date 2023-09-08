This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--backend=onnxruntime \
	--device=cpu \
	--scenario=SingleStream \
	--target_latency=100 \
	--mode=performance \
	--execution_mode=valid \
	--test_query_count=100 \
	--adr.mlperf-inference-implementation.max_batchsize=384 \
	--results_dir=/home/arjun/results_dir_pruned \
	--env.CM_MLPERF_CUSTOM_MODEL_PATH=/home/arjun/CM/repos/local/cache/70929fdd59a946b6/repo/pruned-model-0.99-mlperf-new1/pruned_model.onnx \
	--env.CM_ML_MODEL_FULL_NAME=bert-pruned-99 \
	--quiet
```