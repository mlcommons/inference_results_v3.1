This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_submission \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--precision=int8 \
	--backend=deepsparse \
	--device=cpu \
	--scenario=Offline \
	--execution_mode=valid \
	--adr.mlperf-inference-implementation.max_batchsize=384 \
	--env.CM_MLPERF_SUT_NAME_RUN_CONFIG_SUFFIX1=maxbatchsize.384 \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned97-none \
	--quiet
```