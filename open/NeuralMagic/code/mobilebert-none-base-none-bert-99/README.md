This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mlperf,inference,generate-run-cmds,_populate-readme \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--precision=int8 \
	--backend=deepsparse \
	--device=cpu \
	--scenario=Offline \
	--mode=performance \
	--execution_mode=valid \
	--adr.mlperf-inference-implementation.max_batchsize=384 \
	--results_dir=/home/arjunsuresh/results_dir \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none \
	--quiet
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


5.  `cm run script --tags=get,dataset,squad,language-processing`


6.  `cm run script --tags=get,dataset-aux,squad-vocab`

## Dependent CM scripts for the MLPerf Inference Implementation


1. `cm run script --tags=detect,os`


2. `cm run script --tags=detect,cpu`


3. `cm run script --tags=get,sys-utils-cm`


4. `cm run script --tags=get,python`


5. `cm run script --tags=get,generic-python-lib,_torch`


6. `cm run script --tags=get,generic-python-lib,_transformers`


7. `cm run script --tags=get,ml-model,neural-magic,zoo,_model-stub.zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base-none`


8. `cm run script --tags=get,dataset,squad,original`


9. `cm run script --tags=get,dataset-aux,squad-vocab`


10. `cm run script --tags=generate,user-conf,mlperf,inference`


11. `cm run script --tags=get,loadgen`


12. `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


13. `cm run script --tags=get,generic-python-lib,_deepsparse`


14. `cm run script --tags=get,generic-python-lib,_package.pydantic`


15. `cm run script --tags=get,generic-python-lib,_tokenization`


16. `cm run script --tags=get,generic-python-lib,_six`
