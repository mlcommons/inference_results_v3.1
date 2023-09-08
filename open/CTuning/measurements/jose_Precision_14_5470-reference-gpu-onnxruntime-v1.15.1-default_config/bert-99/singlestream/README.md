This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _populate-readme _all-scenarios \
	--model=bert-99 \
	--device=cuda \
	--implementation=reference \
	--backend=onnxruntime \
	--execution-mode=valid \
	--results_dir=/home/jose/results_dir \
	--category=edge \
	--division=open \
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


5. `cm run script --tags=get,cuda,_cudnn`


6. `cm run script --tags=get,generic-python-lib,_onnxruntime_gpu`


7. `cm run script --tags=get,generic-python-lib,_torch`


8. `cm run script --tags=get,generic-python-lib,_transformers`


9. `cm run script --tags=get,ml-model,language-processing,bert-large,_fp32,_onnx,raw`


10. `cm run script --tags=get,dataset,squad,original`


11. `cm run script --tags=get,dataset-aux,squad-vocab`


12. `cm run script --tags=generate,user-conf,mlperf,inference`


13. `cm run script --tags=get,loadgen`


14. `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


15. `cm run script --tags=get,generic-python-lib,_package.pydantic`


16. `cm run script --tags=get,generic-python-lib,_tokenization`


17. `cm run script --tags=get,generic-python-lib,_six`
