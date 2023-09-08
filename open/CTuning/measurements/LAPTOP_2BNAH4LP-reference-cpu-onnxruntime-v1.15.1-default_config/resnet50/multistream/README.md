This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=generate-run-cmds,inference,_populate-readme,_all-scenarios \
	--model=resnet50 \
	--device=cpu \
	--implementation=reference \
	--backend=onnxruntime \
	--execution-mode=valid \
	--results_dir=C:\Users\MlperfInference/results_dir \
	--category=edge \
	--division=open \
	--quiet
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`

## Dependent CM scripts for the MLPerf Inference Implementation


1. `cm run script --tags=detect,os`


2. `cm run script --tags=detect,cpu`


3. `cm run script --tags=get,sys-utils-cm`


4. `cm run script --tags=get,python`


5. `cm run script --tags=get,generic-python-lib,_onnxruntime`


6. `cm run script --tags=get,ml-model,image-classification,resnet50,_fp32,_onnx,raw`


7. `cm run script --tags=get,dataset,image-classification,imagenet,preprocessed,_default,_full,_NCHW`


8. `cm run script --tags=get,dataset-aux,image-classification,imagenet-aux`


9. `cm run script --tags=generate,user-conf,mlperf,inference`


10. `cm run script --tags=get,loadgen`


11. `cm run script --tags=get,mlcommons,inference,src`


12. `cm run script --tags=get,generic-python-lib,_opencv-python`


13. `cm run script --tags=get,generic-python-lib,_numpy`


14. `cm run script --tags=get,generic-python-lib,_pycocotools`
