This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	generate-run-cmds inference _populate-readme \
	--model=bert-99 \
	--implementation=nvidia-original \
	--device=cuda \
	--backend=tensorrt \
	--category=edge \
	--division=closed \
	--quiet \
	--adr.nvidia-harnesss.power_setting=MaxQ \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4940 \
	--execution-mode=valid \
	--adr.nvidia-harness.tags=_maxq \
	--gpu_name=orin \
	--results_dir=/home/arjun/results_dir \
	--skip_submission_generation=yes
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


4. `cm run script --tags=get,cuda,_cudnn`


5. `cm run script --tags=get,tensorrt`


6. `cm run script --tags=build,nvidia,inference,server,_nvidia-only`


7. `cm run script --tags=get,mlperf,inference,nvidia,scratch,space`


8. `cm run script --tags=get,generic-python-lib,_mlperf_logging`


9. `cm run script --tags=get,ml-model,bert,_onnx,_fp32`


10. `cm run script --tags=get,ml-model,bert,_onnx,_int8`


11. `cm run script --tags=get,squad-vocab`


12. `cm run script --tags=get,mlcommons,inference,src,_deeplearningexamples`


13. `cm run script --tags=get,nvidia,mlperf,inference,common-code,_nvidia-only`


14. `cm run script --tags=generate,user-conf,mlperf,inference`


15. `cm run script --tags=get,generic-python-lib,_transformers`


16. `cm run script --tags=get,generic-python-lib,_onnx`


17. `cm run script --tags=reproduce,mlperf,inference,nvidia,harness,_build_engine,_maxq,_bert-99,_tensorrt,_cuda,_offline,_orin,_bert_`


18. `cm run script --tags=reproduce,mlperf,inference,nvidia,harness,_preprocess_data,_bert-99,_tensorrt,_cuda,_orin,_bert_`
