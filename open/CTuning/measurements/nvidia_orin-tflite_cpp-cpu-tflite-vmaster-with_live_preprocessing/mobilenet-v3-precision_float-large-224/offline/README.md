This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	--tags=run,mobilenet-models,_tflite,_armnn,_neon,_populate-readme,_only-fp32 \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15 \
	--adr.mlperf-power-client.port=4940 \
	--adr.mlperf-inference-implementation.compressed_dataset=on \
	--adr.compiler.tags=gcc \
	--results_dir=/home/arjun/mobilenet_results
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`
