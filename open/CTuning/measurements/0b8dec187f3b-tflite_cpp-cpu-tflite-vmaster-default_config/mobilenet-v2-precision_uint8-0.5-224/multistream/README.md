This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	run mobilenet-models _tflite _populate-readme \
	--adr.compiler.tags=gcc \
	--results_dir=/root/mobilenet_results
```
## Dependent CM scripts 


1.  `cm run script --tags=detect,os`


2.  `cm run script --tags=get,sys-utils-cm`


3.  `cm run script --tags=get,python`


4.  `cm run script --tags=get,mlcommons,inference,src`


5.  `cm run script --tags=get,dataset-aux,imagenet-aux`
