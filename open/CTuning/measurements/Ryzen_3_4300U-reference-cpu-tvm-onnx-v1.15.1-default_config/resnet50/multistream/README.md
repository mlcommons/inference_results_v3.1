This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	- \
	- \
	a \
	d \
	r \
	. \
	p \
	y \
	t \
	h \
	o \
	n \
	. \
	n \
	a \
	m \
	e \
	= \
	m \
	l \
	p \
	e \
	r \
	f \
	  \
	- \
	- \
	a \
	d \
	r \
	. \
	t \
	v \
	m \
	. \
	t \
	a \
	g \
	s \
	= \
	_ \
	p \
	i \
	p \
	- \
	i \
	n \
	s \
	t \
	a \
	l \
	l \
	  \
	- \
	- \
	a \
	d \
	r \
	. \
	t \
	v \
	m \
	- \
	m \
	o \
	d \
	e \
	l \
	. \
	t \
	a \
	g \
	s \
	= \
	_ \
	t \
	u \
	n \
	e \
	- \
	m \
	o \
	d \
	e \
	l
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


6. `cm run script --tags=get,ml-model,image-classification,resnet50,_fp32,_batch_size.1`


7. `cm run script --tags=get,dataset,image-classification,imagenet,preprocessed,_default,_full`


8. `cm run script --tags=get,dataset-aux,image-classification,imagenet-aux`


9. `cm run script --tags=generate,user-conf,mlperf,inference`


10. `cm run script --tags=get,loadgen`


11. `cm run script --tags=get,mlcommons,inference,src`


12. `cm run script --tags=get,generic-python-lib,_opencv-python`


13. `cm run script --tags=get,generic-python-lib,_numpy`


14. `cm run script --tags=get,generic-python-lib,_pycocotools`


15. `cm run script --tags=get,generic-python-lib,_onnx`


16. `cm run script --tags=get,tvm,_pip-install`


17. `cm run script --tags=get,tvm-model,_onnx,_tune-model,_batch_size.1,_model.resnet50`
