This experiment is generated using [MLCommons CM](https://github.com/mlcommons/ck)
## CM Run Command
```
cm run script \
	run mobilenet-models _tflite _performance-only \
	--adr.compiler.tags=gcc \
	--results_dir=/home/cmuser/mobilenet_results \
	--power=yes \
	--adr.mlperf-power-client.power_server=192.168.0.15
```