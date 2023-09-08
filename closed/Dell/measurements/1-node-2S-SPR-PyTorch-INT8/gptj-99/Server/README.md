## Get Started with GPT-J

### Download and Prepare Dataset
```
export WORKLOAD_DATA=/data/mlperf_data/gpt-j
mkdir -p ${WORKLOAD_DATA}
```

+ Download cnn-dailymail calibration set
```
cd <THIS_REPO>/closed/Intel/code/gptj-99/pytorch-cpu/
python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${WORKLOAD_DATA}/calibration-data
```

+ Download cnn-dailymail validation set
```
python download-dataset.py --split validation --output-dir ${WORKLOAD_DATA}/validation-data
```

### Download and prepare model
+ Get finetuned checkpoint
```
CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
wget https://urldefense.com/v3/__https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download__;!!LpKI!n3qqWloKaag2oQ586ugayXoXME1AJzHBKld50UVA74Hx9c-bvLVxZDafFTjMh0MtfCJFikH6MPRXYszQEYI$ [cloud[.]mlcommons[.]org] -O gpt-j-checkpoint.zip
unzip gpt-j-checkpoint.zip
mv gpt-j/checkpoint-final/ ${CHECKPOINT_DIR}
```
Note: wget commands use IPv6 by default, if your system uses IPv4, please add -4 option into the wget command to force it to use IPv4.

### Build & Run Docker container from Dockerfile
If you haven't already done so, build the Intel optimized Docker image for GPT-J using:
```
cd <THIS_REPO>/closed/Intel/code/gptj-99/pytorch-cpu/docker
bash build_gpt-j_container.sh

docker run --name intel_gptj --privileged -itd --net=host --ipc=host -v ${WORKLOAD_DATA}:/opt/workdir/code/gptj-99/pytorch-cpu/data mlperf_inference_gptj:3.1
docker exec -it intel_gptj bash

cd code/gptj-99/pytorch-cpu
```
### Generate quantized INT8 model
```
source setup_env.sh
bash run_quantization.sh
```
### Run Benchmarks
+ Offline (Performance)
```
bash run_offline.sh
```


+ Offline (Accuracy)
```
bash run_offline_accuracy.sh
```

+ Server (Performance)
```
bash run_server.sh
```

+ Server (Accuracy)
```
bash run_server_accuracy.sh
```
