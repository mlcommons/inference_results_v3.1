# Get Started with Intel MLPerf v3.1 Submission with Intel Optimized Docker Images

MLPerf is a benchmark for measuring the performance of machine learning
systems. It provides a set of performance metrics for a variety of machine
learning tasks, including image classification, object detection, machine
translation, and others. The benchmark is representative of real-world
workloads and as a fair and useful way to compare the performance of different
machine learning systems.


In this document, we'll show how to run Intel MLPerf v3.1 submission with Intel
optimized Docker images.

## Intel Docker Images for MLPerf

The Intel optimized Docker images for MLPerf v3.1 can be built using the Dockerfiles.
Example for building docker image with Dockerfile:
```
cd <THIS_REPO>/closed/Intel/code/resnet50/pytorch-cpu/docker/

bash build_resnet50_contanier.sh
```

## HW configuration:

| System Info     | Configuration detail                 |
| --------------- | ------------------------------------ |
| CPU             | SPR                       |
| OS              | CentOS  Stream 8                     |
| Kernel          | 6.1.11-1.el8.elrepo.x86_64 |
| Memory          | 1024GB (16x64GB 4800MT/s [4800MT/s]) |
| Disk            | 1TB NVMe                             |

Best Known Configurations:

```
sudo bash run_clean.sh
```

In the following sections, we'll show you how to set up and run each of the six models:

* [3DUNET](#get-started-with-3dunet)
* [BERT](#get-started-with-bert)
* [DLRM2](#get-started-with-dlrm2)
* [GPT-J](#get-started-with-gpt-j)
* [RESNET50](#get-started-with-resnet50)
* [RETINANET](#get-started-with-retinanet)
* [RNNT](#get-started-with-rnnt)

---


## Get Started with 3DUNET
If you haven't already done so, build the Intel optimized Docker image for 3DUNET using:
```
cd <THIS_REPO>/closed/Intel/code/3d-unet-99.9/pytorch-cpu/docker
bash build_3dunet_container.sh
```

### Prerequisites
Use these commands to prepare the 3DUNET dataset and model on your host system:

```
mkdir 3dunet
cd 3dunet
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
```

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/3dunet`` with the 3dunet folder path created earlier:
```
docker run --name intel_3dunet --privileged -itd -v /path/to/3dunet:/root/mlperf_data/3dunet-kits --net=host --ipc=host mlperf_inference_3dunet:3.1
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it intel_3dunet bash
```

#### Preprocess Data
If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Preprocess the data and download the model using the provided script:
```
cd code/3d-unet-99.9/pytorch-cpu/
bash process_data_model.sh 
```

### Run the Benchmark

```
# 3dunet only has offline mode
bash run.sh perf # offline performance
bash run.sh acc  # offline accuracy
```

### Get the Results

* Check log file. Performance results are in ``./output/mlperf_log_summary.txt``.
  Verify that you see ``results is: valid``.

* For offline mode performance, check the field ``Samples per second:``
* Accuracy results are in ``./output/accuracy.txt``.  Check the field ``mean =``.
* The performance result is controled by the value of "target_qps" in user.conf file. The scripts will automatically select user_default.conf file to calculate corresponding "target_qps" according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user.conf files.
  
Save these output log files elsewhere when each test is completed as
they will be overwritten by the next test.


##  Get started with BERT
The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available).

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for BERT using:
```
cd <THIS_REPO>/closed/Intel/code/bert-99/pytorch-cpu/docker/

bash build_bert-99_contanier.sh
```

### Prerequisites
Use these commands to prepare the BERT dataset and model on your host system:

```
cd /data/mlperf_data   # or path to where you want to store the data
mkdir bert
cd bert
mkdir dataset
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dataset/dev-v1.1.json
git clone https://huggingface.co/bert-large-uncased model
cd model
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
```
Note: wget commands use IPv6 by default, if your system uses IPv4, please add -4 option into the wget command to force it to use IPv4.

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled or built earlier.
Replace /path/of/bert with the bert folder path created earlier (i.e. /data/mlperf_data/bert):

```
docker run --name bert_3-1 --privileged -itd --net=host --ipc=host -v /path/of/bert:/data/mlperf_data/bert <bert docker image ID>
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it bert_3-1 bash
```

#### Convert Dataset and Model
If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

```
cd code/bert-99/pytorch-cpu
export DATA_PATH=/data/mlperf_data/bert
bash convert.sh
```

### Run the Benchmark

```
bash run.sh                    #offline performance
bash run.sh --accuracy         #offline accuracy
bash run_server.sh             #server performance
bash run_server.sh --accuracy  #server accuracy
```


### Get the Results

Check the performance log file ``./test_log/mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user.conf file. The scripts will automatically select user_default.conf file to calculate corresponding "target_qps" according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user.conf files.


Check the accuracy log file ``./test_log/accuracy.txt``.

* Check the field ``f1``


Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

## Get started with DLRM2
If you haven't already done so, build the Intel optimized Docker image for DLRM using:
```
cd <THIS_REPO>/closed/Intel/code/dlrm-99.9/pytorch-cpu-int8/docker
# Please firstly refer to the prerequisite file in the current directory to download the compiler before building the Docker image. 
bash build_dlrm-99.9_container.sh
```

### Prerequisites
Use these commands to prepare the Deep Learning Recommendation Model (DLRM)
dataset and model on your host system:

```
cd /data/   # or path to where you want to store the data
mkdir -p /data/mlperf_data/dlrm_2/model/bf16
mkdir -p /data/mlperf_data/dlrm_2/data_npy

# Prepare DLRM dataset
# Create a directory (such as /data/mlperf_data/dlrm_2/data_npy) which contain:
#	    day_23_dense.npy
#     day_23_sparse_multi_hot.npz
#     day_23_labels.npy
#
# Learn how to get the dataset from:
#     https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch

# Prepare pre-trained DLRM model
cd /data/mlperf_data/dlrm_2/model/bf16
wget https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download
unzip weights.zip

cd <THIS_REPO>/closed/Intel/code/dlrm-99.9/pytorch-cpu/
export MODEL_DIR=/data/mlperf_data/dlrm_2/model/bf16


# dump model from snapshot to torch
bash run_dump_torch_model.sh
```
Note: wget commands use IPv6 by default, if your system uses IPv4, please add -4 option into the wget command to force it to use IPv4.

### Set Up Environment
Follow these steps to set up the docker instance.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/dlrm`` with the ``dlrm`` folder path created earlier (/data/dlrm for example):

```
docker run --name intel_inference_dlrm_int8 --privileged -itd --net=host --ipc=host \
  -v /path/of/dlrm:/data/dlrm_2_dataset mlperf_inference_dlrm2:3.1
```

#### Login to Docker Container
Login into a bashrc shell in the Docker instance.

```
docker exec -it intel_inference_dlrm_int8 bash
```

### Preprocess model and dataset

If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

#### Calibrate and dump int8 model

```
cd /opt/workdir/code/dlrm2-99.9/pytorch-cpu-int8
bash ./run_calibration.sh
```
> Note: run_calibration script does not need to finish, once you see *roc_auc* output you can `ctrl+z` to stop


Export model and dataset directory

```
# export model directory to saved model path
export MODEL_DIR=/data/mlperf_data/dlrm_2/model/bf16

# export dataset directory to saved dataset path where .npy .npz are stored.
export DATA_DIR=/data/mlperf_data/dlrm_2/data_npy
```


### Run the Benchmark

```
# offline performance
source setup_env_offline.sh
bash run_main.sh offline int8                   

# offline accuracy
source setup_env_offline.sh
bash run_main.sh offline accuracy int8	          

# server performance
source setup_env_server.sh
bash run_main.sh server int8

# server accuracy
source setup_env_server.sh   
bash run_main.sh server accuracy int8    
```


---
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
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
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

---
##  Get Started with ResNet50
The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available). Please download the Imagenet dataset on the host system before starting the container.

### Download Imagenet Dataset for Calibration
Download ImageNet (50000) dataset
```
bash download_imagenet.sh
```

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for ResNet50 using:
```
cd <THIS_REPO>/closed/Intel/code/resnet50/pytorch-cpu/docker/

bash build_resnet50_contanier.sh

docker run -v </path/to/ILSVRC2012_img_val>:/opt/workdir/code/resnet50/pytorch-cpu/ILSVRC2012_img_val -it --privileged <resnet docker image ID> /bin/bash

cd code/resnet50/pytorch-cpu
```

### Prepare Calibration Dataset & Download Model ( Inside Container )
If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Prepare calibration 500 images into folders
```
bash prepare_calibration_dataset.sh
```

Download the model
```
bash download_model.sh
```
The downloaded model will be saved as ```resnet50-fp32-model.pth```

### Quantize Torchscript Model and Check Accuracy 
+ Set the following paths:
```
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth
```
+ Generate scales and models
```
bash generate_torch_model.sh
```

The *start* and *end* parts of the model are also saved (respectively named) in ```models```


### Run Benchmark (Common for Docker & Baremetal)

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth
```

#### Performance
+ Offline
```
bash run_offline.sh <batch_size>
```
Note: <batch_size> should be 8 or 256.

+ Server
```
bash run_server.sh
```

#### Accuracy
+ Offline
```
bash run_offline_accuracy.sh <batch_size>
```

+ Server
```
bash run_server_accuracy.sh
```


### Get the Results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user.conf file. The scripts will automatically select user_default.conf file to calculate corresponding "target_qps" according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user.conf files.

Check the ``./offline_accuracy.txt`` or ``./server_accuracy.txt`` log file:

* Check the field ``accuracy``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

##  Get Started with Retinanet

The docker container can be created either by building it using the Dockerfile or pulling the image from Dockerhub (if available). Please download the Imagenet dataset on the host system before starting the container.

### Download the dataset
+ Install dependencies (**python3.9 or above**)
```
pip3 install --upgrade pip --user
pip3 install opencv-python-headless==4.5.3.56 pycocotools==2.0.2 fiftyone==0.16.5
```

+ Setup env vars
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env
```

+ Download OpenImages (264) dataset
```
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
```
Images are downloaded to `${WORKLOAD_DATA}/openimages`

+ Download Calibration images
```
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
```
Calibration dataset downloaded to `${WORKLOAD_DATA}/openimages-calibration`


### Download Model
```
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```
Note: wget commands use IPv6 by default, if your system uses IPv4, please add -4 option into the wget command to force it to use IPv4.

### Build & Run Docker container from Dockerfile
If you haven't already done so, build and run the Intel optimized Docker image for Retinanet using:
```
cd <THIS_REPO>/closed/Intel/code/retinanet/pytorch-cpu/docker/

bash build_retinanet_contanier.sh

docker run --name intel_retinanet --privileged -itd --net=host --ipc=host -v ${WORKLOAD_DATA}:/opt/workdir/code/retinanet/pytorch-cpu/data <resnet docker image ID> 

docker exec -it intel_retinanet bash 

cd code/retinanet/pytorch-cpu/
```

### Calibrate and generate torchscript model

Run Calibration
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

### Set Up Environment
If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```
Export the environment settings
```
source setup_env.sh
```

### Run the Benchmark

```

# Run one of these performance or accuracy scripts at a time
# since the log files will be overwritten on each run

# for offline performance
bash run_offline.sh

# for server performance
bash run_server.sh

# for offline accuracy
bash run_offline_accuracy.sh

# for server accuracy
bash run_server_accuracy.sh
```


### Get the results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user.conf file. The scripts will automatically select user_default.conf file to calculate corresponding "target_qps" according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user.conf files.

Check the ``./accuracy.txt`` log file:

* Check the field ``mAP``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

## Get Started with RNNT

If you haven't already done so, build the Intel optimized Docker image for RNNT using:
```
cd <THIS_REPO>/closed/Intel/code/rnnt/pytorch-cpu/docker/
bash build_rnnt-99_container.sh
```

### Set Up Environment
Follow these steps to set up the docker instance.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we built earlier.
```
docker run --name intel_rnnt --privileged -itd -v /data/mlperf_data:/data/mlperf_data --net=host --ipc=host mlperf_inference_rnnt:3.1
```

#### Login to Docker Container
Get the Docker container ID and login into a bashrc shell in the Docker instance using ``docker exec``.

```
docker ps -a #get container "id"
docker exec -it <id> bash
cd /opt/workdir/code/rnnt/pytorch-cpu
```

+ Setup env vars

```
export LD_LIBRARY_PATH=/opt/workdir/code/rnnt/pytorch-cpu/third_party/lib:$LD_LIBRARY_PATH
```

If you need a proxy to access the internet, replace your host proxy with the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

### Run the Benchmark

The provided ``run.sh`` script abstracts the end-to-end process for RNNT:
| STAGE | STEP  |
| ------- | --- | 
| 0 | Download model |
| 1 | Download dataset |
| 2 | Pre-process dataset |
| 3 | Calibration |
| 4 | Build model |
| 5 | Run Offline/Server accuracy & benchmark |

Run ``run.sh`` with ``STAGE=0`` to invoke all the steps requried to run the benchmark (i.e download the model & dataset, preprocess the data, calibrate and build the model):

```
 SKIP_BUILD=1 STAGE=0 bash run.sh
```
or to skip to stage 5 without previous steps: Offline/Server accuracy and benchmark:
```
 SKIP_BUILD=1 STAGE=5 bash run.sh
```

### Get the Results

Check the appropriate offline or server performance log files, either
``./logs/Server/performance/.../mlperf_log_summary.txt`` or
``./logs/Offline/performance/.../mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``
* The performance result is controled by the value of "target_qps" in user.conf file. The scripts will automatically select user_default.conf file to calculate corresponding "target_qps" according to the number of sockets on customer's platform. Customers can also manully change the value of "target_qps" in corresponding user.conf files.

Check the appropriate offline or server accuracy log file, either
``./logs/Server/accuracy/.../mlperf_log_summary.txt`` or
``./logs/Offline/accuracy/.../mlperf_log_summary.txt``:

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

Check the appropriate offline or server accuracy log file, either
``./logs/Server/accuracy/.../mlperf_log_summary.txt`` or
``./logs/Offline/accuracy/.../mlperf_log_summary.txt``:

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

