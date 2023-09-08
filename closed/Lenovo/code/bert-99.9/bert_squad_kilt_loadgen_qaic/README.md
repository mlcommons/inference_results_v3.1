# MLPerf Inference - Language Models - KILT
This implementation runs language models with the KILT backend.

Currently it supports the following models:
- bert-99
- bert-99.9

# Setup Global Variables
```
export BENCHMARK=bert
export WORKDIR=/local/mnt/workspace
```

# Setup Docker Images
Please follow instructions [here](https://github.com/krai/axs2qaic-docker) to build the relevant docker images.

# Setup Local Environment
Install axs in `$WORKDIR/$USER`.
```
git clone --branch master https://github.com/krai/axs $WORKDIR/$USER/axs
```

```
echo "

# AXS
export PATH='$PATH:$WORKDIR/$USER/axs'
export AXS_WORK_COLLECTION='$WORKDIR/$USER/work_collection' 

" >> ~/.bashrc
```

```
source ~/.bashrc
```

Upon sucessful installation.
```
user@laptop:~/axs$ axs
DefaultKernel{}
```

Import these repos into your work_collection using HTTPS.
```
axs byquery git_repo,collection,repo_name=axs2qaic
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2config
axs byquery git_repo,collection,repo_name=axs2system
axs byquery git_repo,repo_name=kilt-mlperf
```

Obtain the latest image name. For example:
```
elim@dyson:~/axs2qaic-docker-dev$ docker image ls
REPOSITORY                  TAG                   IMAGE ID       CREATED        SIZE
krai/axs.bert               deb_latest_20230731   5b8592b6fbf0   2 hours ago    18.7GB
```

Set image name.
```
export IMAGE_NAME=krai/axs.${BENCHMARK}:deb_latest_${DATE}
```

Launch docker container
```
mkdir -p ${WORKDIR}/${USER}/axs_experiment_${BENCHMARK} && \

docker run -it --name ${USER}_${BENCHMARK} --entrypoint /bin/bash --privileged --group-add $(getent group qaic | cut -d: -f3) ${IMAGE_NAME}
```


# Run Experiment inside Docker
Use one of the assembled SUT in [here](https://github.com/krai/axs2config), or you can customize your own. For instance, if you would like to benchmark q2_pro_dc:
```
export SUT=q2_pro_dc
```

### bert-99
Compile the model for qaic devices
```
axs byquery sut_name=${SUT},kilt_ready,device=qaic,model_name=bert-99,loadgen_scenario=Offline
```

Measure Accuracy (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,loadgen_dataset_size=20,set_device_id=0 , get accuracy
```

Measure Accuracy (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy
```

Measure Performance (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1,loadgen_dataset_size=20 , parse_summary
```

Measure Performance (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=</measured value/> , parse_summary
```

### bert-99.9
Compile the model for qaic devices
```
axs byquery sut_name=${SUT},kilt_ready,device=qaic,model_name=bert-99.9,loadgen_scenario=SingleStream
```

Measure Accuracy (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,loadgen_dataset_size=20,set_device_id=0 , get accuracy
```

Measure Accuracy (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy
```

Measure Performance (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1,loadgen_dataset_size=20 , parse_summary
```

Measure Performance (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=qaic,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=/<measured value/> , parse_summary
```