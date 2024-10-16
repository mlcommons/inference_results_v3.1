# Automation test

## Step1 download data

```
bash download_dataset.sh <model> <location> <conda_path>
 #model: specify a single model name among {resnet50,retinanet,rnnt,3d-unet-99.9,bert-99,gptj-99,dlrm2-99.9} to download individual dataset, or insert `all` to download datasets for all models except DLRM2
 #location: the location to put data and model, such as `/data/mlperf_data`
 #conda_path: optional, only for GPTJ, for example the path to your miniconda3/bin

1.Please note that, run download_dataset.sh before proceeding to the next steps
2.If downloading by yourself, please keep the same directory structure as in download_dataset.sh
3.The data set and model of dlrm2 need to be downloaded by yourself.For the download method, refer to the README of dlrm2's code.
Please put the data of dlrm2 into <data_path>/dlrm_2/data_npy/, and put the model into <data_path>/dlrm_2/model/{int8,bf16}/
e.g., /data/mlperf_data/dlrm_2/model/bf16/dlrm-multihot-pytorch.pt   and  /data/mlperf_data/dlrm_2/model/int8/dlrm-multihot-pytorch_int8.pt

For example, if you want to download the data for resnet50 to the directory /data/mlperf_data, then:
    bash download_dataset.sh resnet50 /data/mlperf_data

```

## Step2 run test

```

 bash run.sh <model> <location> <postfix>
  
 #Fill in the parameters to start the test:
 #model: resnet50,retinanet,rnnt,3d-unet-99.9,bert-99,gptj-99,dlrm2-99.9
 #location: the dataset location specified by "download_dataset.sh"
 #postfix: a postfix for container for differentiation

Other Optional Parameters:
 #PerformanceOnly: True,False(default:False)
 #AccuracyOnly: True,False(default:False)
 #OfflineOnly: True,False(default:False)
 #ServerOnly: True,False(default:False)
 #ComplianceOnly: True,False(default:False)
 #Skip_docker_build: True,False(default:False)
 #Skip_data_proprocess: True,False(default:False)

For example, if you just want to test accuracy, then:
  AccuracyOnly="True" bash run.sh <model> <location> <postfix>
Or if the docker container has been built and the data has been processed, you can skip these steps by:
  Skip_docker_build="True" Skip_data_proprocess="True" bash run.sh <model> <location> <postfix>

```

## log location

```
in docker:
cd /data/mlperf_data/results_v3.1

```

## running information:

```
in docker:
/data/mlperf_data/information_${model}.txt
/data/mlperf_data/information_${model}_2.txt
```

## submission checker
1. Clone the official MLCommons® inference repo into the current `automation` directory

```
git clone https://github.com/mlcommons/inference
```

2. Merge all the log files in `compliance`, `measurements` and `results` into the above inference repository

3. Create a `systems` folder in the `inference` directory, and create `<system_desc_id>.json` files in the `systems` folder to indicate both hardware and software stack information. Naming example: `1-node-2S-SPR-PyTorch-INT8.json`

Let's also take Intel as an example, fill in all the blanks in the json file and replace `submitter` as your company name:


```
{
    "division": "closed",
    "submitter": "Intel",
    "status": "",
    "system_type":"",
    "system_type_detail":"N/A",
    "system_name": "",

    "number_of_nodes": "",
    "host_processor_model_name": "",
    "host_processors_per_node": "",
    "host_processor_core_count": "",
    "host_processor_frequency": "N/A",
    "host_processor_caches": "N/A",
    "host_memory_configuration": "",
    "host_memory_capacity": "",
    "host_storage_capacity": "N/A",
    "host_storage_type": "N/A",
    "host_processor_interconnect": "N/A",
    "host_networking": "N/A",
    "host_networking_topology": "N/A",
    "host_networking_card_count": "N/A",
    "accelerators_per_node": "",
    "accelerator_model_name": "N/A",
    "accelerator_frequency": "N/A",
    "accelerator_host_interconnect": "N/A",
    "accelerator_interconnect": "N/A",
    "accelerator_interconnect_topology": "N/A",
    "accelerator_memory_capacity": "N/A",
    "accelerator_memory_configuration": "N/A",
    "accelerator_on-chip_memories": "N/A",
    "cooling": "",
    "hw_notes": "",

    "framework": "",
    "operating_system": "",
    "other_software_stack": "",
    "sw_notes": "N/A"
}
```

4. Execute the submission checker

```
bash submission_checker.sh . <submitter> 
```
