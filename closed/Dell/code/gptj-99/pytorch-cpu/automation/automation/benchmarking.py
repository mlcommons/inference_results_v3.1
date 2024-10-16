import argparse
import subprocess
import os
import shutil
import os.path as osp
import json
import sys
from typing import Callable
from dir2tree import TreeNode
import sys

print(os.getcwd())

models = {"resnet50", "retinanet", "rnnt","3d-unet-99.9","bert-99","dlrm-99.9","gptj-99","dlrm2-99.9"}
scenarios = {"Offline", "Server"}
tests = {"TEST01", "TEST04", "TEST05"}
results_files = {"mlperf_log_accuracy.json", "mlperf_log_detail.txt", "mlperf_log_summary.txt", "accuracy.txt"}
measurement_files = {"mlperf.conf", "README.md", "user.conf", "run*.sh"}


# benchmark_cmd_args: execute command args when execute easure_relative_path script
   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--BENCHMARK_DIR", type=str, default=None, required=True,
                        help="Benchmarking running directory")
    parser.add_argument("-m", "--model", type=str, default=None, required=True,
                        help="Benchmarking model")
    parser.add_argument("-s", "--scenario", type=str, default=None, required=True,
                        help="Benchmarking scenario, Offline or Server")
    parser.add_argument("-O", "--OUTPUT_DIR", type=str, default=None, required=True,
                        help="Output directory when run benchmark or compliance")
    parser.add_argument("-p", "--performance", action="store_true",
                        help="Run benchmark performance")
    parser.add_argument("-a", "--accuracy", action="store_true",
                        help="Run benchmark accuracy")
    parser.add_argument("-c", "--compliance", action="store_true",
                        help="Run compliance")
    parser.add_argument("-T", "--TEST", type=str, default=None,
                        help="Test mode for compliance")
    parser.add_argument("-N", "--prepare",action="store_true",
                        help="Run envitonment preparation")
    parser.add_argument("-d", "--datatype", type=str, default="int8",
                        help="int8, int4, bf16, for the workload with multiple data type implementations")
 
  #  parser.add_argument("-C", "--check",action="store_true",
  #                      help="Run submission checker")
    args = parser.parse_args()
    return args

args = parse_args()

HOME = os.environ['HOME'] 
RESULTS_DIR = "/data/mlperf_data/results_v3.1/submission-v3-1/closed/Intel/results/1-node-2S-SPR-PyTorch-" + args.datatype.upper() + "/"
COMPLIANCE_OUT_DIR = "/data/mlperf_data/results_v3.1/submission-v3-1/closed/Intel/compliance/1-node-2S-SPR-PyTorch-" + args.datatype.upper() + "/"
MEASUREMENT_DIR = "/data/mlperf_data/results_v3.1/submission-v3-1/closed/Intel/measurements/1-node-2S-SPR-PyTorch-" + args.datatype.upper() + "/"
TEST_DIR = "/data/mlperf_data" + "/inference/compliance/nvidia/"
DATA_DIR = "/data/mlperf_data"

RESULTS_TREE = TreeNode()
with open(osp.join(os.path.split(os.path.realpath(__file__))[0], 'base_info.json'), 'r') as f:
    measure_relative_path, benchmark_cmd_args = json.load(f)
 
def exec_cmd(cmd: Callable, *args, **kargs) -> str: 
    try:
        value = cmd(*args, **kargs)
    except subprocess.CalledProcessError:
        print("Exception occurred trying to execute:\n  " + args[0])
        raise
    except OSError:
        print("Exception occurred trying to copy " + args[0] + " to " + args[1])
        raise
    else:
        if cmd == os.system and value:  # os.system run fail
            raise RuntimeError("Exception occurred trying to execute:\n  " + args[0])
        if cmd == shutil.copy2:
            print("Stored output file: " + args[0] + " in: " + args[1])
        return value
    
def check_fields_in_file(file_path, field1, field2):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if field1 in line and field2 in line:
                    return True
        return False
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False

def benchmark(BENCHMARK_DIR: str, model: str, scenario: str, OUTPUT_DIR: str, mode: str):
    """
    Run benchmarking.

    Args:
        BENCHMARK_DIR: benchmark result directory
        model: benchmark model
        scenario: benchmark scenario
        OUTPUT_DIR: benchmark result output directory
        mode: "performance" or "accuracy"

    Returns:
        None
    """
    global RESULTS_DIR
    
    RESULTS_DIR = osp.join(RESULTS_DIR, model, scenario, mode)
    RESULTS_DIR += "/run_1" if mode == "performance" else ''

    # run benchmarking
    os.chdir(BENCHMARK_DIR)
    model_name = model
    if args.datatype != "int8":
       model_name = model + "-" + args.datatype
 
    if model == "dlrm-99.9":
       benchmark_cmd = " pwd && source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh && bash " + BENCHMARK_DIR + "/" + measure_relative_path[model_name][scenario][mode] + ' ' + benchmark_cmd_args[model_name][scenario][mode]
    elif model == "rnnt":
       benchmark_cmd = " pwd && " + benchmark_cmd_args[model_name][scenario][mode] + ' bash ' + measure_relative_path[model_name][scenario][mode]
    elif model == "dlrm2-99.9":
      benchmark_cmd = " pwd && bash " + BENCHMARK_DIR + "/" + measure_relative_path[model_name][scenario][mode] + ' ' + benchmark_cmd_args[model_name][scenario][mode]
       #benchmark_cmd = " pwd && ./run_main.sh"+' ' + benchmark_cmd_args[model][scenario]["performance"]
    else:
       benchmark_cmd = " pwd && bash " + BENCHMARK_DIR + "/" + measure_relative_path[model_name][scenario][mode] + ' ' + benchmark_cmd_args[model_name][scenario][mode]

    print("==> Runing {} {} {} {} benchmarking {}".format(model, scenario, mode, args.datatype, benchmark_cmd))
    exec_cmd(os.system, benchmark_cmd)

    # check whether the benchmarking is valid, otherwise exit program
    print("==> OUTPUT_DIR is " + OUTPUT_DIR)
    if not osp.exists(OUTPUT_DIR):
        raise FileNotFoundError("OUTPUT_DIR didn't exist!")

    if mode == "performance":
        
        check_valid_cmd1 = ["cat", OUTPUT_DIR + "/mlperf_log_summary.txt"]
        check_valid_cmd2 = ["awk", "{print $4}"]
        check_valid_cmd3 = ["grep", "-x", "VALID"]

        # Create each subprocess.Popen instance and chain them together.
        process1 = subprocess.Popen(check_valid_cmd1, stdout=subprocess.PIPE)
        process2 = subprocess.Popen(check_valid_cmd2, stdin=process1.stdout, stdout=subprocess.PIPE)
        process1.stdout.close()  # Allow process1 to receive a SIGPIPE if process2 exits.
        process3 = subprocess.Popen(check_valid_cmd3, stdin=process2.stdout, stdout=subprocess.PIPE)
        process2.stdout.close()  # Allow process2 to receive a SIGPIPE if process3 exits.
        output, error = process3.communicate()
        output_str = output.decode('utf-8')
        
        if "VALID" in output_str:
            print("{} {} {} benchmarking is VALID".format(model, scenario, mode))
        else:
            #raise ValueError("{} {} {} benchmarking is INVALID".format(model, scenario, mode))
            sys.exit("{} {} {} benchmarking is INVALID".format(model, scenario, mode))

        #exec_cmd(os.system,check_valid_cmd)

    # compare storing result and output result if run performance
    # NOTE: need to consider the accuracy mode and when to store
    if not osp.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if mode == "performance" and osp.exists(osp.join(RESULTS_DIR, "mlperf_log_summary.txt")):
        get_qps_cmd = "cat " + RESULTS_DIR + "/mlperf_log_summary.txt | grep \"per second\""
        refer_qps = float(exec_cmd(subprocess.getoutput, get_qps_cmd).split("\n")[0].split(' ')[-1])

        get_qps_cmd = "cat " + OUTPUT_DIR + "/mlperf_log_summary.txt | grep \"per second\""
        output_qps = float(exec_cmd(subprocess.getoutput, get_qps_cmd).split("\n")[0].split(' ')[-1])

        print("Refer_QPS: {}\nOutput_QPS: {}\n".format(refer_qps, output_qps))

        #if refer_qps > output_qps:  # give up storing
        #    print("Refer_QPS is greater than output_QPS, give up storing!") 
        #    return
    
    # save output file to RESULTS_DIR
    for file in results_files:
        output_file = osp.join(OUTPUT_DIR, file) 

        if not osp.exists(output_file):
            continue
        
        exec_cmd(shutil.copy2, output_file, RESULTS_DIR)

        # print information needed
        if mode == "performance" and file == "mlperf_log_summary.txt":
            get_target_qps_cmd = "cat " + OUTPUT_DIR + "/mlperf_log_summary.txt | grep \"target_qps\""
            target_qps = float(exec_cmd(subprocess.getoutput, get_target_qps_cmd).split(' ')[-1])

            get_qps_cmd = "cat " + OUTPUT_DIR + "/mlperf_log_summary.txt | grep \"per second\""
            output_qps = float(exec_cmd(subprocess.getoutput, get_qps_cmd).split("\n")[0].split(' ')[-1])

            get_percentile_cmd = "cat " + OUTPUT_DIR + "/mlperf_log_summary.txt | grep \"99.00 percentile latency\""
            percentile = float(exec_cmd(subprocess.getoutput, get_percentile_cmd).split(' ')[-1])

            print("\nTarget QPS: {}\nPerf QPS: {}\n99.00 percentile latency: {}\n".format(target_qps, output_qps, percentile))
    
def compliance(BENCHMARK_DIR: str, model: str, scenario: str, OUTPUT_DIR: str, TEST: str):
    """
    Run compliance test.

    Args:
        BENCHMARK_DIR: benchmark result directory
        model: benchmark model
        scenario: benchmark scenario
        OUTPUT_DIR: compliance result output directory
        TSET: compliance test
    
    Returns:
        None
    """
    global RESULTS_DIR, COMPLIANCE_OUT_DIR, TEST_DIR

    RESULTS_DIR = osp.join(RESULTS_DIR, model, scenario)
    COMPLIANCE_OUT_DIR = osp.join(COMPLIANCE_OUT_DIR, model, scenario)
    TEST_DIR = osp.join(TEST_DIR, TEST)

    if not osp.exists(COMPLIANCE_OUT_DIR):
        os.makedirs(COMPLIANCE_OUT_DIR)

    # copy audit.config to benchmark directory 
    audit_path = TEST_DIR
    if TEST == "TEST01": 
        if model == "bert-99":
            audit_path=osp.join(audit_path,"bert")
        elif model == "3d-unet-99.9":
            audit_path = osp.join(audit_path,"3d-unet")
        elif model == "dlrm-99.9":
            audit_path = osp.join(audit_path,"dlrm")
        elif model == "dlrm2-99.9":
            audit_path = osp.join(audit_path, "dlrm-v2")
        else:
            audit_path = osp.join(audit_path, model)

    audit_path = osp.join(audit_path, "audit.config")

    exec_cmd(shutil.copy2, audit_path, BENCHMARK_DIR)

    # run compliance benchmarking
    os.chdir(BENCHMARK_DIR)
    model_name = model
    if args.datatype != "int8":
       model_name = model + "-" + args.datatype
 
    if model == "dlrm-99.9":
       benchmark_cmd = " pwd && source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh && bash " + BENCHMARK_DIR + "/" + measure_relative_path[model_name][scenario]["performance"] + ' ' + benchmark_cmd_args[model_name][scenario]["performance"]
    elif model == "rnnt":
       benchmark_cmd = " pwd && " + benchmark_cmd_args[model_name][scenario]["performance"] + ' bash ' + measure_relative_path[model_name][scenario]["performance"]
    elif model == "dlrm2-99.9":
       #benchmark_cmd = " pwd && ." + BENCHMARK_DIR + "/" + measure_relative_path[model][scenario]["performance"] + ' ' + benchmark_cmd_args[model][scenario]["performance"]
       benchmark_cmd = " pwd && ./run_main.sh"+' ' + benchmark_cmd_args[model_name][scenario]["performance"]
    else:
       benchmark_cmd = " pwd && bash " + BENCHMARK_DIR + "/" + measure_relative_path[model_name][scenario]["performance"] + ' ' + benchmark_cmd_args[model_name][scenario]["performance"]

    print("==> Runing {} {} compliance benchmarking".format(model, scenario))
    exec_cmd(os.system, benchmark_cmd)

    # check whether audit in mlperf_log_detail.txt
    if not osp.exists(OUTPUT_DIR):
        raise FileNotFoundError("OUTPUT_DIR didn't exist!")
    
    check_audit_cmd1 = ["cat", OUTPUT_DIR + "/mlperf_log_detail.txt"]
    check_audit_cmd2 = ["grep", "audit"]
    try:
        # Create each subprocess.Popen instance and chain them together.
        process1 = subprocess.Popen(check_audit_cmd1, stdout=subprocess.PIPE)
        process2 = subprocess.Popen(check_audit_cmd2, stdin=process1.stdout, stdout=subprocess.PIPE)
        process1.stdout.close()  # Allow process1 to receive a SIGPIPE if process2 exits.
        output, error = process2.communicate()
    except Exception:
        print("Exception occurred trying to execute:\n  " + ' | '.join([' '.join(check_audit_cmd1), ' '.join(check_audit_cmd2)]))
        raise
    finally:
        os.remove(BENCHMARK_DIR + "/audit.config")

    # run compliance
    print("==> Runing {} {} compliance".format(model, scenario))
    compliance_cmd = "python3 " + TEST_DIR + "/run_verification.py -r " +  RESULTS_DIR + " -c " + OUTPUT_DIR + " -o " + COMPLIANCE_OUT_DIR
    exec_cmd(os.system, compliance_cmd)
    # TEST01 part3:
    if TEST == "TEST01" and check_fields_in_file(COMPLIANCE_OUT_DIR+"/TEST01/verify_accuracy.txt","TEST","PASS") == False:
        compliance_cmd = "bash " + TEST_DIR + "/create_accuracy_baseline.sh " +RESULTS_DIR + "/accuracy/mlperf_log_accuracy.json "+OUTPUT_DIR+ "/mlperf_log_accuracy.json"
        exec_cmd(os.system, compliance_cmd)
        if model == "gptj-99":
            compliance_cmd="python evaluation.py --mlperf-accuracy-file ./mlperf_log_accuracy_baseline.json " \
                            "--dataset-file ./data/validation-data/cnn_dailymail_validation.json " \
                            "--model-name-or-path ./data/gpt-j-checkpoint/ " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
            compliance_cmd="python evaluation.py --mlperf-accuracy-file "+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "--dataset-file ./data/validation-data/cnn_dailymail_validation.json " \
                            "--model-name-or-path ./data/gpt-j-checkpoint/ " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt "
            exec_cmd(os.system,compliance_cmd)
        if model == "dlrm2-99.9":
            compliance_cmd="python tools/accuracy-dlrm.py --mlperf-accuracy-file ./mlperf_log_accuracy_baseline.json " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
            compliance_cmd="python tools/accuracy-dlrm.py --mlperf-accuracy-file "+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
        if model == "rnnt":
            compliance_cmd="python -u eval_accuracy.py "\
                            "--log_path=./mlperf_log_accuracy_baseline.json " \
                            "--manifest_path="+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
            compliance_cmd="python -u eval_accuracy.py "\
                            "--log_path="+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "--manifest_path="+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
        if model == "retinanet":
            compliance_cmd="python -u retinanet-env/mlperf_inference/vision/classification_and_detection/tools/accuracy-openimages.py "\
                            "--mlperf-accuracy-file ./mlperf_log_accuracy_baseline.json " \
                            "--openimages-dir data/openimages 2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
            compliance_cmd="python -u retinanet-env/mlperf_inference/vision/classification_and_detection/tools/accuracy-openimages.py "\
                            "--mlperf-accuracy-file"+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "--openimages-dir data/openimages 2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
        if model == "resnet50":
            compliance_cmd="python -u rn50-mlperf/mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py " \
                            "--mlperf-accuracy-file ./mlperf_log_accuracy_baseline.json " \
                            "--imagenet-val-file ILSVRC2012_img_val/val_map.txt " \
                            "dtype int32 2>&1|tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
            compliance_cmd="python -u rn50-mlperf/mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py " \
                            "--mlperf-accuracy-file "+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "--imagenet-val-file ILSVRC2012_img_val/val_map.txt " \
                            "dtype int32 2>&1|tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system,compliance_cmd)
        if model == "bert-99":
            compliance_cmd="python ./inference/language/bert/accuracy-squad.py " \
                            "--vocab_file "+DATA_DIR+"/bert/model/vocab.txt " \
                            "--val_data "+DATA_DIR+"/bert/dataset/dev-v1.1.json " \
                            "--log_file ./mlperf_log_accuracy_baseline.json " \
                            "--out_file predictions.json " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system, compliance_cmd)
            compliance_cmd="python ./inference/language/bert/accuracy-squad.py " \
                            "--vocab_file "+DATA_DIR+"/bert/model/vocab.txt " \
                            "--val_data "+DATA_DIR+"/bert/dataset/dev-v1.1.json " \
                            "--log_file "+OUTPUT_DIR +"/mlperf_log_accuracy.json " \
                            "--out_file predictions.json " \
                            "2>&1 | tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system, compliance_cmd)
        if model == "3d-unet-99.9":
            compliance_cmd="python3 accuracy_kits.py --log_file=./mlperf_log_accuracy_baseline.json 2>&1|tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/baseline_accuracy.txt"
            exec_cmd(os.system, compliance_cmd)
            compliance_cmd="python3 accuracy_kits.py --log_file="+OUTPUT_DIR +"/mlperf_log_accuracy.json 2>&1|tee "+COMPLIANCE_OUT_DIR+"/TEST01/accuracy/compliance_accuracy.txt"
            exec_cmd(os.system, compliance_cmd)            
    print("Finished {} {} {} compliance".format(model, scenario, TEST))

def update_measurement(BENCHMARK_DIR: str, model: str, scenario: str, mode: str):
    """
    Update measurement file in MEASUREMENT_DIR.

    Args:
        BENCHMARK_DIR: benchmark result directory
        model: benchmark model
        scenario: benchmark scenario
        mode: "performance" or "accuracy"

    Returns:
        None
    """
    global MEASUREMENT_DIR

    MEASUREMENT_DIR = osp.join(MEASUREMENT_DIR, model, scenario)

    if model == "bert-99":
        cmd="cp " + BENCHMARK_DIR + "/inference/mlperf.conf " + BENCHMARK_DIR
        exec_cmd(os.system,cmd)


    if not osp.exists(MEASUREMENT_DIR):
        os.makedirs(MEASUREMENT_DIR)

    # copy measurement files to MEASUREMENT_DIR
    for file in measurement_files:
        if file == "mlperf.conf" or file == "user.conf":
            original_file = osp.join(BENCHMARK_DIR, measure_relative_path[model]["path"], file)
        elif file == "README.md":
            original_file = osp.join(BENCHMARK_DIR, file)
        else:
            original_file = osp.join(BENCHMARK_DIR, measure_relative_path[model][scenario][mode])

        exec_cmd(shutil.copy2, original_file, MEASUREMENT_DIR)  # cover if exist

    # compare the copied user.conf target QPS with mlperf_log_summary.txt stored in RESULTS_DIR, sed if needed
    if mode == "accuracy":
        return

    user_conf = osp.join(MEASUREMENT_DIR, "user.conf")

    copy_target_qps_cmd = "cat " + user_conf + " | grep -n \"" + scenario + ".target_qps\""
    exec_return = exec_cmd(subprocess.getoutput, copy_target_qps_cmd)
    line, copy_target_qps = exec_return.split(':')[0], float(exec_return.split('=')[-1])
    
    """ print("=> Finding result storage path now:")
    path = RESULTS_TREE.find_path(model, scenario, mode)
    if len(path) != 1:
        raise ValueError("More than one path or no path found, please check the args!")
    else:
        print("Found path {}".format(path[0]))

    storage_target_qps_cmd = "cat " + path[0] + "/run_1/mlperf_log_summary.txt | grep \"target_qps\""
    storage_target_qps = float(exec_cmd(subprocess.getoutput, storage_target_qps_cmd).split(' ')[-1])

    print("\nCopy target QPS: {}\nStored target QPS: {}\n".format(copy_target_qps, storage_target_qps))

    if storage_target_qps > copy_target_qps:    # not update
        sed_cmd = "sed -i \"" + line + "s/[0-9].\+/" + str(storage_target_qps) + "/g\" " + user_conf

        if not exec_cmd(os.system, sed_cmd):    # run success
            print("Replace {} {}.target_qps with {}".format(user_conf, scenario, storage_target_qps)) """

def preparation(BENCHMARK_DIR: str,model: str,scenario: str):
    if model == "bert-99":
        preparation_cmd="bash " + BENCHMARK_DIR + "/convert.sh"
        exec_cmd(os.system,preparation_cmd)
        print("Data preparation is complete")
        print("{} preparation is complete".format(model))
    if model == "dlrm-99.9":
        #preparation_cmd="cp -r " + BENCHMARK_DIR + "/python " + BENCHMARK_DIR + "/automation/"
        #exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/dump_model_dataset.sh"
        exec_cmd(os.system,preparation_cmd)
        print("Data preparation is complete")
        print("{} preparation is complete".format(model))
    if model == "3d-unet-99.9":
        preparation_cmd="rm -rf "+ BENCHMARK_DIR + "/build/model/3dunet_kits19_pytorch_checkpoint.pth"
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/process_data_model.sh"
        exec_cmd(os.system,preparation_cmd)
        print("Data preparation is complete")
        print("{} preparation is complete".format(model))
    if model == "resnet50":
        #preparation_cmd="bash " + BENCHMARK_DIR + "/download_imagenet.sh"
        #exec_cmd(os.system,preparation_cmd)
        preparation_cmd="cp -f " + BENCHMARK_DIR + "/val_data/*.txt /data/mlperf_data/resnet50/ILSVRC2012_img_val/"
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="cp -rf " + DATA_DIR + "/resnet50/* " + BENCHMARK_DIR + "/."
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/prepare_calibration_dataset.sh"
        exec_cmd(os.system,preparation_cmd)
        #preparation_cmd="bash " + BENCHMARK_DIR + "/download_model.sh"
        #exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/generate_torch_model.sh"
        exec_cmd(os.system,preparation_cmd)
        print("{} preparation is complete".format(model))
    if model == "retinanet":
        preparation_cmd="bash " + BENCHMARK_DIR + "/openimages_mlperf.sh --dataset-path /opt/workdir/code/retinanet/pytorch-cpu/data/openimages"
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/openimages_calibration_mlperf.sh --dataset-path /opt/workdir/code/retinanet/pytorch-cpu/data/openimages-calibration"
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/run_calibration.sh"
        exec_cmd(os.system,preparation_cmd)
        preparation_cmd="source " + BENCHMARK_DIR + "/setup_env.sh"
        exec_cmd(os.system,preparation_cmd)
        print("{} preparation is complete".format(model))
    if model == "rnnt":
        preparation_cmd="SKIP_BUILD=1 STAGE=1 SINGLE_PHASE=yes bash " + BENCHMARK_DIR + "/run.sh"
        exec_cmd(os.system,preparation_cmd)
        print("{} preparation is complete".format(model))
    if model == "gptj-99":
        if args.datatype == "int4":
            preparation_cmd="bash " + BENCHMARK_DIR + "/run_quantization_int4.sh"
            exec_cmd(os.system,preparation_cmd)
        print("{} preparation is complete".format(model))
    if model == "dlrm2-99.9":
        # if scenario == "Server":
        #     preparation_cmd=". ./setup_env_server.sh"
        # if scenario == "Offline":
        #     preparation_cmd=". ./setup_env_offline.sh"
        # exec_cmd(os.system,preparation_cmd)
        preparation_cmd="bash " + BENCHMARK_DIR + "/run_calibration.sh"
        exec_cmd(os.system,preparation_cmd)
       
        print("{} preparation is complete".format(model))
    

""" def check(BENCHMARK_DIR: str,model: str):
    check_cmd="rm -rf /data/mlperf_data/results_v3.1/new-submission-v3-1"
    exec_cmd(os.system,check_cmd)
    check_cmd="python3 ~/inference/tools/submission/truncate_accuracy_log.py --input /data/mlperf_data/results_v3.1/submission-v3-1 --submitter Intel --output /data/mlperf_data/results_v3.1/new-submission-v3-1"  
    exec_cmd(os.system,check_cmd)
    #check_cmd="python3 ~/inference/tools/submission/submission_checker.py --input ~/new-submission-v3-1"
    #exec_cmd(os.system,check_cmd)
    os.system("python3 ~/inference/tools/submission/submission_checker.py --input /data/mlperf_data/results_v3.1/new-submission-v3-1")
 """


def main(args):
    global RESULTS_TREE

    # check whether the args is valid
    if args.BENCHMARK_DIR and not osp.exists(args.BENCHMARK_DIR):
        raise FileNotFoundError("BENCHMARK_DIR didn't exist!")

    if args.model and args.model not in models:
        raise ValueError("Model not Found!")

    if args.scenario and args.scenario not in scenarios:
        raise ValueError("Scenario not Found!")
        
    if args.TEST and args.TEST not in tests:
        raise ValueError("TEST didn't exist!")
          
    # build RESULTS tree
    if not osp.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    RESULTS_TREE = TreeNode.build_tree(RESULTS_DIR)
    # TreeNode.print_tree(RESULTS_TREE)

    # run benchmarking or compliance
    if args.prepare:
        preparation(args.BENCHMARK_DIR,args.model,args.scenario)

    if args.performance:
        benchmark(args.BENCHMARK_DIR, args.model, args.scenario, args.OUTPUT_DIR, "performance")
        RESULTS_TREE.update_tree()
        update_measurement(args.BENCHMARK_DIR, args.model, args.scenario, "performance")

    if args.accuracy:
        benchmark(args.BENCHMARK_DIR, args.model, args.scenario, args.OUTPUT_DIR, "accuracy")
        RESULTS_TREE.update_tree()
        update_measurement(args.BENCHMARK_DIR, args.model, args.scenario, "accuracy")

    if args.compliance and "gptj" not in args.model:
        compliance(args.BENCHMARK_DIR, args.model, args.scenario, args.OUTPUT_DIR, args.TEST)

    #if args.check:
       # check(args.BENCHMARK_DIR, args.model)


    # TreeNode.print_tree(RESULTS_TREE)

if __name__ == "__main__":
    main(args)
