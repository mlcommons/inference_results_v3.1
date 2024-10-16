###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import yaml
import typing
import subprocess
import logging
import sys
import json
import shutil
import re
import os
from pathlib import Path
import time

scenarios_config = yaml.full_load(open("scenarios.yaml"))
logging.basicConfig(level=logging.INFO)

modes = ["Server", "Offline"]
system_desc_id = "HLS-Gaudi2-PT"
implementation_id = "PyTorch"


def get_configuration(scenarios) -> typing.Tuple[str, str]:
    runs = []
    for scenario in scenarios:
        if scenario in scenarios_config["scenarios"]:
            for mode in modes:
                runs.append((scenario, mode))
        else:
            try:
                scenario, mode = scenario.split("_")
                assert mode in modes
                runs.append((scenario, mode))
            except:
                logging.error(
                    f"Scenario {scenario} not supported, see scenarios.yaml to view supported scenarios"
                )
                exit()
    return runs


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="List of scenarios e.g. gpt-j_Server or gpt-j separated by space, to run all possible scenarios set first element to 'all'",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./results",
        help="Path to save results folder in",
    )
    parser.add_argument(
        "--mlperf-path", required=True, help="Path to mlperf inference directory"
    )
    parser.add_argument("--systems-dir-path", required=True)
    parser.add_argument("--measurements-dir-path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    configuration = get_configuration(args.scenarios)

    output_dir = Path(args.output_dir).absolute()
    logs_dir = output_dir / "logs"
    # for reference https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#563-inference
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(exist_ok=True)

    division_dir = submission_dir / "closed"
    division_dir.mkdir(exist_ok=True)
    company_dir = division_dir / "Intel-HabanaLabs"
    company_dir.mkdir(exist_ok=True)

    code_dir = company_dir / "code"
    code_dir.mkdir(exist_ok=True)

    results_dir = company_dir / "results"
    results_dir.mkdir(exist_ok=True)

    systems_dir = company_dir / "systems"
    systems_dir.mkdir(exist_ok=True)

    measurements_dir = company_dir / "measurements"
    measurements_dir.mkdir(exist_ok=True)

    mlperf_path = Path(args.mlperf_path)
    # for each run
    for scenario, mode in configuration:
        benchmark = scenarios_config["scenarios"][scenario]["benchmark"]

        # systems dir
        shutil.copyfile(
            f"{args.systems_dir_path}/{system_desc_id}.json",
            systems_dir / f"{system_desc_id}.json",
        )

        # code dir
        current_dir = os.getcwd()
        shutil.copytree(
            current_dir,
            code_dir / benchmark,
            ignore=shutil.ignore_patterns(
                ".graph_dumps", "__pycache__", ".gitignore", "internal", output_dir, "results"
            ),
            dirs_exist_ok=True,
        )
        # move general README.md out of benchmark to code directory
        shutil.move(
                code_dir / benchmark / "README.md",
                code_dir / "README.md"
            )

        # measurements dir
        measurements_dir_path = Path(args.measurements_dir_path)
        Path(measurements_dir / system_desc_id / benchmark / mode).mkdir(
            exist_ok=True, parents=True
        )
        shutil.copytree(
            measurements_dir_path / benchmark,
            measurements_dir / system_desc_id / benchmark,
            dirs_exist_ok=True,
        )
        code_dir_path = Path(scenarios_config["scenarios"][scenario]["code_dir"])
        shutil.copyfile(
            code_dir_path / "mlperf.conf",
            measurements_dir / system_desc_id / benchmark / mode / "mlperf.conf",
        )
        shutil.copyfile(
            measurements_dir_path / "calibration_process.md",
            measurements_dir / system_desc_id / benchmark / mode / "calibration_process.md",
        )
        if benchmark == "gptj-99.9":
            config_file = "fp8-99.9.conf"
        else:
            config_file = "fp8-99.conf"

        shutil.copyfile(
            code_dir_path / "configs" / config_file,
            measurements_dir / system_desc_id / benchmark / mode / "user.conf",
        )

        # results dir
        shutil.copytree(
            logs_dir / scenario / mode / "accuracy",
            results_dir / system_desc_id / benchmark / mode / "accuracy",
            ignore=shutil.ignore_patterns("mlperf_log_trace.json"),
        )
        shutil.copytree(
            logs_dir / scenario / mode / "performance",
            results_dir / system_desc_id / benchmark / mode / "performance",
            ignore=shutil.ignore_patterns(
                "mlperf_log_trace.json", "mlperf_log_accuracy.json"
            ),
        )

    #truncate accuracy logs
    accuracy_logs_backup = output_dir / "backup"
    command = f"python {mlperf_path / 'tools/submission/truncate_accuracy_log.py'} --input {submission_dir} --submitter Intel-HabanaLabs --backup {accuracy_logs_backup}"
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        sys.exit("Failed truncating logs")

    # submission checker
    command = f"python {mlperf_path / 'tools/submission/submission_checker.py'} --input {submission_dir} --csv {output_dir / 'summary.csv'}"
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        sys.exit("Submission checker failed")

    # zip submission folder
    command = f"tar -cvzf {output_dir}/submission.gz -C {os.path.dirname(submission_dir)} {os.path.basename(submission_dir)}"
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        sys.exit("Failed packaging submission folder")


if __name__ == "__main__":
    main()
