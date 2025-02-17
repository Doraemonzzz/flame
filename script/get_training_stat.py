import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def print_array(array):
    string = array[0]
    for col in array[1:]:
        string += f",{col}"

    print(string)


def print_dict(array, dict):
    string = f"{dict[array[0]]}"
    for col in array[1:]:
        string += f",{dict[col]}"

    print(string)


def get_stat(log_file, cols):
    res = {}
    for col in cols:
        res[col] = 0
    res["memory"] = []
    res["tgs"] = []
    loss = []
    memory_tgs_pattern = r"memory:\s*(\d+\.?\d*)GiB.*?tgs:\s*(\d+(?:,\d+)*)"
    token_loss_pattern = r"token:\s*([\d,]+\.\d+)\s+loss:\s*(\d+\.\d+)"
    mode_pattern = r"'mode':\s*'(\w+)'"
    selective_pattern = r"'selective_ac_option':\s*'(\d+)'"
    finish = False

    earliest_time = None
    latest_time = None

    with open(log_file, "r") as f:
        for line in f.readlines():
            if "Training completed" in line:
                finish = True

            if "INFO" in line:
                time_match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
                if time_match:
                    try:
                        current_time = datetime.strptime(
                            time_match.group(1), "%Y-%m-%d %H:%M:%S"
                        )

                        # 更新最早和最晚时间
                        if earliest_time is None or current_time < earliest_time:
                            earliest_time = current_time
                        if latest_time is None or current_time > latest_time:
                            latest_time = current_time
                    except ValueError:
                        continue

            for col in ["model_source", "model_type"]:
                if col in line:
                    res[col] = line.split()[-1].strip(",").strip("'").strip('"')
            for col in ["parameters"]:
                if col in line and res[col] == 0:
                    res[col] = round(float(line.split()[-1].replace(",", "")) / 1e9, 4)
            for col in [
                "tgs",
            ]:
                if col in line:
                    match = re.search(memory_tgs_pattern, line)
                    if match:
                        res["memory"].append(float(match.group(1)))
                        res["tgs"].append(float(match.group(2).replace(",", "")))
            if "loss" in line:
                match = re.search(token_loss_pattern, line)
                if match:
                    res["token"] = round(
                        float(match.group(1).replace(",", "")) / 1e9, 2
                    )
                    res["loss"] = round(float(match.group(2)), 4)
                    loss.append(res["loss"])
            for col in [
                "batch_size",
                "seq_len",
                "context_len",
                "'compile'",
                "tensor_parallel_degree",
                "'eps'",
                "varlen",
                "warmup_steps",
            ]:
                if col in line:
                    res[col] = line.split()[-1].strip(",")

            if '"steps"' in line:
                res["steps"] = line.split()[-1].strip(",")

    res["memory"] = round(np.mean(res["memory"][-10:]), 2)
    res["tgs"] = round(np.mean(res["tgs"][-10:]), 2)
    res["loss_avg_5"] = round(np.mean(loss[-5:]), 4)

    # get time
    try:
        duration = latest_time - earliest_time
        time = duration.total_seconds() / 3600
        res["time"] = round(time, 2)
    except:
        res["time"] = 0

    return res, finish


def main(args):
    cols = [
        "name",
        "model_source",
        "model_type",
        "parameters",
        "loss",
        "loss_avg_5",
        "token",
        "memory",
        "tgs",
        "batch_size",
        "context_len",
        "seq_len",
        "'eps'",
        "steps",
        "warmup_steps",
        "varlen",
        "time",
    ]
    print_array(cols)

    log_file = args.log_file
    log_dir = args.log_dir
    if log_dir == "":
        stat, finish = get_stat(log_file, cols)
        model_name = Path(log_file).stem[18:]
        stat["name"] = model_name
        if finish:
            print_dict(cols, stat)
    else:
        for log_file in os.listdir(log_dir):
            stat, finish = get_stat(os.path.join(log_dir, log_file), cols)
            model_name = Path(log_file).stem[18:]
            stat["name"] = model_name
            if finish:
                print_dict(cols, stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    args = parser.parse_args()
    main(args)
