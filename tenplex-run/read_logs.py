import argparse
import os
import re

import numpy as np
import pandas as pd


def analyse_log(lines):
    times = []
    for lin in lines:
        mat = re.search(r"CUDA out of memory", lin)
        if mat:
            return "OOM"

        mat = re.search(r"elapsed time per iteration \(ms\): (\d+\.\d+)", lin)
        if mat:
            times.append(float(mat.group(1)))

    if times:
        tim = np.array(times)
        mean = np.mean(tim[1:] / 1000)
        return f"{mean:.1f}"

    return "error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname")
    args = parser.parse_args()
    models = []
    model_sizes = []
    datasets = []
    cluster_sizes = []
    pp_sizes = []
    mp_sizes = []
    dp_sizes = []
    batch_sizes = []
    micro_batch_sizes = []
    step_times = []
    for ent in os.scandir(args.dirname):
        if ent.is_dir() and ent.name.startswith("logs-"):
            mat = re.match(
                r"logs-(.+)-(.+)-(.+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", ent.name)
            models.append(mat.group(1))
            model_sizes.append(mat.group(2))
            datasets.append(mat.group(3))
            cluster_size = int(mat.group(4))
            cluster_sizes.append(cluster_size)
            pp_size = int(mat.group(5))
            pp_sizes.append(pp_size)
            mp_size = int(mat.group(6))
            mp_sizes.append(mp_size)
            dp_sizes.append(cluster_size // (pp_size * mp_size))
            batch_sizes.append(int(mat.group(7)))
            micro_batch_sizes.append(int(mat.group(8)))

            all_error = True
            for inner_ent in os.scandir(f"./{ent.path}"):
                if inner_ent.is_file():
                    with open(inner_ent.path, "r") as log_file:
                        log_lines = log_file.readlines()
                    step_time = analyse_log(log_lines)
                    if step_time != "error":
                        step_times.append(step_time)
                        all_error = False
                        break
            if all_error:
                step_times.append("error")

    dataf = pd.DataFrame({
        "model": models,
        "model size": model_sizes,
        "dataset": datasets,
        "cluster size": cluster_sizes,
        "PP size": pp_sizes,
        "MP size": mp_sizes,
        "DP size": dp_sizes,
        "batch size": batch_sizes,
        "micro batch size": micro_batch_sizes,
        "step time": step_times
    })
    dataf.to_csv("./experiments.csv", index=False)


if __name__ == "__main__":
    main()
