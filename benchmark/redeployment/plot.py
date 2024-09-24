import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def extract_time(path: str, pattern: str) -> float:
    with open(path, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    for line in lines:
        mat = re.search(pattern, line)
        if mat:
            return float(mat.group(1))

    print(f"extracting time failed for {path}")
    return 0.0


def reconfig_time(log_dir: str) -> float:
    file_name = "tenplex-state-transformer-0-0.err.log"
    pattern = r"State transformation took (\d+\.\d+)s"

    return extract_time(os.path.join(log_dir, file_name), pattern)


def parse_logs():
    sizes = ["xl", "2.7B", "6.7B"]
    log_dirs = glob.glob("logs-*")

    times = []
    times_central = []
    for size in sizes:
        size_log_dirs = list(filter(lambda x: size in x, log_dirs))
        log_dir = list(filter(lambda x: "central" not in x, size_log_dirs))[0]
        log_dir_central = list(filter(lambda x: "central" in x, size_log_dirs))[0]

        recon_time = reconfig_time(log_dir)
        recon_time_central = reconfig_time(log_dir_central)
        times.append(recon_time)
        times_central.append(recon_time_central)

    return times, times_central


def plot_redeploy(times_tenplex: list, times_central: list):
    sys = "Tenplex"
    width = 0.3

    size = [
        "1.3B",
        "2.7B",
        "6.7B",
    ]

    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    plt.rc("figure", figsize=[8, 4.5])
    fig, ax = plt.subplots()

    x = np.arange(len(size))

    ax.bar(
        x,
        times_tenplex,
        width=width,
        label=sys,
        hatch="//",
        fill=False,
        edgecolor="tab:blue",
        linewidth=linewidth,
    )
    ax.bar(
        x + 1.1 * width,
        times_central,
        width=width,
        label=f"{sys} (central)",
        hatch="--",
        fill=False,
        edgecolor="tab:orange",
        linewidth=linewidth,
    )

    fontsize = 26
    labelsize = 22
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=labelsize)
    ax.set_xlabel("Model size", fontsize=fontsize)
    ax.set_ylabel("Time in seconds", fontsize=fontsize)
    ax.legend(loc="upper left", fontsize=labelsize)
    ax.set_xticks(x + 0.5 * width, size)

    fig.tight_layout()
    plt.savefig("./redeployment.pdf")


def main():
    times, times_central = parse_logs()
    plot_redeploy(times, times_central)


if __name__ == "__main__":
    main()
