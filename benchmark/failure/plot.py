import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    sys = "Scalai"
    width = 0.3

    failures = [4, 8, 12]

    times = np.array([23.5, 19.6, 19.6])
    tenplex_rerun = np.array([0, 0, 360])
    times_tenplex = times + tenplex_rerun
    baseline_rerun = np.array([360, 360, 360])
    times_baseline = times + baseline_rerun

    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    plt.rc("figure", figsize=[8, 4.5])
    fig, ax = plt.subplots()

    x = np.arange(len(failures))

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
        times_baseline,
        width=width,
        label="Baseline",
        hatch="--",
        fill=False,
        edgecolor="tab:orange",
        linewidth=linewidth,
    )

    fontsize = 26
    labelsize = 22
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(top=600)
    ax.tick_params(labelsize=labelsize)
    ax.set_xlabel("Number of GPU failures", fontsize=fontsize)
    ax.set_ylabel("Time in seconds", fontsize=fontsize)
    ax.legend(fontsize=labelsize)
    ax.set_xticks(x + 0.5 * width, failures)

    fig.tight_layout()
    plt.savefig("./failure.pdf")


if __name__ == "__main__":
    main()
