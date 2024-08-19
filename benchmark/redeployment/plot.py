import os

import matplotlib.pyplot as plt
import numpy as np


def parse_logs():
    # TODO
    pass


def main():
    sys = "Scalai"
    width = 0.3

    size = [
        "1.3B",
        "2.7B",
        "6.7B",
    ]
    times_tenplex = [4.8, 8.5, 15.3]

    times_central = [10, 16, 31.1]

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
    plt.savefig("./redeploy.pdf")


if __name__ == "__main__":
    main()
