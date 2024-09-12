import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    sys = "Tenplex"
    # GPT-3 XL
    width = 0.2

    times_singularity = [97, 122]

    tenplex_up_start = 1691135057.0869706
    tenplex_up_complete = 1691135144.2678535
    tenplex_down_start = 1691134455.769136
    tenplex_down_complete = 1691134525.9654973
    times_tenplex = [
        tenplex_up_complete - tenplex_up_start,
        tenplex_down_complete - tenplex_down_start,
    ]

    deepspeed_up = 115
    deepspeed_down = 195
    times_deepspeed = [deepspeed_up, deepspeed_down]

    print(f"singularity {times_singularity}")
    print(f"deepspeed {times_deepspeed}")
    print(f"tenplex {times_tenplex}")

    xps = ["8 to 16", "16 to 8"]
    x = np.arange(len(xps))

    plt.rc("figure", figsize=[8, 4.5])
    fig, ax = plt.subplots()

    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 3
    ax.bar(
        x,
        times_singularity,
        width=width,
        label="Singularity",
        fill=False,
        hatch="//",
        edgecolor="tab:blue",
        linewidth=linewidth,
    )
    ax.bar(
        x + 1.1 * width,
        times_deepspeed,
        width=width,
        label="Deepspeed",
        fill=False,
        hatch="--",
        edgecolor="tab:orange",
        linewidth=linewidth,
    )
    ax.bar(
        x + 2.2 * width,
        times_tenplex,
        width=width,
        label=sys,
        fill=False,
        hatch="xx",
        edgecolor="tab:green",
        linewidth=linewidth,
    )

    fontsize = 26
    labelsize = 22
    ax.tick_params(labelsize=labelsize)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(top=225)
    ax.set_xlabel("Number of devices", fontsize=fontsize)
    ax.set_ylabel("Time (seconds)", fontsize=fontsize)
    ax.legend(loc="upper left", fontsize=labelsize)
    ax.set_xticks(x + 1.1 * width, xps)

    fig.tight_layout()
    plt.savefig("./scaling_latency.pdf")


if __name__ == "__main__":
    main()
