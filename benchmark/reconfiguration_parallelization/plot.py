import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    sys = "Tenplex"
    width = 0.3

    size = [
        "1.3B",
        "2.7B",
        "6.7B",
    ]
    times_tenplex = {}
    times_tenplex["dp"] = [8.5, 11.5, 24.2]
    times_tenplex["mp"] = [7.4, 10.5, 20.2]
    times_tenplex["pp"] = [4.6, 5.9, 10.8]

    times_central = {}
    times_central["dp"] = [27.2, 39, 96.1]
    times_central["mp"] = [20.4, 28.5, 74]
    times_central["pp"] = [4.6, 18.2, 38]

    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    for xp in times_tenplex.keys():
        plt.rc("figure", figsize=[8, 4.5])
        fig, ax = plt.subplots()

        x = np.arange(len(size))

        ax.bar(
            x,
            times_tenplex[xp],
            width=width,
            label=sys,
            hatch="//",
            fill=False,
            edgecolor="tab:blue",
            linewidth=linewidth,
        )
        ax.bar(
            x + 1.1 * width,
            times_central[xp],
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
        plt.savefig(f"./time-size-mdp-{xp}.pdf")


if __name__ == "__main__":
    main()
