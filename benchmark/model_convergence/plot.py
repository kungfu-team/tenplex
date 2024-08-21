import os

import matplotlib.pyplot as plt
import pandas as pd


def subsample(arr, every):
    return arr[::every]


def schedule_to_points(schedule, batch_size=2048):
    x = []
    y = []
    size_before = 0

    for step, size in schedule.items():
        progress = step * batch_size
        if step == 0:
            x.append(progress)
            y.append(size)
        elif size == 0:
            x.append(progress - 1)
            y.append(size_before)
        else:
            x.append(progress - 1)
            y.append(size_before)
            x.append(progress)
            y.append(size)

        size_before = size

    return x, y


def main():
    folder = "data/go"
    #  dataset = "openwebtext"
    #  dataset_seed = 42
    #  seq_len = 1024
    subsample_every = 2

    schedule_no_scaling = pd.read_csv(f"./{folder}/no_scaling/a/loss.csv")
    schedule_up = pd.read_csv(f"./{folder}/pp_up/b/loss.csv")
    schedule_down = pd.read_csv(f"./{folder}/pp_down/b/loss.csv")

    linestyles = {
        "loosely dotted": (0, (1, 10)),
        "dotted": (0, (1, 1)),
        "densely dotted": (0, (1, 1)),
        "loosely dashed": (0, (5, 10)),
        "dashed": (0, (5, 5)),
        "densely dashed": (0, (5, 1)),
    }

    plt.rc("figure", figsize=[8, 4.5])
    fig, ax = plt.subplots()

    linewidth = 2
    ax.plot(
        subsample(schedule_no_scaling["Step"], subsample_every),
        subsample(schedule_no_scaling["Value"], subsample_every),
        label=("No resource change"),
        linestyle="solid",
        linewidth=linewidth,
        color="black",
    )
    ax.plot(
        subsample(schedule_down["Step"], subsample_every),
        subsample(schedule_down["Value"], subsample_every),
        label=("Resource decrease"),
        linestyle="dotted",
        linewidth=linewidth,
        color="tab:red",
    )
    ax.plot(
        subsample(schedule_up["Step"], subsample_every),
        subsample(schedule_up["Value"], subsample_every),
        label=("Resource increase"),
        linestyle="dashed",
        linewidth=linewidth,
        color="tab:blue",
    )

    plt.axvline(x=100, linewidth=linewidth, color="tab:orange")
    ax.set_xlim(0, 200)
    ax.set_ylim(top=10)

    fontsize = 26
    labelsize = 22
    ax.tick_params(labelsize=labelsize)
    ax.legend(fontsize=labelsize, loc="upper right")
    ax.set_xlabel("Step", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./te-pp.pdf")


if __name__ == "__main__":
    main()
