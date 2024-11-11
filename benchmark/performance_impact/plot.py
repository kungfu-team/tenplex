#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fmt_key(k):
    p, t, d = k
    # return f"T{t}\nP{p}\nD{d}"
    return f"P{p}\nT{t}\nD{d}"


def filter_none(data: dict):
    new_data = {}
    for k, v in data.items():
        if v and v > 0.0:
            new_data[k] = v
    return new_data


def to_dict(df):
    tps = df["tp"].to_numpy()
    dps = df["dp"].to_numpy()
    pps = df["pp"].to_numpy()
    throughputs = df["throughput"].to_numpy()

    dic = {}
    for tp, dp, pp, throughput in zip(tps, dps, pps, throughputs):
        dic[(tp, dp, pp)] = throughput

    return dic


def to_str_keys(data):
    keys = []
    for d in data:
        k = [str(e) for e in d[:3]]
        keys.append(fmt_key(k))
    return keys


def to_throughput(data, batch_size):
    for i, d in enumerate(data):
        data[i][3] = batch_size / d[3]
    return data


def plot_throughput():
    plt.rcParams["hatch.linewidth"] = 3
    width = 0.4  # the width of the bars
    plt.rc("figure", figsize=[10, 4.5])
    hatch = ["//", "--"]
    edgecolor = ["tab:blue", "tab:orange"]
    fontsize = 18
    labelsize = 16
    ylim = 45

    # P,T,D
    data_8 = [
        [1, 4, 2, 13.9],
        [1, 8, 1, 34, 7],
        [2, 4, 1, 14],
        [4, 2, 1, 5.8],
    ]
    data_16 = [
        [1, 4, 4, 9],
        [1, 8, 2, 21.7],
        [1, 16, 1, 40.5],
        [2, 8, 1, 23],
        [4, 2, 2, 3.6],
        [4, 4, 1, 9.1],
        [8, 2, 1, 3.9],
    ]
    batch_size = 128
    data_8 = to_throughput(data_8, batch_size)
    data_16 = to_throughput(data_16, batch_size)

    fig, ax = plt.subplots(1, 2)

    # 8 GPUs
    keys = to_str_keys(data_8)
    vals = [x[3] for x in data_8]
    x = np.arange(len(keys))
    ax[0].bar(
        x,
        vals,
        width,
        label="8 GPUs",
        hatch=hatch[0],
        fill=False,
        edgecolor=edgecolor[0],
    )

    ax[0].grid(axis="y")
    ax[0].set_axisbelow(True)
    ax[0].tick_params(labelsize=labelsize)
    # ax[0].set_ylabel("Throughput (samples/s)", fontsize=fontsize)
    ax[0].set_xticks(x, keys)
    ax[0].legend(loc="upper right", fontsize=labelsize)
    ax[0].set_ylim(0, ylim)

    keys = to_str_keys(data_16)
    vals = [x[3] for x in data_16]
    x = np.arange(len(keys))
    ax[1].bar(
        x,
        vals,
        width,
        label="16 GPUs",
        hatch=hatch[1],
        fill=False,
        edgecolor=edgecolor[1],
    )

    ax[1].grid(axis="y")
    ax[1].set_axisbelow(True)
    ax[1].tick_params(labelsize=labelsize)
    ax[1].set_xticks(x, keys)
    ax[1].legend(loc="upper right", fontsize=labelsize)
    ax[1].set_ylim(0, ylim)

    fig.supxlabel("Parallelization configuration", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("performance_impact.pdf")


if __name__ == "__main__":
    plot_throughput()
