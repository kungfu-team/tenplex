#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fmt_key(k):
    m, d, p = k
    return "%d\n%d\n%d" % (m, p, d)


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


def plot_throughput():
    bert_df = pd.read_csv("bert.csv", index_col=False)
    bert = to_dict(bert_df)
    gpt_df = pd.read_csv("gpt.csv")
    gpt = to_dict(gpt_df)
    models = ["BERT-large", "GPT-3 2.7B"]

    plt.rcParams["hatch.linewidth"] = 3
    width = 0.4  # the width of the bars
    plt.rc("figure", figsize=[10, 4.5])
    hatch = ["//", "--"]
    edgecolor = ["tab:blue", "tab:orange"]
    fontsize = 18
    labelsize = 16

    fig, ax = plt.subplots(1, 2)

    bert = filter_none(bert)
    keys = bert.keys()
    vals = bert.values()
    x = np.arange(len(keys))
    rects = ax[0].bar(
        x,
        vals,
        width,
        label=models[0],
        hatch=hatch[0],
        fill=False,
        edgecolor=edgecolor[0],
    )

    key_labels = [fmt_key(k) for k in keys]
    ax[0].grid(axis="y")
    ax[0].set_axisbelow(True)
    ax[0].tick_params(labelsize=labelsize)
    ax[0].set_ylabel("Throughput (samples/s)", fontsize=fontsize)
    ax[0].set_xticks(x, key_labels)
    ax[0].legend(loc="upper right", fontsize=labelsize)
    # ax[0].set_ylim(0, 190)

    gpt = filter_none(gpt)
    keys = gpt.keys()
    vals = gpt.values()
    x = np.arange(len(keys))
    rects = ax[1].bar(
        x,
        vals,
        width,
        label=models[1],
        hatch=hatch[1],
        fill=False,
        edgecolor=edgecolor[1],
    )
    # ax.bar_label(rects, padding=3)

    key_labels = [fmt_key(k) for k in keys]
    ax[1].grid(axis="y")
    ax[1].set_axisbelow(True)
    ax[1].tick_params(labelsize=labelsize)
    ax[1].set_xticks(x, key_labels)
    ax[1].legend(loc="upper right", fontsize=labelsize)
    ax[1].set_ylim(0, 45)

    fig.supxlabel("Parallelization configuration", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("performance_impact.pdf")


if __name__ == "__main__":
    plot_throughput()
