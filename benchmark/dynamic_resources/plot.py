import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp


def subsample(arr, every):
    return arr[::every]


def get_time(dataf, step):
    steps = dataf["Step"]
    step = steps[steps == step].index[0]
    time = dataf["Wall time"][step]
    return time


def zero_time(data):
    time = data["Wall time"]
    time_zero = time[0]
    time = time - time_zero
    time = time / 60.0  # to minutes
    data["Wall time"] = time
    return data


def plot_loss(data, ax, subsample_every, label, linesty, colour):
    linewidth = 1.5
    time = data["Wall time"]
    time = subsample(time, subsample_every)
    loss = sp.signal.savgol_filter(data["Value"], window_length=50, polyorder=5)
    loss = subsample(loss, subsample_every)
    ax.plot(
        time,
        loss,
        label=label,
        linewidth=linewidth,
        linestyle=linesty,
        color=colour,
    )


def plot_loss_pause(data, ax, subsample_every, label, linesty, colour):
    linewidth = 1.5
    time = data["Wall time"]
    time = subsample(time, subsample_every)
    loss = sp.signal.savgol_filter(data["Value"], window_length=50, polyorder=5)
    loss = subsample(loss, subsample_every)

    interval = 35
    duration = int(time[len(time) - 1])
    label_i = label
    for i, sta in enumerate(range(0, duration, 3 * interval)):
        l = find_idx(time, sta)
        u = find_idx(time, sta + 2 * interval)
        x = time[l:u]
        y = loss[l:u]

        if i > 0:
            label_i = None
        ax.plot(
            x,
            y,
            label=label_i,
            linewidth=linewidth,
            linestyle=linesty,
            color=colour,
        )


def find_idx(time, point):
    for i in range(len(time)):
        if time[i] >= point:
            return i
    print(f"cannot find index for {point}")
    return -1


def add_pause(data, interval=35):
    time = data["Wall time"]
    duration = int(time[len(time) - 1])
    time_pause = time.copy()
    for sta in range(2 * interval, duration, 2 * interval):
        idx = find_idx(time, sta)
        time_pause[idx:] = time_pause[idx:] + interval
    data["Wall time"] = time_pause
    return data


def main():
    folder = "data"
    subsample_every = 1
    linestyles = {
        "loosely dotted": (0, (1, 10)),
        "dotted": (0, (1, 1)),
        "densely dotted": (0, (1, 1)),
        "loosely dashed": (0, (5, 10)),
        "dashed": (0, (5, 5)),
        "densely dashed": (0, (5, 1)),
    }

    tenplex = pd.read_csv(f"./{folder}/tenplex/start-16/loss.csv")
    dp_only = pd.read_csv(f"./{folder}/dp-only/pp4-mp2/loss.csv")

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    # Scaling lines
    end = 538
    for sca in range(35, end, 35):
        plt.axvline(sca, c="tab:orange")

    # Tenplex
    tenplex = zero_time(tenplex)
    tenplex_time = tenplex["Wall time"]
    tenplex_finish = tenplex_time[len(tenplex_time) - 1]
    tenplex_step = tenplex["Step"]
    tenplex_final_step = tenplex_step[len(tenplex_step) - 1]

    # Scale DP (Deepspeed)
    dp_only = zero_time(dp_only)
    dp_only = add_pause(dp_only)
    dp_only = dp_only.loc[dp_only["Wall time"] <= tenplex_finish]
    dp_only_step = dp_only["Step"]
    dp_only_final_step = dp_only_step[len(dp_only_step) - 1]

    # Shorten Tenplex
    tenplex = tenplex.loc[tenplex["Step"] <= dp_only_final_step]
    tenplex_new_time = tenplex["Wall time"]
    tenplex_new_final_time = tenplex_new_time[len(tenplex_new_time) - 1]
    tenplex_new_step = tenplex["Wall time"]
    tenplex_new_final_step = tenplex_new_step[len(tenplex_new_step) - 1]
    plt.axvline(tenplex_new_final_step, c="black")

    # plot
    plot_loss(tenplex, ax, subsample_every, "Tenplex", "solid", "black")
    #  plot_loss(dp_only, ax, subsample_every, "DeepSpeed", "dashed", "tab:red")
    plot_loss_pause(dp_only, ax, subsample_every, "Tenplex (DP)", "dashed", "tab:red")

    #  ax.set_ylim(top=6.25, bottom=0)
    ax.set_ylim(top=5, bottom=2)
    ax.set_xlim(left=0, right=end)
    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="lower left", fontsize=labelsize)
    ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./dynamic_resources.pdf")

    print(f"DP only final step {dp_only_final_step}")
    print(f"Tenplex final step {tenplex_final_step}")
    print(f"DP only final time {end}")
    print(f"Tenplex final time {tenplex_new_final_time}")


if __name__ == "__main__":
    main()
