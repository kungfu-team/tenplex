import argparse
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import numpy as np


def zero_time(data):
    time = data["wall_time"]
    time_zero = time[0]
    time = time - time_zero
    time = time / 60.0  # to minutes
    data["wall_time"] = time
    return data


def plot_loss(data, ax, label, linesty, colour, use_step=False):
    linewidth = 1.5
    if use_step:
        time = data["step"]
    else:
        time = data["wall_time"]
    loss = sp.signal.savgol_filter(data["loss"], window_length=50, polyorder=5)
    ax.plot(
        time,
        loss,
        label=label,
        linewidth=linewidth,
        linestyle=linesty,
        color=colour,
    )


def plot_loss_pause(data, ax, label, linesty, colour, use_step=False):
    linewidth = 1.5
    if use_step:
        time = data["step"]
    else:
        time = data["wall_time"]
    loss = sp.signal.savgol_filter(data["loss"], window_length=50, polyorder=5)

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
    time = data["wall_time"]
    duration = int(time[len(time) - 1])
    time_pause = time.copy()
    for sta in range(2 * interval, duration, 2 * interval):
        idx = find_idx(time, sta)
        time_pause[idx:] = time_pause[idx:] + interval
    data["wall_time"] = time_pause
    return data


# linestyles = {
#     "loosely dotted": (0, (1, 10)),
#     "dotted": (0, (1, 1)),
#     "densely dotted": (0, (1, 1)),
#     "loosely dashed": (0, (5, 10)),
#     "dashed": (0, (5, 5)),
#     "densely dashed": (0, (5, 1)),
# }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-step", action="store_true")
    args = parser.parse_args()
    use_step = args.use_step

    # tenplex = pd.read_csv("./loss_tenplex.csv")
    tenplex = np.load("./tenplex_loss.npz")
    tenplex = dict(tenplex)
    tenplex_dp = pd.read_csv("./loss_dp_only.csv")
    pytorch = np.load("./pytorch_loss.npz")
    pytorch = dict(pytorch)

    time_key = "wall_time"
    loss_key = "loss"
    step_key = "step"
    tenplex_dp = {time_key: tenplex_dp["Wall time"].to_numpy(),
           step_key: tenplex_dp["Step"].to_numpy(),
           loss_key: tenplex_dp["Value"].to_numpy()
       }

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    # Scaling lines
    if not use_step:
        for sca in range(35, 800, 35):
            plt.axvline(sca, c="tab:orange")

    # Tenplex
    tenplex = zero_time(tenplex)
    tenplex_final_time = tenplex[time_key][-1]
    tenplex_final_step = tenplex[step_key][-1]

    # Tenplex DP only
    tenplex_dp = zero_time(tenplex_dp)
    tenplex_dp = add_pause(tenplex_dp)
    tenplex_dp_final_time = tenplex_dp[time_key][-1]
    tenplex_dp_final_step = tenplex_dp[step_key][-1]

    # Pytorch
    pytorch = zero_time(pytorch)
    pytorch = add_pause(pytorch, interval=35)
    pytorch_final_step = pytorch[step_key][-1]
    pytorch_final_time = pytorch[time_key][-1]

    # plot
    print("plot Tenplex")
    plot_loss(tenplex, ax, "Tenplex", "solid", "black", use_step=use_step)
    print("plot Tenplex DP only")
    plot_loss_pause(tenplex_dp, ax, "Tenplex (DP)", "dashed", "tab:red", use_step=use_step)
    print("plot Pytorch")
    plot_loss(pytorch, ax, "PyTorch", "dotted", "tab:olive", use_step=use_step)
    if not use_step:
        plt.axvline(tenplex_final_time, c="black")

    ax.set_ylim(top=6, bottom=2)
    if use_step:
        right = pytorch_final_step
    else:
        right = pytorch_final_time
    ax.set_xlim(left=0, right=right)
    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="upper right", fontsize=labelsize)
    if use_step:
        ax.set_xlabel("Step", fontsize=fontsize)
    else:
        ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./dynamic_resources.pdf")

    print(f"DP only final step {tenplex_dp_final_step}")
    print(f"Tenplex final step {tenplex_final_step}")
    print(f"Pytorch final step {pytorch_final_step}")
    print(f"DP only final time {tenplex_dp_final_time}")
    print(f"Tenplex final time {tenplex_final_time}")
    print(f"Pytorch final time {pytorch_final_time}")


if __name__ == "__main__":
    main()
