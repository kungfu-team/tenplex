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


def plot_loss(data, ax, label, linesty, colour):
    linewidth = 1.5
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


def plot_loss_pause(data, ax, label, linesty, colour):
    linewidth = 1.5
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
    tenplex = pd.read_csv("./loss_tenplex.csv")
    dp_only = pd.read_csv("./loss_dp_only.csv")
    pytorch = np.load("./loss.npz")
    pytorch = dict(pytorch)

    time_key = "wall_time"
    loss_key = "loss"
    step_key = "step"
    tenplex = {time_key: tenplex["Wall time"].to_numpy(),
               step_key: tenplex["Step"].to_numpy(),
               loss_key: tenplex["Value"].to_numpy()
               }
    dp_only = {time_key: dp_only["Wall time"].to_numpy(),
               step_key: dp_only["Step"].to_numpy(),
               loss_key: dp_only["Value"].to_numpy()
               }

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    # Scaling lines
    # end = 538
    for sca in range(35, 800, 35):
        plt.axvline(sca, c="tab:orange")

    # Tenplex
    tenplex = zero_time(tenplex)
    tenplex_final_time = tenplex[time_key][-1]
    tenplex_final_step = tenplex[step_key][-1]

    # Tenplex DP only
    dp_only = zero_time(dp_only)
    dp_only = add_pause(dp_only)
    dp_only_final_time = dp_only[time_key][-1]
    dp_only_final_step = dp_only[step_key][-1]

    # Pytorch
    pytorch = zero_time(pytorch)
    pytorch = add_pause(pytorch, interval=35)
    pytorch_final_step = pytorch[step_key][-1]
    pytorch_final_time = pytorch[time_key][-1]

    # plot
    print("plot Tenplex")
    plot_loss(tenplex, ax, "Tenplex", "solid", "black")
    print("plot Tenplex DP only")
    plot_loss_pause(dp_only, ax, "Tenplex (DP)", "dashed", "tab:red")
    print("plot Pytorch")
    plot_loss(pytorch, ax, "PyTorch", "dotted", "tab:olive")
    plt.axvline(tenplex_final_time, c="black")

    ax.set_ylim(top=6, bottom=2)
    ax.set_xlim(left=0, right=pytorch_final_time)
    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="upper right", fontsize=labelsize)
    ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./dynamic_resources.pdf")

    print(f"DP only final step {dp_only_final_step}")
    print(f"Tenplex final step {tenplex_final_step}")
    print(f"Pytorch final step {pytorch_final_step}")
    print(f"DP only final time {dp_only_final_time}")
    print(f"Tenplex final time {tenplex_final_time}")
    print(f"Pytorch final time {pytorch_final_time}")


if __name__ == "__main__":
    main()
