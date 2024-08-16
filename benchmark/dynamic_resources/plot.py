import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def zero_time(data):
    time = data["wall_time"]
    time_zero = time[0]
    time = time - time_zero
    data["wall_time"] = time / 60  # to minutes
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


def cut_short(data, step):
    indices = data["step"] <= step
    data["wall_time"] = data["wall_time"][indices]
    data["step"] = data["step"][indices]
    data["loss"] = data["loss"][indices]
    return data


def main():
    sys = "Tenplex"
    time_key = "wall_time"
    step_key = "step"

    tenplex = np.load("dyn-res-tenplex_loss.npz")
    tenplex = dict(tenplex)
    tenplex_dp = np.load("dyn-res-tenplex-dp_loss.npz")
    tenplex_dp = dict(tenplex_dp)
    tde = np.load("dyn-res-tde_loss.npz")
    tde = dict(tde)

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    # Scaling lines
    for sca in range(35, 800, 35):
        plt.axvline(sca, c="tab:grey", linewidth=0.75)

    # Tenplex
    tenplex = zero_time(tenplex)
    tenplex_final_step = tenplex[step_key][-1]

    # Tenplex-DP
    tenplex_dp = zero_time(tenplex_dp)
    tenplex_dp = add_pause(tenplex_dp)
    tenplex_dp_final_step = tenplex_dp[step_key][-1]

    # TDE
    tde = zero_time(tde)
    tde = add_pause(tde)
    tde_final_step = tde[step_key][-1]

    # Cut short
    min_final_step = min(tenplex_final_step, tenplex_dp_final_step, tde_final_step)
    tenplex = cut_short(tenplex, min_final_step)
    tenplex_dp = cut_short(tenplex_dp, min_final_step)
    tde = cut_short(tde, min_final_step)

    # Final step and time
    tenplex_final_time = tenplex[time_key][-1]
    tenplex_final_step = tenplex[step_key][-1]
    tenplex_dp_final_time = tenplex_dp[time_key][-1]
    tenplex_dp_final_step = tenplex_dp[step_key][-1]
    tde_final_step = tde[step_key][-1]
    tde_final_time = tde[time_key][-1]

    # plot
    print(f"plot {sys}")
    plot_loss(tenplex, ax, sys, "solid", "black")
    print(f"plot {sys}-DP")
    plot_loss(tenplex_dp, ax, f"{sys}-DP", "dashed", "tab:red")

    print("plot TDE")
    plot_loss(tde, ax, "Torch Distributed Elastic", "dotted", "tab:blue")

    ax.set_ylim(bottom=0, top=8)
    right = max(tenplex_final_time, tenplex_dp_final_time, tde_final_time)
    ax.set_xlim(left=0, right=right)
    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="upper right", fontsize=labelsize)
    ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./dynamic_resources.pdf")

    print(f"{sys} final step {tenplex_final_step}")
    print(f"{sys}-DP final step {tenplex_dp_final_step}")
    print(f"TDE final step {tde_final_step}")
    print(f"{sys} final time {tenplex_final_time}")
    print(f"{sys}-DP final time {tenplex_dp_final_time}")
    print(f"TDE final time {tde_final_time}")


if __name__ == "__main__":
    main()
