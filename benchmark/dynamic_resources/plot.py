import argparse

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


def plot_throughput(data, ax, label, linesty, colour):
    linewidth = 1.5
    time = data["wall_time"]
    throughput = sp.signal.savgol_filter(
        data["throughput"], window_length=50, polyorder=5
    )
    # throughput = data["throughput"]
    ax.plot(
        time,
        throughput,
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


def cut_short(data, step):
    indices = data["step"] <= step
    data["wall_time"] = data["wall_time"][indices]
    data["step"] = data["step"][indices]
    data["loss"] = data["loss"][indices]
    return data


def calc_throughput(data, batch_size=128):
    throughput = [0]

    for i in range(1, len(data["step"])):
        current_step = data["step"][i]
        last_step = data["step"][i - 1]
        current_time = data["wall_time"][i]
        last_time = data["wall_time"][i - 1]
        steps = current_step - last_step
        time = (current_time - last_time) * 60  # to seconds
        samples = batch_size * steps
        thr = samples / time
        throughput.append(thr)

    data["throughput"] = np.array(throughput)
    return data


# linestyles = {
#     "loosely dotted": (0, (1, 10)),
#     "dotted": (0, (1, 1)),
#     "densely dotted": (0, (1, 1)),
#     "loosely dashed": (0, (5, 10)),
#     "dashed": (0, (5, 5)),
#     "densely dashed": (0, (5, 1)),
# }


def gen_throughput_fig(ax, tenplex: dict, tenplex_dp: dict, pytorch: dict, sys: str):
    plot_throughput(tenplex, ax, sys, "solid", "black")
    plot_throughput(tenplex_dp, ax, f"{sys} DP", "dashed", "tab:red")
    plot_throughput(
        pytorch,
        ax,
        "Torch Distributed Elastic",
        "dotted",
        "tab:blue",
    )
    ax.set_ylim(bottom=0)

    time_key = "wall_time"
    tenplex_final_time = tenplex[time_key][-1]
    tenplex_dp_final_time = tenplex_dp[time_key][-1]
    pytorch_final_time = pytorch[time_key][-1]
    right = max(tenplex_final_time, tenplex_dp_final_time, pytorch_final_time)

    ax.set_xlim(left=0, right=right)
    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
        fontsize=labelsize,
    )
    ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Througput (samples/s)", fontsize=fontsize)


def main():
    sys = "Scalai"
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-step", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    use_step = args.use_step

    # tenplex = pd.read_csv("./loss_tenplex.csv")
    tenplex = np.load("./data/tenplex_loss.npz")
    tenplex = dict(tenplex)
    tenplex_dp = np.load("./data/tenplex_dp_loss.npz")
    tenplex_dp = dict(tenplex_dp)
    pytorch = np.load("./data/pytorch_loss.npz")

    pytorch = dict(pytorch)

    time_key = "wall_time"
    step_key = "step"

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    # Scaling lines
    if not use_step:
        for sca in range(35, 800, 35):
            plt.axvline(sca, c="tab:grey", linewidth=0.75)

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
    pytorch = add_pause(pytorch)
    pytorch_final_step = pytorch[step_key][-1]
    pytorch_final_time = pytorch[time_key][-1]

    # Cut short
    min_final_step = min(tenplex_final_step, tenplex_dp_final_step, pytorch_final_step)
    tenplex = cut_short(tenplex, min_final_step)
    tenplex_dp = cut_short(tenplex_dp, min_final_step)
    pytorch = cut_short(pytorch, min_final_step)

    # Update final
    tenplex_final_time = tenplex[time_key][-1]
    tenplex_final_step = tenplex[step_key][-1]
    tenplex_dp_final_time = tenplex_dp[time_key][-1]
    tenplex_dp_final_step = tenplex_dp[step_key][-1]
    pytorch_final_step = pytorch[step_key][-1]
    pytorch_final_time = pytorch[time_key][-1]

    # Throughput
    tenplex = calc_throughput(tenplex)
    tenplex_dp = calc_throughput(tenplex_dp)
    pytorch = calc_throughput(pytorch)

    # plot
    if args.throughput:
        gen_throughput_fig(ax, tenplex, tenplex_dp, pytorch, sys)
        fig.tight_layout()
        plt.savefig("./dynamic_resources_throughput.pdf")
        return

    print("plot Tenplex")
    plot_loss(tenplex, ax, sys, "solid", "black", use_step=use_step)
    print("plot Tenplex DP only")
    plot_loss(tenplex_dp, ax, f"{sys} DP", "dashed", "tab:red", use_step=use_step)

    print("plot Pytorch")
    plot_loss(
        pytorch,
        ax,
        "Torch Distributed Elastic",
        "dotted",
        "tab:blue",
        use_step=use_step,
    )
    ax.set_ylim(bottom=0, top=8)

    # if not use_step:
    #     plt.axvline(tenplex_final_time, c="black")

    if use_step:
        right = max(tenplex_final_step, tenplex_dp_final_step, pytorch_final_step)
    else:
        right = max(tenplex_final_time, tenplex_dp_final_time, pytorch_final_time)
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
