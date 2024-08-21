import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from tensorboard.backend.event_processing import event_accumulator


def load_metrics(tb_path):
    metrics = []
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    data_points = ea.Scalars("lm loss")
    for event in data_points:
        metrics.append([event.wall_time, event.step, event.value])
    return metrics


def load_tenplex(job_id: str) -> dict:
    num_containers = 4
    metrics = []
    for i in range(num_containers):
        tb_path = f"training/{job_id}/{i}/ckpt/tensorboard"
        if os.path.exists(tb_path):
            sub_metrics = load_metrics(tb_path)
            metrics.extend(sub_metrics)
    metrics.sort(key=lambda x: x[0])

    wall_times = [x[0] for x in metrics]
    steps = [x[1] for x in metrics]
    loss = [x[2] for x in metrics]
    return {"wall_time": wall_times, "step": steps, "loss": loss}

def subsample(arr, every):
    return arr[::every]


def plot_convergence(static, up, down, para: str):
    subsample_every = 1

    plt.rc("figure", figsize=[8, 4.5])
    fig, ax = plt.subplots()

    linewidth = 2
    loss = sp.signal.savgol_filter(static["loss"], window_length=16, polyorder=2)
    ax.plot(
        subsample(static["step"], subsample_every),
        # subsample(static["loss"], subsample_every),
        subsample(loss, subsample_every),
        label=("No resource change"),
        linestyle="solid",
        linewidth=linewidth,
        color="black",
    )
    loss = sp.signal.savgol_filter(down["loss"], window_length=16, polyorder=2)
    ax.plot(
        subsample(down["step"], subsample_every),
        subsample(loss, subsample_every),
        label=("Resource decrease"),
        linestyle="dotted",
        linewidth=linewidth,
        color="tab:red",
    )
    loss = sp.signal.savgol_filter(up["loss"], window_length=16, polyorder=2)
    ax.plot(
        subsample(up["step"], subsample_every),
        subsample(loss, subsample_every),
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
    plt.savefig(f"convergence_{para}.pdf")

def main():
    paras = ["dp", "tp", "pp"]

    static = load_tenplex("static")

    for para in paras:
        up = load_tenplex(f"{para}-up")
        down = load_tenplex(f"{para}-down")
        plot_convergence(static, up, down, para)


if __name__ == "__main__":
    main()
