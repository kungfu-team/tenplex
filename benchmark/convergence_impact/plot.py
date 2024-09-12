import re

import matplotlib.pyplot as plt
import scipy as sp


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


def parse_log(fname):
    with open(fname, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    steps = []
    losses = []
    pattern = r"Step: (\d+).*Loss: (\d*\.\d*)"
    for line in lines:
        mat = re.match(pattern, line)
        if mat:
            step = int(mat.group(1))
            loss = float(mat.group(2))
            steps.append(step)
            losses.append(loss)

    return steps, losses


def plot_inconsistent(static_fname, dynamic_fname, out_name):
    mnist_step, mnist_loss = parse_log(static_fname)
    incon_step, incon_loss = parse_log(dynamic_fname)

    plt.rc("figure", figsize=[4, 2.25])
    fig, ax = plt.subplots()

    linewidth = 2
    ax.plot(
        mnist_step,
        sp.signal.savgol_filter(mnist_loss, window_length=50, polyorder=5),
        label=("Static GPUs"),
        linestyle="solid",
        linewidth=linewidth,
        color="black",
    )
    ax.plot(
        incon_step,
        sp.signal.savgol_filter(incon_loss, window_length=50, polyorder=5),
        label=("Dynamic GPUs"),
        linestyle="dashed",
        linewidth=linewidth,
        color="tab:red",
    )
    plt.axvline(150, linewidth=linewidth, color="tab:orange")

    ax.set_ylim(bottom=0, top=0.4)
    fontsize = 16
    labelsize = 14
    legendsize = 12
    ax.tick_params(labelsize=labelsize)
    ax.legend(fontsize=legendsize)
    ax.set_xlabel("Step", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig(f"{out_name}.pdf")


def main():
    mnist_fname = "mnist.log"
    inconsistent_dataset_fname = "inconsistent_dataset.log"
    inconsistent_batch_size_fname = "inconsistent_batch_size.log"

    plot_inconsistent(mnist_fname, inconsistent_dataset_fname, "inconsistent_dataset")
    plot_inconsistent(
        mnist_fname, inconsistent_batch_size_fname, "inconsistent_batch_size"
    )


if __name__ == "__main__":
    main()
