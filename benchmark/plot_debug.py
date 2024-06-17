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


def main():
    use_step = False

    tenplex = np.load("./tenplex_tenplex_loss.npz")
    tenplex = dict(tenplex)
    no_tenplex = np.load("./no_tenplex_loss.npz")
    no_tenplex = dict(no_tenplex)

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    tenplex = zero_time(tenplex)
    no_tenplex = zero_time(no_tenplex)

    plot_loss(tenplex, ax, "Tenplex", "solid", "black", use_step=use_step)
    plot_loss(no_tenplex, ax, "NO Tenplex", "dashed", "tab:red", use_step=use_step)

    fontsize = 18
    labelsize = 16
    ax.set_ylim(bottom=0, top=8)
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="upper right", fontsize=labelsize)
    if use_step:
        ax.set_xlabel("Step", fontsize=fontsize)
    else:
        ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./debug.pdf")


if __name__ == "__main__":
    main()
