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
    use_step = True

    tenplex = np.load("./tenplex_loss.npz")
    tenplex = dict(tenplex)
    tde = np.load("./tde_loss.npz")
    tde = dict(tde)

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    tenplex = zero_time(tenplex)
    tde = zero_time(tde)

    plot_loss(tenplex, ax, "Tenplex", "solid", "black", use_step=use_step)
    plot_loss(tde, ax, "NO Tenplex", "dashed", "tab:red", use_step=use_step)

    fontsize = 18
    labelsize = 16
    # ax.set_ylim(bottom=0, top=8)
    # ax.set_ylim(bottom=0)
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
