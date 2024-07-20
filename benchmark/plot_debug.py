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
    loss = data["loss"]
    loss = sp.signal.savgol_filter(loss, window_length=50, polyorder=5)
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

    tenplex = np.load("./tenplex_loss.npz")  # DP=2, TP=2
    tenplex = dict(tenplex)
    tde = np.load("./tde_loss.npz")
    tde = dict(tde)
    tenplex_debug = np.load("./tenplex_debug_loss.npz")  # DP=2, PP=2
    tenplex_debug = dict(tenplex_debug)
    tenplex_debug_2 = np.load("./tenplex_debug_2_loss.npz")  # DP=2, PP=2
    tenplex_debug_2 = dict(tenplex_debug_2)
    tenplex_debug_tp2 = np.load("./tenplex_debug_tp2_loss.npz")  # DP=2, TP=2
    tenplex_debug_tp2 = dict(tenplex_debug_tp2)

    plt.rc("figure", figsize=[8, 3.5])
    fig, ax = plt.subplots()

    tenplex = zero_time(tenplex)
    tde = zero_time(tde)
    tenplex_debug = zero_time(tenplex_debug)

    plot_loss(tenplex, ax, "Tenplex", "solid", "black", use_step=use_step)
    plot_loss(tde, ax, "TDE", "solid", "tab:red", use_step=use_step)
    plot_loss(
        tenplex_debug,
        ax,
        "Tenplex+MLM-split",
        "solid",
        "tab:blue",
        use_step=use_step,
    )
    plot_loss(
        tenplex_debug_2,
        ax,
        "Tenplex+MLM-split (2)",
        "solid",
        "tab:green",
        use_step=use_step,
    )
    plot_loss(tenplex_debug_tp2, ax, "TP2", "solid", "tab:orange", use_step=use_step)

    fontsize = 18
    labelsize = 16
    legendsize = 10
    # ax.set_ylim(bottom=0, top=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 500)
    ax.tick_params(labelsize=labelsize)
    ax.legend(loc="lower left", fontsize=legendsize)
    if use_step:
        ax.set_xlabel("Step", fontsize=fontsize)
    else:
        ax.set_xlabel("Time (minutes)", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig("./debug.pdf")


if __name__ == "__main__":
    main()
