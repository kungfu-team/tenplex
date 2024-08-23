import os

import matplotlib.pyplot as plt


def main():
    parallelisms = ["pp", "mp", "dp"]
    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    for par in parallelisms:
        plt.rc("figure", figsize=[8, 4.5])
        fig, ax = plt.subplots()

        width = 0.3

        nd = ["4 to 8", "8 to 16", "16 to 32"]
        print('TODO: extract data from log')
        t_dp = [19.4, 25.7, 31.2]
        t_pp = [11.3, 8.3, 6.6]
        t_mp = [18.4, 10.7, 7.1]
        if par == "dp":
            t = t_dp
        elif par == "pp":
            t = t_pp
        elif par == "mp":
            t = t_mp
        else:
            return
        ax.bar(
            nd,
            t,
            width=width,
            hatch="//",
            fill=False,
            edgecolor="tab:blue",
            linewidth=linewidth,
        )

        fontsize = 26
        labelsize = 22
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlabel("Number of devices", fontsize=fontsize)
        ax.set_ylabel("Time in seconds", fontsize=fontsize)
        ax.set_ylim(top=32)

        fig.tight_layout()
        plt.savefig(f"./time-nd-{par}.pdf")


if __name__ == "__main__":
    main()
