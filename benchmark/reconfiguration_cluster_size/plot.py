import re

import matplotlib.pyplot as plt


def match_timestamp(pattern, lines):
    for i, line in enumerate(lines):
        mat = re.search(pattern, line)
        if mat:
            return float(mat.group(1)), i

    return -1.0, -1


def extract_times(paras, sizes):
    reconf_times = {}
    for para in paras:
        reconf_times[para] = []
        for size in sizes:
            log_file_name = f"reconfig_{size}_{para}.log"

            with open(log_file_name, "r", encoding="utf-8") as log_file:
                lines = log_file.readlines()

            pattern = r"Finished pretrain (\d+\.\d+)"
            finish_timestamp, finish_line = match_timestamp(pattern, lines)
            pattern = r"Start pretrain (\d+\.\d+)"
            start_timestamp, _ = match_timestamp(pattern, lines[finish_line+1:])

            latency = start_timestamp - finish_timestamp
            reconf_times[para].append(latency)

    return reconf_times


def main():
    parallelisms = ["pp", "mp", "dp"]
    sizes = [8, 16, 32]
    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    reconf_times = extract_times(parallelisms, sizes)

    for par in parallelisms:
        plt.rc("figure", figsize=[8, 4.5])
        fig, ax = plt.subplots()

        width = 0.3

        nd = ["4 to 8", "8 to 16", "16 to 32"]
        t = reconf_times[par]

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
