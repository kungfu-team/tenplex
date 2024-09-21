import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_log(filename):
    lines = [l.strip() for l in open(filename) if "Img/sec" in l]
    s = lines[0]
    parts = s.split(" ")
    return float(parts[2])


def parse_logs(output="throughput.csv"):
    a = parse_log("1.log")
    b = parse_log("2.log")
    c = parse_log("3.log")
    rows = [
        ("name", "throughput"),
        ("Horovod", a),
        ("Elastic", b),
        ("Tenplex", c),
    ]
    with open(output, "w") as f:
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")


def plot():
    sys = "Tenplex"
    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2
    width = 0.3
    fontsize = 20
    labelsize = 16

    data = pd.read_csv("./throughput.csv")
    data = data.replace(to_replace="Tenplex", value=sys)

    plt.rc("figure", figsize=[8, 4])
    fig, ax = plt.subplots()

    ax.bar(
        data["name"],
        data["throughput"],
        width=width,
        hatch="//",
        fill=False,
        edgecolor="tab:blue",
        linewidth=linewidth,
    )

    for i, value in enumerate(data["throughput"]):
        plt.text(
            i,
            value + 5,
            str(value),
            ha="center",
            c="tab:blue",
            fontsize=fontsize,
        )

    ax.set_xlabel("System", fontsize=fontsize)
    ax.set_ylabel("Throughput (samples/s)", fontsize=fontsize)
    ax.tick_params(labelsize=labelsize)
    ax.set_ylim(0, 500)

    fig.tight_layout()
    plt.savefig("./throughput.pdf")


if __name__ == "__main__":
    parse_logs()
    plot()
