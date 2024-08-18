#!/usr/bin/env python3
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grep(pattern, filename):
    for line in open(filename):
        if pattern in line:
            yield line.strip()


def parse_line(line):
    result = dict()
    for part in line.split("|"):
        kv = part.split(":")
        if len(kv) == 2:
            k, v = kv[0].strip(), kv[1].strip()
            result[k] = v
    return result


def parse_mdp(line):
    p = re.search("mp(\d+)-dp(\d+)-pp(\d+)", line)
    mdp = [p.group(1), p.group(2), p.group(3)]
    m, d, p = [int(x) for x in mdp]
    return m, d, p


def have_failed(filename):
    # print(filename)
    failed = False
    errs = [
        "RuntimeError",
        "OutOfMemoryError",
        "IndexError",
    ]
    for line in open(filename):
        for e in errs:
            if e in line:
                print(line.strip())
                failed = True
                # break
    return failed


pattern = "elapsed time per iteration"
key1 = "elapsed time per iteration (ms)"
key2 = "global batch size"


def parse_job_log(name):
    mdp = parse_mdp(name)
    files = glob.glob(name + "/*.log")
    failed = False
    for f in files:
        failed |= have_failed(f)
    if failed:
        print("%s failed" % (name))
        return (mdp, 0)

    lines = sum([list(grep(pattern, f)) for f in files], [])

    values = []
    for l in lines:
        r = parse_line(l)
        v1 = float(r[key1])
        v2 = int(r[key2])
        v = v2 / v1 * 1000
        values.append(v)

    value = np.mean(values)
    print("%s succ: %.3f" % (name, value))

    return (mdp, value)


def select_logs(log_dirs, name, size):
    for f in log_dirs:
        m, d, p = parse_mdp(f)
        if m * d * p == size and name in f:
            yield f


def parse_logs(jobnames):
    records = []
    for f in jobnames:
        records.append(parse_job_log(f.strip()))
    return records


def write_throughput(metrics, fname):
    tps = [m[0][0] for m in metrics]
    dps = [m[0][1] for m in metrics]
    pps = [m[0][2] for m in metrics]
    throughputs = [m[1] for m in metrics]
    df = pd.DataFrame({"tp": tps, "dp": dps, "pp": pps, "throughput": throughputs})
    df.to_csv(fname, index=False)


def main():
    log_dirs = list(glob.glob("logs-*"))
    bert_jobs = sorted(select_logs(log_dirs, "bert", 16))
    bert_metrics = parse_logs(bert_jobs)
    write_throughput(bert_metrics, "bert.csv")
    gpt_jobs = sorted(select_logs(log_dirs, "gpt", 16))
    gpt_metrics = parse_logs(gpt_jobs)
    write_throughput(gpt_metrics, "gpt.csv")


if __name__ == "__main__":
    main()
