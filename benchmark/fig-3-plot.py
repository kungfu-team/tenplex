#!/usr/bin/env python3
import glob
import numpy as np
import re
import matplotlib.pyplot as plt


def grep(pattern, filename):
    for line in open(filename):
        if pattern in line:
            yield line.strip()


def parse_line(line):
    result = dict()
    for part in line.split('|'):
        kv = part.split(':')
        if len(kv) == 2:
            k, v = kv[0].strip(), kv[1].strip()
            result[k] = v
    return result


def parse_mdp(line):
    p = re.search('mp(\d+)-dp(\d+)-pp(\d+)', line)
    mdp = [p.group(1), p.group(2), p.group(3)]
    m, d, p = [int(x) for x in mdp]
    return m, d, p


def have_failed(filename):
    # print(filename)
    failed = False
    for line in open(filename):
        if 'RuntimeError' in line:
            failed = True
            break

    return failed


pattern = 'elapsed time per iteration'
key1 = 'elapsed time per iteration (ms)'
key2 = 'global batch size'


def parse_job_log(name):
    mdp = parse_mdp(name)
    files = glob.glob(name + "/*.log")
    # print(files)
    failed = False
    for f in files:
        failed |= have_failed(f)
    # print(files)
    # failed = have_failed(name + "/stage-00-worker-00.out.log")
    if failed:
        print('%s failed' % (name))
        return (mdp, 0)
    # print('%s succ' % (name))

    lines = sum([list(grep(pattern, f)) for f in files], [])

    values = []
    for l in lines:
        r = parse_line(l)
        v1 = float(r[key1])
        v2 = int(r[key2])
        v = v2 / v1 * 1000
        # print('{} {}'.format(v1, v2))
        # print(r[key])
        values.append(v)

    value = np.mean(values)
    print('%s succ: %.3f' % (name, value))

    return (mdp, value)
    # print(np.mean(values))
    # for k, v in sorted(r.items()):
    #     # print('{}: {}'.format(k, v))


def plot(records, name):
    size = "xl"
    # batch_size = 128
    plt.rcParams["hatch.linewidth"] = 3
    linewidth = 2

    throughputs = [r[1] for r in records]
    labels = ['%d\n%d\n%d' % r[0] for r in records]

    plt.rc("figure", figsize=[8, 4])
    fig, ax = plt.subplots()

    ax.bar(
        labels,
        throughputs,
        hatch="//",
        fill=False,
        edgecolor="tab:blue",
        linewidth=linewidth,
    )

    fontsize = 18
    labelsize = 16
    ax.tick_params(labelsize=labelsize)
    ax.set_ylabel("Throughput (samples/s)", fontsize=fontsize)
    ax.set_xlabel("Parallelization configuration", fontsize=fontsize)

    fig.tight_layout()
    plt.savefig(name)


def plot_jobs(jobnames):
    records = []
    for f in open(jobnames):
        records.append(parse_job_log(f.strip()))

    # for ((m, d, p), value) in records:
    #     print('{}-{}-{} {}'.format(m, d, p, value))

    # plot(records, 'bert_large.pdf')
    print()

def main():
    plot_jobs('benchmark/bert.txt')
    plot_jobs('benchmark/gpt.txt')

main()
