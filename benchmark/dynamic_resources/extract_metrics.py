import argparse
import os

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_metrics(tb_path):
    metrics = []
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    data_points = ea.Scalars("lm loss")
    for event in data_points:
        metrics.append([event.wall_time, event.step, event.value])
    return metrics


def load_tenplex(job_id: str):
    num_containers = 4
    metrics = []
    for i in range(num_containers):
        tb_path = f"training/{job_id}/{i}/ckpt/tensorboard"
        if os.path.exists(tb_path):
            sub_metrics = load_metrics(tb_path)
            metrics.extend(sub_metrics)
    metrics.sort(key=lambda x: x[0])
    return metrics


def load_tde():
    tb_path = "/mnt/k1d2/ckpt/tensorboard"
    metrics = load_metrics(tb_path)
    return metrics


def write_metrics(metrics: list, exper: str):
    wall_times = [x[0] for x in metrics]
    steps = [x[1] for x in metrics]
    loss = [x[2] for x in metrics]

    np.savez(f"{exper}_loss.npz", wall_time=wall_times, step=steps, loss=loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--exper", default=None, type=str)
    args = parser.parse_args()

    if args.exper is None:
        print("Argument --exper is None")
        return

    if args.exper == "dyn-res-tde":
        metrics = load_tde()
    else:
        metrics = load_tenplex(args.exper)

    write_metrics(metrics, args.exper)


if __name__ == "__main__":
    main()
