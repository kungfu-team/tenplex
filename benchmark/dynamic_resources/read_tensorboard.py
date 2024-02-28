import argparse
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


def load_metrics(tb_path):
    metrics = []
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    data_points = ea.Scalars("lm loss")
    for event in data_points:
        metrics.append([event.wall_time, event.step, event.value])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tenplex",
                        action="store_true")
    args = parser.parse_args()

    metrics = []
    if args.tenplex:
        num_workers = 16
        job_id = "tenplex-dyn-res"
        for i in range(num_workers):
            tb_path = f"training/{job_id}/{i}/ckpt/tensorboard"
            if os.path.exists(tb_path):
                sub_metrics = load_metrics(tb_path)
                metrics.extend(sub_metrics)
        metrics.sort(key=lambda x: x[0])
    else:
        tb_path = "/mnt/k1d2/ckpt/tensorboard"
        metrics = load_metrics(tb_path)

    wall_times = [x[0] for x in metrics]
    steps = [x[1] for x in metrics]
    loss = [x[2] for x in metrics]

    np.savez("loss.npz", wall_time=wall_times, step=steps, loss=loss)

    fig, axes = plt.subplots(2,1)
    axes[0].plot(wall_times, loss)
    axes[1].plot(steps, loss)
    fig.savefig("fig.pdf")


if __name__ == "__main__":
    main()
