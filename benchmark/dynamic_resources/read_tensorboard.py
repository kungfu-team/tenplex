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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--exper", default="", type=str)
    args = parser.parse_args()

    if args.exper not in ["tenplex", "tenplex_dp", "pytorch"]:
        print(f"argument {args.exper} is wrong")
        return

    job_id_map = {"tenplex": "dyn-res-ten", "tenplex_dp": "dyn-res-tdp"}
    metrics = []
    if args.exper in ["tenplex", "tenplex_dp"]:
        num_workers = 16
        job_id = job_id_map[args.exper]
        for i in range(num_workers):
            tb_path = f"training/{job_id}/{i}/ckpt/tensorboard"
            if os.path.exists(tb_path):
                sub_metrics = load_metrics(tb_path)
                metrics.extend(sub_metrics)
        metrics.sort(key=lambda x: x[0])
    elif args.exper == "pytorch":
        tb_path = "/mnt/k1d2/ckpt/tensorboard"
        metrics = load_metrics(tb_path)
    else:
        raise NotImplementedError()

    wall_times = [x[0] for x in metrics]
    steps = [x[1] for x in metrics]
    loss = [x[2] for x in metrics]

    np.savez(f"{args.exper}_loss.npz", wall_time=wall_times, step=steps, loss=loss)


if __name__ == "__main__":
    main()
