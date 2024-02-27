from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

def main():
    num_workers = 16
    job_id = "296d68f822"

    data = []
    for i in range(num_workers):
        tb_path = f"training/{job_id}/{i}/ckpt/tensorboard"
        if os.path.exists(tb_path):
            ea = event_accumulator.EventAccumulator(tb_path)
            ea.Reload()
            data_points = ea.Scalars("lm loss")
            for event in data_points:
                data.append([event.wall_time, event.step, event.value])

    data.sort(key=lambda x: x[0])
    # [print(x) for x in data]
    wall_times = [x[0] for x in data]
    steps = [x[1] for x in data]
    loss = [x[2] for x in data]

    fig, axes = plt.subplots(2,1)
    axes[0].plot(wall_times, loss)
    axes[1].plot(steps, loss)
    fig.savefig("fig.pdf")


if __name__ == "__main__":
    main()
