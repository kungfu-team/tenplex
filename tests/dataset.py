import subprocess
from tenplex.dataset import GPTDataset as TenplexGPTDataset


def main():
    num_scaling = 5
    idx_path = "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt"
    dp_size = 2
    dp_rank = 1
    job_id = "dataset-test"
    batch_size = 128

    for _ in range(num_scaling):
        progress = num_scaling * batch_size * 2048

        mount_cmd = [
            "mlfs", "mount",
            "-idx-name", "openwebtext",
            "-index-url", f"{idx_path}",
            "-ctrl-port", "20010",
            "-progress", f"{progress}",
            "-global-batch-size", f"{batch_size}",
            "-dp-size",  f"{dp_size}",
            "-job", job_id,
        ]
        subprocess.run(mount_cmd, check=True)
        print("finished MLFS mount")

        mlfs_path = "/mnt/mlfs"
        dataset = TenplexGPTDataset(mlfs_path, job_id, dp_rank)

        for i, sample in enumerate(dataset):
            txt = sample["text"]
            if i > 10_000:
                break


if __name__ == "__main__":
    main()
