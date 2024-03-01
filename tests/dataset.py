from tenplex.dataset import GPTDataset as TenplexGPTDataset


def main():
    dp_rank = 2
    mlfs_path = "/mnt/mlfs"
    dataset = TenplexGPTDataset(mlfs_path, dp_rank)


if __name__ == "__main__":
    main()
