import io
import os

import numpy as np
import torch


class MLFSDataset(torch.utils.data.Dataset):

    def __init__(self, mlfs_path: str, job_id: str, dp_rank: int):
        self.mlfs_path = mlfs_path
        self.dp_rank = dp_rank

        path = os.path.join(self.mlfs_path, f"job/{job_id}", "head.txt")
        with open(path, "r", encoding="utf-8") as head_file:
            progress_path = head_file.read().strip()

        print(f"progress path {progress_path}")
        path = self.mlfs_path + progress_path
        with open(path, "r", encoding="utf-8") as progress_file:
            rank_paths = progress_file.readlines()

        rank_path = rank_paths[self.dp_rank].strip()

        path = self.mlfs_path + os.path.join(rank_path, "list.txt")
        with open(path, "r", encoding="utf-8") as list_file:
            self.data_file_paths = list_file.readlines()

        self.index_files_cache = {}
        self.offset = 0
        self.use_index_file(0)
        self.samples_per_file = self.num_samples

    def use_index_file(self, file_idx: int):
        self.current_file_idx = file_idx
        self.npzs_path = self.mlfs_path + self.data_file_paths[file_idx].strip(
        )

        self.indices_path = f"{self.npzs_path}.idx"
        with open(self.indices_path, "r", encoding="utf-8") as indices_file:
            lines = indices_file.readlines()

        line_split = lines[1].split(" ")
        self.num_samples = int(line_split[1])

        # First 2 lines are metadata
        lines = lines[2:]

        self.indices = []
        for line in lines:
            splitted = line.split(" ")
            self.indices.append((int(splitted[0]), int(splitted[1])))

        self.index_files_cache[file_idx] = self.indices

    def __len__(self):
        return self.indices[-1][1]

    def read_npz(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file

        if file_idx in self.index_files_cache:
            indices = self.index_files_cache[file_idx]
        else:
            self.use_index_file(file_idx)
            indices = self.indices

        sample_indices = indices[sample_idx]
        size = sample_indices[1] - sample_indices[0]
        with open(self.npzs_path, "rb") as npzs_file:
            npzs_file.seek(sample_indices[0])
            npz_sample = npzs_file.read(size)

        return npz_sample


class BERTDataset(MLFSDataset):
    def __getitem__(self, idx):
        npz_sample = self.read_npz(idx)
        buf = io.BytesIO(npz_sample)
        sample = np.load(buf)

        train_sample = {
            "text": sample["text"],
            "types": sample["types"],
            "labels": sample["labels"],
            "is_random": int(sample["is_random"]),
            "loss_mask": sample["loss_mask"],
            "padding_mask": sample["padding_mask"],
            "truncated": int(sample["truncated"]),
        }

        buf.close()

        return train_sample


class GPTDataset(MLFSDataset):
    def __getitem__(self, idx):
        npz_sample = self.read_npz(idx)
        buf = io.BytesIO(npz_sample)
        sample = np.load(buf)

        train_sample = {"text": sample["text"]}

        buf.close()

        return train_sample
