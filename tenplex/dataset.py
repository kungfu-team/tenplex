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

        self.offset = 0
        self.use_index_file(0)

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

    def __len__(self):
        return self.indices[-1][1]


class BERTDataset(MLFSDataset):

    def __getitem__(self, idx):
        file_idx = idx - self.offset
        if file_idx >= self.num_samples:
            self.offset = self.offset + self.num_samples
            self.use_index_file(self.current_file_idx + 1)
            file_idx = idx - self.offset

        file_indices = self.indices[file_idx]
        size = file_indices[1] - file_indices[0]
        with open(self.npzs_path, "rb") as npzs_file:
            npzs_file.seek(self.indices[file_idx][0])
            npz_sample = npzs_file.read(size)

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
        try:
            return self.f(idx)
        except Exception as e:
            # dump here
            print(f'self.offset: {self.offset}')
            print(f'idx: {idx}')
            file_idx = idx - self.offset
            print(f'file_idx: {file_idx}')
            raise e

    def f(self, idx):
        file_idx = idx - self.offset
        if file_idx < 0:
            raise ValueError("file index {file_id} < 0, offset {self.offset}")
        if file_idx >= self.num_samples:
            print(f'moving to next file: {self.current_file_idx} + 1')
            self.offset = self.offset + self.num_samples
            self.use_index_file(self.current_file_idx + 1)
            print(f'moved to next file: {self.current_file_idx} with num samples {self.num_samples}')
            file_idx = idx - self.offset
            if file_idx >= self.num_samples:
                raise ValueError('move next is not enough: file_idx >= self.num_samples')
        if file_idx > len(self):
            raise ValueError(f"file index larger length {len(self)}")

        file_indices = self.indices[file_idx]
        size = file_indices[1] - file_indices[0]
        with open(self.npzs_path, "rb") as npzs_file:
            npzs_file.seek(self.indices[file_idx][0])
            npz_sample = npzs_file.read(size)

        buf = io.BytesIO(npz_sample)
        sample = np.load(buf)

        train_sample = {"text": sample["text"]}

        buf.close()

        return train_sample
