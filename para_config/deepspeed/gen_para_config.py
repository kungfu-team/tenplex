import collections
import json
import os

import torch


def create_value_dict(value):
    if isinstance(value, (collections.OrderedDict, dict)):
        elements = {}
        for key, val in value.items():
            elements[key] = create_value_dict(val)
        return elements
    if isinstance(value, torch.Tensor):
        return tuple(value.size())
    if isinstance(value, torch.Size):
        return "Size " + str(tuple(value))
    if isinstance(value, (list, set, tuple)):
        elements = {}
        for i, val in enumerate(value):
            elements[i] = create_value_dict(val)
        return elements
    if isinstance(value, (int, float, str)):
        return value

    return str(type(value))


def create_ckpt_dict(ckpt):
    elements = {}

    for key, val in ckpt.items():
        elements[key] = create_value_dict(val)

    return elements


def mdp():
    job_id = "41508a65bd"
    base_dir = os.path.join(
        os.path.expanduser('~'),
        f".tenplex/training/{job_id}")
    size = 4
    pp = 2
    mp = 2
    dp = size // (pp * mp)
    step = 50
    out_dir = "./test"
    #  out_dir = os.path.join(
    #      os.path.expanduser('~'),
    #      "Elasticity/Repo/transformer-checkpoint/deepspeed/gpt-2/6dot7B")
    out_dir = os.path.join(out_dir, f"pp{pp:02d}/mp{mp:02d}/dp{dp:02d}")
    os.makedirs(out_dir, exist_ok=True)

    for rank in range(size):
        rank_input_path = os.path.join(base_dir,
                                       f"{rank}/ckpt/global_step{step}")

        if not os.path.isdir(rank_input_path):
            continue

        rank_output_path = os.path.join(out_dir, f"rank{rank:02d}")
        os.makedirs(rank_output_path)

        for entry in os.scandir(rank_input_path):
            if not entry.is_file():
                continue

            ckpt = torch.load(entry.path)
            ckpt_dict = create_ckpt_dict(ckpt)
            new_name = entry.name.split(".")[0] + ".json"
            out_path = os.path.join(rank_output_path, new_name)

            with open(out_path, "w") as dict_file:
                json.dump(ckpt_dict, dict_file, indent=4)


def main():
    mdp()


if __name__ == "__main__":
    main()
