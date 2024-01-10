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


def compare_partitions(part_a, part_b):
    for key in part_a.keys():
        if key not in part_b:
            print(f"{key} not in b")
        else:
            if part_a[key] != part_b[key]:
                print(f"{key} is unequal")


def two():
    base_dir = "/data/marcel/deepspeed/mp2"
    step = 2000
    ranks = ["00", "01"]
    models = []
    opts = []

    for rank in ranks:
        path = os.path.join(base_dir, f"global_step{step}/mp_rank_{rank}_model_states.pt")
        ckpt = torch.load(path)
        model = create_ckpt_dict(ckpt)
        models.append(model)
        with open(f"model_mp2_{rank}.json", "w") as dict_file:
            json.dump(model, dict_file, indent=4)

        path = os.path.join(base_dir, f"global_step{step}/zero_pp_rank_0_mp_rank_{rank}_optim_states.pt")
        ckpt = torch.load(path)
        opt = create_ckpt_dict(ckpt)
        opts.append(opt)
        with open(f"opt_mp2_{rank}.json", "w") as dict_file:
            json.dump(opt, dict_file, indent=4)


def one():
    base_dir = "/data/marcel/training/0/ckpt"
    step = 50
    rank = 0

    path = os.path.join(base_dir, f"global_step{step}/mp_rank_{rank:02d}_model_states.pt")
    ckpt = torch.load(path)
    model = create_ckpt_dict(ckpt)
    with open("model.json", "w") as dict_file:
        json.dump(model, dict_file, indent=4)

    path = os.path.join(base_dir, f"global_step{step}/layer_06-model_00-model_states.pt")
    ckpt = torch.load(path)
    model = create_ckpt_dict(ckpt)
    with open("layer.json", "w") as dict_file:
        json.dump(model, dict_file, indent=4)


def mdp():
    mp_rank = 0
    base_dir = f"/data/marcel/cont_training/{mp_rank}/ckpt"
    step = 100
    pp_rank = 0

    path = os.path.join(base_dir, f"global_step{step}/mp_rank_{mp_rank:02d}_model_states.pt")
    ckpt = torch.load(path)
    model = create_ckpt_dict(ckpt)
    with open(f"model_{mp_rank:02d}.json", "w") as dict_file:
        json.dump(model, dict_file, indent=4)

    for layer in range(24):
        path = os.path.join(base_dir, f"global_step{step}/layer_{layer:02d}-model_{pp_rank:02d}-model_states.pt")
        try:
            ckpt = torch.load(path)
        except Exception as e:
            print(e)
            continue
        model = create_ckpt_dict(ckpt)
        with open(f"layer_{pp_rank:02d}_{layer:02d}.json", "w") as dict_file:
            json.dump(model, dict_file, indent=4)


def main():
    mdp()


if __name__ == "__main__":
    main()
