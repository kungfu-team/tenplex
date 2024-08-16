import copy
import json
import os


def is_rng_in_keys(keys):
    """Check for RNG inside the key sequence"""
    for key in keys:
        if isinstance(key, str):
            if "rng" in key:
                return True
    return False


def find_tensors(value, keys: list = None):
    """Search for tensors within a checkpoint
    and return the key sequences to those tensors,
    except for RNG tensors"""
    if keys is None:
        keys = []
    if isinstance(value, dict):
        new_tensor_keys = []
        for key, val in value.items():
            if key == "tensor":
                if not is_rng_in_keys(keys):
                    return [keys]
                return None

            new_keys = copy.deepcopy(keys)
            new_keys.append(key)
            tensor_keys = find_tensors(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys
    if isinstance(value, (list, set, tuple)):
        new_tensor_keys = []
        for i, val in enumerate(value):
            new_keys = copy.deepcopy(keys)
            new_keys.append(i)
            tensor_keys = find_tensors(val, new_keys)
            if tensor_keys:
                new_tensor_keys.extend(tensor_keys)
        return new_tensor_keys

    return None


def gen_key_order(
    framework: str,
    model: str,
    model_size: str,
    precision: str,
    pp_size: int,
    tp_size: int,
    dp_size: int,
):
    total_size = pp_size * tp_size * dp_size
    direc = f"{framework}/{precision}/{model}/{model_size}"
    direc = os.path.join(direc, f"pp{pp_size:02d}/mp{tp_size:02d}/dp{dp_size:02d}")

    model_keys = {}

    for rank in range(total_size):
        rank_dir = os.path.join(direc, f"rank{rank:02d}")

        if not os.path.exists(rank_dir):
            continue

        path = ""
        for entry in os.scandir(rank_dir):
            if not entry.is_file():
                continue
            path = entry.path

        with open(path, "r", encoding="utf-8") as json_file:
            rank_struct = json.load(json_file)

        model_keys[rank] = find_tensors(rank_struct["model"], ["model"])

    path = os.path.join(direc, "model_keys.json")
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(model_keys, json_file, indent=4)
