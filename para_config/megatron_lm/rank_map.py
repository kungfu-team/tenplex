import json
import os


def gen_rank_map(
    framework: str,
    model: str,
    model_size: str,
    precision: str,
    pp_size: int,
    tp_size: int,
    dp_size: int,
    job_dir: str,
    repo: str,
):
    size = pp_size * tp_size * dp_size
    out_dir = os.path.join(repo, f"{framework}/{precision}/{model}/{model_size}")
    out_dir = os.path.join(out_dir, f"pp{pp_size:02d}/mp{tp_size:02d}/dp{dp_size:02d}")
    gpus_container = 4
    num_nodes = size // gpus_container

    mapping = {}

    for rank in range(size):
        for node in range(num_nodes):
            rank_path = os.path.join(
                job_dir, f"{node}/ckpt/{rank}/rank_{rank:02d}.json"
            )

            if not os.path.exists(rank_path):
                continue

            with open(rank_path, "r", encoding="utf-8") as rank_file:
                ranks = json.load(rank_file)

            mapping[rank] = {
                "pp_rank": ranks["pp"],
                "mp_rank": ranks["tp"],
                "dp_rank": ranks["dp"],
            }

    path = os.path.join(out_dir, "rank_map.json")
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(mapping, json_file, indent=4)
