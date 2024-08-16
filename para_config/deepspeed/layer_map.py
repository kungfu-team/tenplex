import json
import os
import re


def main():
    framework = 'deepspeed'
    pp_size = 2
    mp_size = 1
    dp_size = 2
    total_size = pp_size * mp_size * dp_size
    model_size = 'medium'
    direc = f'{framework}/gpt-2/{model_size}/pp{pp_size:02d}/mp{mp_size:02d}/dp{dp_size:02d}'

    mapping = dict()

    for rank in range(total_size):
        rank_dir = os.path.join(direc, f'rank{rank:02d}')
        if not os.path.exists(rank_dir):
            continue

        layer_numbers = []
        tp_rank = None
        for entry in os.scandir(rank_dir):
            pattern = r'layer_(\d+)-model_(\d+)-model_states.json'
            mat = re.match(pattern, entry.name)
            if mat is None:
                continue
            layer_num = int(mat.group(1))
            tp_rank = int(mat.group(2))
            layer_numbers.append(layer_num)

        if layer_numbers:
            mapping[rank] = {
                'tp_rank': tp_rank,
                'layer_numbers': layer_numbers
            }

    with open(f'{direc}/layer_map.json', 'w') as json_file:
        json.dump(mapping, json_file, indent=4)


if __name__ == "__main__":
    main()
