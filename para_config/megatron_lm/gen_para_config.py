import argparse
import json
import os
import subprocess

from structure import gen_structure
from util import remove_dir


def create_schedule(size: int):
    schedule = []
    schedule.append({"step": 0, "size": size})
    schedule.append({"step": 50, "size": 0})
    with open("schedule.json", "w", encoding="utf-8") as json_file:
        json.dump(schedule, json_file)


def create_para_config(pp: int, tp: int, dp: int):
    size = pp * tp * dp
    para_config = {size: {"dp_size": dp, "pp_size": pp, "mp_size": tp}}
    with open("para_config.json", "w", encoding="utf-8") as json_file:
        json.dump(para_config, json_file)


def create_training_args(model: str, size: str, precision: str, hosts: [str]) -> [str]:
    home = os.path.expanduser("~")
    user = os.getlogin()
    dataset = "enwiki"
    dataset_index = "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt"
    if model == "bert":
        dataset = "openwebtext"
        dataset_index = "/data/megatron-lm/bert/openwebtext/npzs_seq1024/indices.txt"

    args = [
        "-image",
        "kungfu.azurecr.io/mw-megatron-lm-23.06-update:v0.0.1",
        "-user",
        user,
        "-mlfs-port",
        str(20010),
        "-tenplex-prefix",
        os.path.join(home, ".tenplex"),
        "-framework",
        "megatron-lm",
        "-model",
        model,
        "-model-size",
        size,
        "-dataset",
        dataset,
        "-batch-size",
        str(128),
        "-micro-batch-size",
        str(8),
        "-precision",
        precision,
        "-index-url",
        dataset_index,
        "-hosts",
        ",".join(hosts),
        "-schedule-file",
        "schedule.json",
        "-para-config",
        "para_config.json",
        "-gpu-per-host",
        str(4),
        "-gpu-per-container",
        str(4),
        "-seq-length",
        str(1024),
        "-jobid",
        "gen-para-config",
        "-gen-para-config",
    ]

    return args


def commit_para_config(repo: str, model: str, size: str, pp: int, tp: int, dp: int):
    res = subprocess.run(
        f"cd {repo} && git diff", shell=True, check=True, capture_output=True, text=True
    )
    if res.stdout == "":
        print("Repo has not changed")
        return

    msg = f"update megatron-lm {model} {size} p{pp}t{tp}d{dp}"
    subprocess.run(
        f"cd {repo} && git commit -am'{msg}' && git push",
        shell=True,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--size", type=str)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--tp", type=int)
    args = parser.parse_args()
    hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3", "10.10.10.4"]
    home = os.path.expanduser("~")
    repo = os.path.join(home, "Tenplex/repo/transformer-checkpoint")

    cluster_size = args.dp * args.pp * args.tp
    create_schedule(cluster_size)
    create_para_config(args.pp, args.tp, args.dp)

    remove_dir("logs")
    remove_dir("training")

    cmd_args = create_training_args(args.model, args.size, args.precision, hosts)
    cmd = ["tenplex-run"]
    cmd.extend(cmd_args)
    cmd.extend(["2>&1", "|", "tee gen_para_config_training.log"])
    cmd_str = " ".join(cmd)
    subprocess.run(cmd_str, shell=True, check=True)

    gen_structure(
        args.model, args.size, args.precision, args.pp, args.tp, args.dp, repo
    )

    commit_para_config(repo, args.model, args.size, args.pp, args.tp, args.dp)


if __name__ == "__main__":
    main()
