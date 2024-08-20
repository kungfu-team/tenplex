import argparse
import shutil
import subprocess
import time


def create_cmd(worker: int, image: str, model_size: str, com_backend: str):
    docker_cmd = (
        "docker run "
        "--rm "
        "--network host "
        "-v /mnt/k1d2/megatron-lm/gpt-2:/data/gpt "
        "-v /mnt/k1d2/ckpt:/data/ckpt "
        "--device /dev/infiniband/uverbs0 "
        "--expose 29400 "
        f"--gpus all "
        f"--name worker{worker} "
        "--ulimit memlock=-1 "
        "--shm-size=1gb "
        "--env GLOO_SOCKET_IFNAME=eth0 "
        # "--env NCCL_DEBUG=INFO "
        # "--env TORCH_CPP_LOG_LEVEL=INFO "
        "-d "
        f"-t {image} "
    )
    if model_size == "large":
        docker_cmd += f"./run_gpt.sh {worker} {com_backend}"
    elif model_size == "xl":
        docker_cmd += f"./run_gpt_xl.sh {worker} {com_backend}"
    else:
        raise NotImplementedError

    return docker_cmd


def start_container(hosts: list, i: int, image: str, model_size: str, com_backend: str):
    docker_cmd = create_cmd(i, image, model_size, com_backend)
    host = hosts[i]
    cmd = f'ssh {host} "{docker_cmd}"'
    subprocess.run(cmd, check=True, shell=True)


def stop_container(host):
    subprocess.run(
        f"ssh {host} \"docker ps -aq -f name='worker' | xargs docker stop\"",
        check=True,
        shell=True,
    )


def main():
    # hosts = ["komodo01", "komodo02", "komodo03", "komodo04"]
    hosts = ["komodo01", "komodo02"]
    num_hosts = len(hosts)
    # image = "kungfu.azurecr.io/mw-megatron-deepspeed:latest"
    image = "kungfu.azurecr.io/mw-megatron-deepspeed-update:latest"
    model_size = "xl"
    parser = argparse.ArgumentParser(description="Deepspeed")
    parser.add_argument(
        "--scale-up",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    scale_up = args.scale_up
    com_backend = "nccl" if scale_up else "gloo"

    # Pull image
    for host in hosts:
        subprocess.run(f"ssh {host} docker pull {image}".split(" "), check=True)

    # Start
    cur_hosts = num_hosts
    if scale_up:
        cur_hosts = num_hosts // 2
    for i in range(cur_hosts):
        start_container(hosts, i, image, model_size, com_backend)
    print("Started training")

    # Wait
    time.sleep(3 * 60)

    # Scale
    if scale_up:
        for i in range(num_hosts // 2, num_hosts):
            start_container(hosts, i, image, model_size, com_backend)
            print(f"Finished docker run -d {time.time()}")
    else:
        for i in range(num_hosts // 2, num_hosts):
            stop_container(hosts[i])
            print(f"Stopped container {i}")
        print(f"Finished docker stop {time.time()}")

    # Wait
    time.sleep(3 * 60)

    # Stop
    cur_hosts = num_hosts
    if not scale_up:
        cur_hosts = num_hosts // 2
    for i in range(cur_hosts):
        stop_container(hosts[i])
    print("Stopped Training")

    # Collect logs
    name = "scale_up" if scale_up else "scale_down"
    for i in range(num_hosts):
        shutil.copyfile(f"/mnt/k1d2/ckpt/worker{i}.log", f"{name}_{i}.log")


if __name__ == "__main__":
    main()
