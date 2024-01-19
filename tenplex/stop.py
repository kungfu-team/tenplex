import os

import requests
import torch


def check_stop(scheduler_addr: str):
    if scheduler_addr is None:
        return False
    stop = False
    rank = torch.distributed.get_rank()
    if rank == 0:
        url = scheduler_addr
        url = os.path.join(url, "stop")
        req = requests.get(url, timeout=12)
        txt = req.text
        if txt == "stop":
            stop = True
    if stop:
        stop_ten = torch.tensor(1, dtype=torch.int32, device=torch.device("cuda"))
    else:
        stop_ten = torch.tensor(0, dtype=torch.int32, device=torch.device("cuda"))
    torch.distributed.all_reduce(stop_ten)
    if stop_ten == 1:
        return True
    return False
