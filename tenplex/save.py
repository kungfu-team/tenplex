import os

from .mlfs_client import MLFSClient


def save(ckpt: dict, job_id: str, step: int, device_rank: int, mlfs_path: str,
         ip: str, port: int):
    save_path = os.path.join(mlfs_path, f"save{step}")
    if os.path.exists(save_path):
        print(f"SAVER save directory {save_path} already exists")
        return

    client = MLFSClient(ip, port)
    client.save_traverse(
        ckpt,
        os.path.join(f"job/{job_id}",
                     os.path.join(f"save{step}", str(device_rank))))
    client.upload_txt(f"job/{job_id}/iter", str(step))

    print(f"saved checkpoint at {save_path}")
