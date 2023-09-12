from .mlfs_client import MLFSClient


def save(ckpt: dict, job_id: str, step: int, device_rank: int, mlfs_path: str,
         ip: str, port: int):
    path = f"/job/{job_id}/save/{device_rank}"
    print(f"save checkpoint to {path}")
    client = MLFSClient(ip, port)
    client.delete(path)

    client.save_traverse(ckpt, path)
    client.upload_txt(f"job/{job_id}/iter", str(step))

    print(f"did save checkpoint to {path}")
