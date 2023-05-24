from .mlfs_client import MLFSClient


def save(ckpt: dict, job_id: str, step: int, device_rank: int, mlfs_path: str,
         ip: str, port: int):
    save_path = f"/job/{job_id}/save{step}/{device_rank}"
    print(f"save checkpoint to {save_path}")
    client = MLFSClient(ip, port)
    dire = client.get_dir(save_path)
    if not dire[0].startswith("ERROR"):
        print(f"save directory {save_path} already exists")
        return

    client.save_traverse(ckpt, save_path)
    client.upload_txt(f"job/{job_id}/iter", str(step))

    print(f"did save checkpoint to {save_path}")
