import requests

from .mlfs_client import MLFSClient


def save(
    ckpt: dict,
    job_id: str,
    step: int,
    device_rank: int,
    mlfs_path: str,
    ip: str,
    port: int,
):
    path = f"/job/{job_id}/save/{device_rank}"
    print(f"save checkpoint to {path}")

    client = MLFSClient(ip, port)

    dire = None
    try:
        dire = client.get_dir(path)
    except requests.HTTPError:
        print(f"{path} does not exist yet")

    if dire:
        try:
            client.delete(path)
            print("deleted previous save dir")
        except requests.HTTPError as err:
            print(f"save delete {path} {err}")
            print(f"number of elements in dir {len(dire)}")
            raise err

    client.save_traverse(ckpt, path)
    client.upload_txt(f"job/{job_id}/iter", str(step))

    print(f"did save checkpoint to {path}")
