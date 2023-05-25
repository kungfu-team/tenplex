import argparse

from tenplex.load import load_http


def main():
    #  parser = argparse.ArgumentParser(description='Write checkpoint')
    #  parser.add_argument('--device-rank', type=int)
    #  parser.add_argument('--mlfs-path', type=str)
    #  args = parser.parse_args()

    job_id = "13b4a21fc1"
    device_rank = 0
    #  ip = "155.198.152.18"
    ip = "localhost"
    port = 20010

    ckpt, step = load_http(job_id, device_rank, ip, port)
    print(f"ckpt {ckpt.keys()}")
    print(f"step {step}")

    print(ckpt["optimizer"]["fp32_from_fp16_params"][0])


if __name__ == '__main__':
    main()
