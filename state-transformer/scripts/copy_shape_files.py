import os
import shutil


def main():
    size = 8
    pp = 4
    mp = 2
    dp = size // (pp * mp)

    src_dir = "/data/marcel/training"
    dst_dir = os.path.join(
        os.path.expanduser('~'),
        "Elasticity/Repo/transformer-checkpoint/deepspeed/gpt-2/6dot7B")
    dst_dir = os.path.join(dst_dir, f"pp{pp:02d}/mp{mp:02d}/dp{dp:02d}")

    for rank in range(size):
        src_file = os.path.join(src_dir,
                                f"{rank}/ckpt/param_shapes_{rank:02d}.json")
        dst_rank_dir = os.path.join(dst_dir, f"rank{rank:02d}")
        if not os.path.exists(dst_rank_dir):
            os.makedirs(dst_rank_dir)
        dst_file = os.path.join(dst_rank_dir,
                                f"optimiser_param_shapes_{rank:02d}.json")
        shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    main()
