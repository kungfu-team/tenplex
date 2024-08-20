import glob
import subprocess


def main():
    # hosts = ["komodo01", "komodo02", "komodo03", "komodo04"]
    hosts = ["komodo01", "komodo02"]

    for ho in hosts:
        subprocess.run(
            f"ssh {ho} docker ps -q -f name='worker' | xargs docker stop".split(" "),
            check=False,
        )


if __name__ == "__main__":
    main()
