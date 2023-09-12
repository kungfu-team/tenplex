from tenplex.mlfs_client import MLFSClient


def createTestTree(client):
    client.upload_txt("/a/b.txt", "1")
    client.upload_txt("/a/c/d.txt", "2")
    client.upload_txt("/a/c/e.txt", "3")


def test():
    ip = "localhost"
    port = 20010
    client = MLFSClient(ip, port)

    createTestTree(client)

    path = "/a"
    num_files, num_dirs = client.delete(path)
    print(f"num files {num_files}")
    print(f"num dirs {num_dirs}")
    assert num_files == 3
    assert num_dirs == 2


if __name__ == "__main__":
    test()
