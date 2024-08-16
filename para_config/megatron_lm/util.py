import os
import shutil

def remove_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
