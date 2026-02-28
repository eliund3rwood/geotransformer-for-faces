import os
import numpy as np
import shutil

def split(train=0.8, path="UHM_downsampled"):
    
    files = [f for f in os.listdir(path) if f.endswith(".ply")]
    n = len(files)
    k = int(n * train)

    np.random.seed(42)
    perm = np.random.permutation(n)
    train_idx = perm[:k]
    test_idx = perm[k:]

    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i in train_idx:
        src = os.path.join(path, files[i])
        dst = os.path.join(train_dir, files[i])
        shutil.move(src, dst)

    for i in test_idx:
        src = os.path.join(path, files[i])
        dst = os.path.join(test_dir, files[i])
        shutil.move(src, dst)





