import tifffile
import os

folder = "test_op"

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    try:
        img = tifffile.imread(path)
        print(f, img.shape, img.dtype)
    except Exception as e:
        print(f, "ERROR:", e)