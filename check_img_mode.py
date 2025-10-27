# check_img_mode.py — prints mode and shape for one sample image in your dataset
import os
from PIL import Image
import numpy as np

root = 'dataset_images'
for cls in sorted(os.listdir(root)):
    cls_path = os.path.join(root, cls)
    if not os.path.isdir(cls_path):
        continue
    for fname in os.listdir(cls_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fp = os.path.join(cls_path, fname)
        try:
            img = Image.open(fp)
            print("sample file:", fp)
            print("PIL mode:", img.mode)
            arr = np.array(img)
            print("numpy shape:", arr.shape)
        except Exception as e:
            print("ERROR reading:", fp, e)
        raise SystemExit  # stop after first image
print("No images found.")
