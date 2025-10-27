"""Compute is_probable_mri() scores for the top-10 false positives and report gate decisions.

Usage:
  python compute_mri_scores_top10.py --dir dataset_images/no --files no520.jpg no445.jpg ... --threshold 0.45

If --files not provided, defaults to the known top-10.
"""
import argparse
import os
from PIL import Image
import numpy as np


def is_probable_mri(pil_image, target_size=(224,224)):
    """Heuristic MRI score in [0,1].
    Uses: low saturation (grayscale-ish), dark corners, bright center.
    """
    img = pil_image.copy().convert('RGB').resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0

    # Convert to HSV to measure saturation
    import colorsys
    hsv = np.zeros(arr.shape, dtype=np.float32)
    # vectorized rgb->hsv over pixels
    r = arr[:,:,0].flatten()
    g = arr[:,:,1].flatten()
    b = arr[:,:,2].flatten()
    hsv_flat = [colorsys.rgb_to_hsv(rr, gg, bb) for rr, gg, bb in zip(r, g, b)]
    hsv_flat = np.array(hsv_flat)
    s = hsv_flat[:,1].reshape((target_size[1], target_size[0]))
    mean_s = float(np.mean(s))

    # Corner darkness: average brightness in four corners
    gray = np.dot(arr, [0.2989, 0.5870, 0.1140])
    h, w = gray.shape
    pad = int(min(h,w) * 0.15)
    corners = [gray[:pad, :pad], gray[:pad, -pad:], gray[-pad:, :pad], gray[-pad:, -pad:]]
    corner_darkness = np.mean([1.0 - np.mean(c) for c in corners])  # 0..1

    # Center brightness: mean brightness in center square
    ch = int(h*0.4); cw = int(w*0.4)
    start_h = (h - ch)//2; start_w = (w - cw)//2
    center = gray[start_h:start_h+ch, start_w:start_w+cw]
    center_brightness = float(np.mean(center))  # 0..1

    # Combine features: favor low saturation, high corner darkness, medium-high center brightness
    sat_score = max(0.0, 1.0 - mean_s*2.0)  # low saturation -> near 1
    corner_score = corner_darkness
    center_score = np.clip((center_brightness - 0.3) / 0.7, 0.0, 1.0)

    # Weighted sum
    score = 0.45 * sat_score + 0.35 * corner_score + 0.20 * center_score
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='dataset_images/no')
    parser.add_argument('--files', nargs='*', help='list of filenames relative to --dir')
    parser.add_argument('--threshold', type=float, default=0.45)
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = ['no520.jpg','no445.jpg','no365.jpg','no472.jpg','no383.jpg','no456.jpg','no543.jpg','no1306.jpg','no815.jpg','no789.jpg']

    print('Threshold:', args.threshold)
    print('\nResults:')
    for f in files:
        p = os.path.join(args.dir, f)
        if not os.path.isfile(p):
            print(f, 'MISSING')
            continue
        img = Image.open(p)
        score = is_probable_mri(img)
        blocked = score < args.threshold
        print(f, '->', f'score={score:.6f}', 'blocked=' + str(blocked))

if __name__ == '__main__':
    main()
