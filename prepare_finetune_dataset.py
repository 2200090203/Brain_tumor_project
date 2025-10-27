"""Prepare a small fine-tuning dataset directory using existing YES MRI images and mined hard negatives.

This script creates `data_finetune/` with `train/yes`, `train/no`, `val/yes`, `val/no` by sampling images.

Usage:
  python prepare_finetune_dataset.py --yes_dir dataset_images/yes --hard_no_dir hard_negatives/no --out data_finetune --val_frac 0.2
"""
import argparse
import os
import random
import shutil


def make_dirs(base):
    for split in ['train', 'val']:
        for cls in ['yes', 'no']:
            d = os.path.join(base, split, cls)
            os.makedirs(d, exist_ok=True)


def copy_sample(src_dir, dst_dir, n, shuffle=True):
    files = [f for f in sorted(os.listdir(src_dir)) if os.path.isfile(os.path.join(src_dir, f))]
    if shuffle:
        random.shuffle(files)
    files = files[:n]
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def prepare(yes_dir, hard_no_dir, out_dir, val_frac=0.2, max_yes=None, max_no=None):
    # list files
    yes_files = [f for f in sorted(os.listdir(yes_dir)) if os.path.isfile(os.path.join(yes_dir, f))]
    no_files = [f for f in sorted(os.listdir(hard_no_dir)) if os.path.isfile(os.path.join(hard_no_dir, f))]

    if max_yes:
        yes_files = yes_files[:max_yes]
    if max_no:
        no_files = no_files[:max_no]

    random.shuffle(yes_files)
    random.shuffle(no_files)

    n_yes = len(yes_files)
    n_no = len(no_files)
    n_yes_val = int(n_yes * val_frac)
    n_no_val = int(n_no * val_frac)

    make_dirs(out_dir)

    # copy yes
    for i, f in enumerate(yes_files):
        split = 'val' if i < n_yes_val else 'train'
        dst = os.path.join(out_dir, split, 'yes', f)
        shutil.copy2(os.path.join(yes_dir, f), dst)

    # copy no
    for i, f in enumerate(no_files):
        split = 'val' if i < n_no_val else 'train'
        dst = os.path.join(out_dir, split, 'no', f)
        shutil.copy2(os.path.join(hard_no_dir, f), dst)

    print(f'Prepared fine-tune dataset in {out_dir} (train: yes={n_yes-n_yes_val}, no={n_no-n_no_val}; val: yes={n_yes_val}, no={n_no_val})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes_dir', default='dataset_images/yes')
    parser.add_argument('--hard_no_dir', default='hard_negatives/no')
    parser.add_argument('--out', default='data_finetune')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--max_yes', type=int, default=None)
    parser.add_argument('--max_no', type=int, default=None)
    args = parser.parse_args()
    prepare(args.yes_dir, args.hard_no_dir, args.out, args.val_frac, args.max_yes, args.max_no)

if __name__ == '__main__':
    main()
