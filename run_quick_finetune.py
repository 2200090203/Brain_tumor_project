"""Run a quick fine-tune using existing fine_tune.py with the prepared `data_finetune`.

This script calls the fine_tune CLI with conservative settings (epochs=3, unfreeze_layers=10) to quickly validate the pipeline.
"""
import subprocess
import sys
import os

cmd = [
    sys.executable, 'fine_tune.py',
    '--weights', 'models/best_model.h5',
    '--data_dir', 'data_finetune',
    '--img_size', '224',
    '--batch_size', '8',
    '--epochs', '3',
    '--unfreeze_layers', '10',
    '--save_dir', 'models/finetune_quick'
]
print('Running:', ' '.join(cmd))
ret = subprocess.run(cmd)
print('Return code:', ret.returncode)
