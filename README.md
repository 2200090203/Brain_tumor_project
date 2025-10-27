Brain Tumor Detection - Project Files

Files included:
- train.py         : Training script. Run with: python train.py --data_dir /path/to/Brain_Tumor_Dataset
- requirements.txt : Python packages required. Create a virtualenv and pip install -r requirements.txt
- This package DOES NOT include the dataset. Download the Br35H dataset from Kaggle and extract to a folder.

Steps to run:
1. Download dataset from Kaggle: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
2. Unzip so you have a folder with 'yes' and 'no' subfolders.
3. Place dataset folder path when running the script:
   python train.py --data_dir /full/path/to/Brain_Tumor_Dataset --epochs 20
4. Model & figures will be saved into models/ and figs/ directories.
