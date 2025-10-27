# Copilot instructions for brain_tumor_project

This project is a lightweight TensorFlow/Keras image classification pipeline for brain tumor detection using transfer learning. The goal of these instructions is to help an AI coding agent be productive quickly by documenting the architecture, conventions, workflows, and key files.

Key facts (quick):
- Main scripts: `train.py`, `fine_tune.py`, `evaluate_model.py`, `dataset.py`, `model.py`.
- Models and artifacts are saved under `models/` (examples: `best_model.h5`, `best_model.weights.h5`, `final_model.h5`).
- The dataset layout expected by utilities: either a root with class subfolders (e.g., `dataset_images/yes`, `dataset_images/no`) or explicit `train/` and `val/` subfolders.
- The code uses Keras ImageDataGenerator and expects 3-channel RGB input; grayscale images are converted to RGB via `color_mode='rgb'`.

What to change and why (concise rules):
- Prefer changing model architecture in `model.py` or the `load_model_from_file` fallback in `fine_tune.py`. Keep the EfficientNetB0 backbone usage consistent where intended.
- When editing training loops, preserve callbacks behavior: ModelCheckpoint may save either full model or weights-only; code expects both variants and uses robust loading in `evaluate_model.py` and `fine_tune.py`.
- Data generators: preserve `class_mode='categorical'` and `shuffle=False` for validation generators to keep `val_gen.classes` aligned with predictions.

Important files and patterns (examples):
- `train.py`: builds model via `build_model(...)` and uses `create_generators()` from `dataset.py`. It computes class weights using `sklearn.utils.class_weight` and uses ModelCheckpoint saving to `models/best_model.h5`.
- `fine_tune.py`: supports two dataset layouts (train/val folders OR single folder with `validation_split`). Unfreezing is implemented by counting layers with weights in `unfreeze_last_n_layers()`.
- `evaluate_model.py`: provides a resilient `try_load_weights()` that attempts `model.load_weights()` first, then `tf.keras.models.load_model()`, and finally `by_name` transfers.
- `dataset.py`: uses `ImageDataGenerator.flow_from_directory()` and returns `(train_generator, val_generator)`.

Developer workflows (commands you can run):
- Train (transfer-learn):
  - python train.py --data_dir <DATA_DIR> --epochs 15 --img_size 224 --batch_size 16
- Fine-tune (unfreeze last N layers):
  - python fine_tune.py --weights models/best_model.h5 --data_dir <DATA_DIR> --unfreeze_layers 50 --epochs 10 --save_dir models
- Evaluate:
  - python evaluate_model.py --data_dir dataset_images --weights models/best_model.weights.h5 --img_size 224

Conventions and gotchas discovered:
- Models may be saved either as full HDF5 model files (model.save(...)) or weights-only (`model.save_weights(...)`). Use `evaluate_model.py`'s robust loader when writing code that loads weights.
- When using Windows, `fine_tune.py` includes a fallback for pickle-related callback issues: if callbacks cause a pickle error, training is retried without callbacks and weights are saved manually.
- Validation generators must be created with `shuffle=False` (this is used by `evaluate_model.py` to map predictions to `val_gen.classes`). Do not change this unless you update evaluation logic.
- Image color mode: Pillow sometimes opens grayscale images as single-channel; the code forces `color_mode='rgb'` in `evaluate_model.py` and elsewhere to avoid shape mismatches with ImageNet backbones.

Integration points & external dependencies:
- Top dependencies: `tensorflow`, `matplotlib`, `scikit-learn`, `pillow`, `opencv-python` (see `requirements.txt`).
- Dataset: not included in repo. The README recommends downloading the Br35H dataset (Kaggle). Expect dataset to be placed in `dataset_images/` by default.

When making edits, prefer small, testable changes:
- Add a unit-test-like script under `tests/` only if you need CI; otherwise use the provided `evaluate_model.py` to sanity-check model loading and inference on a small sample folder.

If uncertain about which model file to load, follow `evaluate_model.py`'s strategy: try `model.load_weights`, then `tf.keras.models.load_model`, then `by_name` transfer.

Files to inspect first (ordered):
1. `README.md` (high-level run instructions)
2. `train.py` (entrypoint for training)
3. `fine_tune.py` (fine-tuning logic and Windows callback fallback)
4. `evaluate_model.py` (robust weight/model loading + evaluation pipeline)
5. `dataset.py` (generator settings and augmentation)
6. `model.py` (backbone and classifier head)

If you make changes that touch saved model formats, update `evaluate_model.py` and `fine_tune.py`'s loading code accordingly.

If anything here is missing or unclear, ask for the exact workflow you want to automate (train/fine-tune/evaluate) and include the sample dataset layout path. Feedback will be used to iterate on this file.
