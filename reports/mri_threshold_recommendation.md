# MRI Heuristic & Threshold Recommendations

Date: 2025-10-19

Summary
-------
We evaluated the TFLite brain-tumor classifier on `dataset_images/no` (1500 negatives) and found a small fraction of high-score false positives: 41 images >0.5 (2.73%), 16 images >0.8 (1.07%), and 5 images >0.9 (0.33%). The top-10 false positives were inspected and all 10 returned high MRI-heuristic scores (0.83–0.92), so the existing heuristic gate (default threshold 0.45) would not block them.

Conclusion
----------
- Raising the MRI-heuristic threshold alone is unlikely to remove these particular false positives because they are MRI-like (or possibly mislabelled). Changing the threshold risks gating true MRIs as well.

Recommendations (short)
-----------------------
1. Manually inspect the top false positives (gallery created at `inspect_top10_no.html`) to check for mislabels or consistent artifacts. If they are mislabelled MRIs, fix labels and retrain.

2. Hard-negative mining and fine-tuning:
   - Collect all `no` images with model probability > 0.8 into a `hard_negatives/` folder.
   - Add them (labelled correctly) to the fine-tuning dataset and run a short fine-tune for a few epochs with class weights.
   - This directly teaches the model to reduce its score on near-MRI negatives.

3. Train a lightweight MRI vs non-MRI filter (recommended):
   - Heuristic is a useful quick gate but limited. A small CNN (MobileNetV2 or EfficientNet-lite) trained to discriminate MRI vs non-MRI will be far more reliable and fast.
   - Use existing `dataset_images/yes` as MRI positives and sample a diverse `non-MRI` set: random photos, CT/XRays, medical images that are not brain MRIs, and the `no` folder negatives.
   - Deploy the filter as the first step: if image is likely non-MRI, return a helpful error to the user explaining it doesn't look like a brain MRI.

4. Short-term configurable mitigation:
   - Keep the current heuristic gate at 0.45 to block obvious non-MRI photos.
   - Add an admin-only higher-sensitivity mode or telemetry that flags high-confidence positives for human review before notifying users.

Concrete action plan (commands & files)
--------------------------------------
- Create hard negatives folder (example):

```powershell
python evaluate_tflite_counts.py --model models/model.tflite --dir dataset_images/no --topk 0 > no_probs.txt
# parse no_probs.txt and copy files with prob > 0.8 into hard_negatives/
```

- Scaffold MRI filter training (example outline):
  - `mri_filter/train_filter.py` — small script that builds a lightweight Keras model, trains on `dataset_images/yes` and a `dataset_images_non_mri/` folder.
  - Evaluate on holdout set and export `models/mri_filter.tflite` for fast inference.

- Fine-tuning with hard negatives:
  - Add `hard_negatives/` to training with label `no` (or augment `train/`), reduce learning rate, unfreeze last block, train 3–5 epochs.

Notes and considerations
------------------------
- If these false positives come from a subset of scanners, slices, or preprocessing differences, identify and add scanner-specific augmentations or examples to training.
- Always preserve a validation split of genuine MRIs to make sure gating/fine-tuning does not reduce true positive recall.

Next steps I can take for you
----------------------------
- (A) Generate the `hard_negatives/` folder by extracting all `no` images with prob > 0.8 (I can run this now and create the folder).
- (B) Scaffold the MRI vs non-MRI filter training pipeline and optionally run a short prototype training (small subset to validate the idea).
- (C) Create the fine-tune script and run a small fine-tuning experiment using the mined hard negatives.

Choose A, B, or C and I will proceed. If you prefer, I can start with A to collect hard negatives for manual review.
