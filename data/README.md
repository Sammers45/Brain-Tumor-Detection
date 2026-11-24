# Data

**Source:** *Brain MRI Images for Brain Tumor Detection* (Navoneel Chakrabarty, Kaggle):
<https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>

253 axial brain MRI slices (JPG/PNG, mixed sizes and contrasts):

| Folder | Label | Count |
|---|---|---|
| `yes/` | 1 = tumour | 155 |
| `no/`  | 0 = no tumour | 98 |

## Setup

Download the dataset and put it here:

```
data/raw/brain_tumor_dataset/
├── yes/
└── no/
```

Then build the manifest:

```bash
PYTHONPATH=src python -m brain_tumor.data
```

This writes `data/processed/manifest.csv` (committed). It has one row per image with
`path, label, md5, group, split, cv_fold`. Because the split is stored in a file, every
run, notebook and teammate uses exactly the same train/val/test images.

## Known data issues (found by `brain_tumor.data`)

- **25 byte-identical duplicate files** and **73 images in 32 near-duplicate groups**
  (the same scan saved again at a different size or compression). These are grouped so
  that copies of one scan never end up in both train and test. A random split would
  leak them and inflate test scores.
- Class imbalance of 61% vs 39%. This is handled with class weights, stratified splits
  and threshold tuning.
- These are 2D slices without patient IDs, so the pipeline can't guarantee that two
  *different* slices of the same patient stay in the same split. Treat the reported
  metrics as optimistic for that reason.
- The dataset is small and comes from a single public source. A model trained on it is
  a learning and portfolio project. It is **not** clinically validated.
