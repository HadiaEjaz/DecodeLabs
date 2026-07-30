# Data Classification Using AI — Iris Species Classifier

A supervised learning pipeline that classifies iris flowers into one of three species
(Setosa, Versicolor, Virginica) based on four physical measurements — built as
Project 2 (Data Classification Using AI) for DecodeLabs' AI Industrial Training Kit.

## How it works

The pipeline follows the **Input → Process → Output** framework from the project brief.

### Input
- **Dataset**: the classic Iris benchmark — 150 samples, 3 balanced classes, 4 features
  (sepal length, sepal width, petal length, petal width, all in cm).
- **Feature Scaling**: raw measurements are transformed using `StandardScaler` so every
  feature has mean 0 and variance 1. Without this, a feature with naturally larger
  numbers could unfairly dominate the distance calculations the algorithm depends on.

### Process
- **Train/Test Split**: the data is shuffled and split 80/20, stratified so each species
  is proportionally represented in both the training and test sets. The model only ever
  learns from the training set — the test set stays locked away for validation.
- **Algorithm — K-Nearest Neighbors (KNN)**: built on the "proximity principle" —
  similar things exist close together. To classify a new flower, KNN looks at its `K`
  closest neighbors in the training data and takes a majority vote on species.
- **Choosing K**: the script tests K values from 1–20 and reports the error rate for
  each, then picks a robust value rather than just the lowest error — very small K
  (like 1) tends to overfit to noise, while very large K becomes too generic.

### Output
- **Accuracy**: overall percentage of correct predictions.
- **F1 Score** (macro-averaged): balances precision and recall across all three
  species, which matters because accuracy alone can hide mistakes on harder-to-tell-
  apart classes.
- **Confusion Matrix**: shows exactly which species get confused with which — rows are
  actual species, columns are predicted species.

## Example output

```
Accuracy : 0.967
F1 Score : 0.967  (macro-averaged across all 3 classes)

Confusion Matrix (rows = actual, columns = predicted)
                setosa  versicolor   virginica
    setosa          10           0           0
versicolor           0           9           1
 virginica           0           0          10
```

## Files

| File | Purpose |
|---|---|
| `iris_classifier.py` | Full pipeline: load data, scale, split, tune K, train, evaluate |
| `requirements.txt` | Python dependencies (scikit-learn, numpy) |

## Running it

```bash
pip install -r requirements.txt
python iris_classifier.py
```

To classify a new flower of your own, edit the measurements passed to
`predict_new_sample()` near the bottom of the script.

## Why KNN, and why scale first?

KNN classifies a point by measuring its distance to every other point in the training
set — so if one feature is measured in different units or a wider range than the
others, it will unfairly dominate that distance calculation. Scaling every feature to
the same range first ensures the model treats sepal width and petal length as equally
important, deciding purely based on the actual shape of the flower rather than
arbitrary measurement scale.

---
Built as part of the DecodeLabs AI Engineering track, Batch 2026.
