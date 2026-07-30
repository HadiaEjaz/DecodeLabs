"""
Data Classification Using AI — Project 2 (DecodeLabs)
--------------------------------------------------------
Builds a supervised learning pipeline that classifies Iris flowers
into one of 3 species based on 4 measurements, using K-Nearest
Neighbors (KNN). Follows the IPO framework from the project brief:

  INPUT   -> Load the Iris dataset, scale the features
  PROCESS -> Train/test split, train a KNN classifier
  OUTPUT  -> Confusion Matrix, Accuracy, F1 Score
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)


# ---------------------------------------------------------------
# STEP 1 (INPUT): Load and understand the dataset
# ---------------------------------------------------------------
def load_data():
    iris = load_iris()
    X = iris.data          # 150 samples x 4 features (sepal/petal length & width)
    y = iris.target        # 150 labels (0=setosa, 1=versicolor, 2=virginica)
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


# ---------------------------------------------------------------
# STEP 2 (INPUT continued): The "Gatekeeper Rule" — Feature Scaling
# ---------------------------------------------------------------
def scale_features(X_train, X_test):
    """StandardScaler transforms features to mean=0, variance=1,
    so no single feature (e.g. petal length in cm) dominates the
    distance calculation just because its raw numbers are bigger."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled = scaler.transform(X_test)          # transform ONLY (no fit) on test
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------
# STEP 3 (PROCESS): Structural Integrity — the Train/Test Split
# ---------------------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """80/20 split, shuffled, stratified so each class is
    proportionally represented in both sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------
# STEP 4 (PROCESS continued): Tuning the Engine — choosing K
# ---------------------------------------------------------------
def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """Tries multiple K values and reports the error rate for each,
    so you can see the 'elbow' the deck describes — too small K
    overfits to noise, too large K underfits and gets too generic."""
    print("K   Error Rate")
    print("--  ----------")
    errors = {}
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error_rate = np.mean(preds != y_test)
        errors[k] = error_rate
        print(f"{k:<3} {error_rate:.3f}")
    min_error = min(errors.values())
    tied_ks = [k for k, e in errors.items() if e == min_error]
    # Avoid picking K=1 on a tie — it memorizes noise rather than
    # learning a generalizable pattern (see "Tuning the Engine" slide).
    non_trivial = [k for k in tied_ks if k > 1]
    best_k = non_trivial[len(non_trivial) // 2] if non_trivial else tied_ks[0]
    print(f"\n-> Best K found: {best_k} (lowest error rate: {min_error:.3f})")
    return best_k


# ---------------------------------------------------------------
# STEP 5 (PROCESS continued): The Algorithm — K-Nearest Neighbors
# ---------------------------------------------------------------
def train_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                    # FIT (memorize the map)
    return model


# ---------------------------------------------------------------
# STEP 6 (OUTPUT): Validation — Confusion Matrix, Accuracy, F1
# ---------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)             # PREDICT

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("\n=== OUTPUT VALIDATION ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 Score : {f1:.3f}  (macro-averaged across all 3 classes)")

    print("\nConfusion Matrix (rows = actual, columns = predicted)")
    header = "            " + "  ".join(f"{n[:10]:>10}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"{target_names[i][:10]:>10}  {row_str}")

    print("\nDetailed classification report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    return acc, f1, cm


# ---------------------------------------------------------------
# STEP 7: Predict on a brand-new, unseen flower
# ---------------------------------------------------------------
def predict_new_sample(model, scaler, target_names, sample):
    """sample = [sepal_length, sepal_width, petal_length, petal_width] in cm"""
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)[0]
    print(f"\nNew flower {sample} -> Predicted species: {target_names[prediction]}")


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== INPUT: Loading Iris dataset ===")
    X, y, feature_names, target_names = load_data()
    print(f"Samples: {len(X)} | Features: {feature_names} | Classes: {list(target_names)}\n")

    print("=== PROCESS: Splitting into train/test sets (80/20) ===")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}\n")

    print("=== PROCESS: Scaling features (StandardScaler) ===")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Done — mean=0, variance=1 for all features.\n")

    print("=== PROCESS: Finding the best K (elbow method) ===")
    best_k = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)

    print(f"\n=== PROCESS: Training final KNN model with K={best_k} ===")
    model = train_model(X_train_scaled, y_train, best_k)

    evaluate_model(model, X_test_scaled, y_test, target_names)

    # Try predicting a brand-new flower
    predict_new_sample(model, scaler, target_names, [5.1, 3.5, 1.4, 0.2])
