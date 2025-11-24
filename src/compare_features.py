import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib
import os
import numpy as np

# ==============================
# Paths
# ==============================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# Load processed data
# ==============================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print(f"✅ Loaded full dataset: X_train={X_train.shape}, X_test={X_test.shape}")

# Load top 10 feature names
top_features = joblib.load(os.path.join(DATA_DIR, "top10_features.joblib"))
print(f"🔥 Top 10 features: {top_features}")

# Reduced datasets
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# ==============================
# Helper function: train & evaluate
# ==============================


def train_and_evaluate(name, Xtr, Xte):
    print(f"\n🚀 Training model using: {name}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)

    # Detect binary vs multiclass
    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) == 2

    # Compute metrics safely
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary" if is_binary else "weighted")

    # ROC-AUC for multiclass (one-vs-rest)
    try:
        roc = roc_auc_score(
            y_test,
            proba if not is_binary else proba[:, 1],
            multi_class="ovr" if not is_binary else "raise",
        )
    except Exception:
        roc = np.nan

    print(f"\n📊 Results ({name}):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print("Classification Report:\n", classification_report(y_test, preds, digits=3))

    return {"name": name, "acc": acc, "f1": f1, "roc": roc, "model": model}


# ==============================
# Train both versions
# ==============================
all_results = []
all_results.append(train_and_evaluate("All Features", X_train, X_test))
all_results.append(train_and_evaluate("Top 10 Features", X_train_top, X_test_top))

# ==============================
# Compare & save results
# ==============================
print("\n✅ Comparison Summary:")
for r in all_results:
    print(f"{r['name']:>15} → Accuracy: {r['acc']:.4f} | F1: {r['f1']:.4f} | ROC-AUC: {r['roc']:.4f}")

# Save best model
best = max(all_results, key=lambda x: (x["roc"] if not np.isnan(x["roc"]) else 0))
joblib.dump(best["model"], os.path.join(MODEL_DIR, f"best_model_{best['name'].replace(' ', '_').lower()}.joblib"))
print(f"\n🏆 Best model saved: {best['name']} → ROC-AUC={best['roc']:.4f}")
