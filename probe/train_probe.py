"""Train a linear probe (logistic regression) on residual activations.

Functions return the trained probe and evaluation diagnostics.
"""
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def train_probe(unsafe_acts: np.ndarray, safe_acts: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    X = np.vstack([unsafe_acts, safe_acts])
    y = np.concatenate([np.ones(len(unsafe_acts)), np.zeros(len(safe_acts))])

    stratify = y if len(np.unique(y)) > 1 else None
    if stratify is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    probe = LogisticRegression(max_iter=2000)
    probe.fit(X_train, y_train)

    preds = probe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ROC AUC if probabilities available and both classes present
    roc = None
    if hasattr(probe, "predict_proba") and len(np.unique(y_test)) > 1:
        try:
            probs = probe.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probs)
        except Exception:
            roc = None

    report = classification_report(y_test, preds, zero_division=0)

    return {
        "probe": probe,
        "metrics": {"accuracy": acc, "roc_auc": roc},
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
    }
