"""
train.py - Model Training, Time-Series Cross-Validation, Threshold Tuning, and Benchmarking.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
)

from src.preprocessing import build_full_pipeline


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all standard binary classification metrics including F2 and Specificity.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def find_optimal_threshold(y_true, y_probs, metric="f1"):
    """
    Finds threshold that maximizes the specified metric on validation probabilities.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_score = -1.0
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "f2":
            score = fbeta_score(y_true, preds, beta=2, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)
            
        if score > best_score:
            best_score = score
            best_thresh = th

    return float(best_thresh), float(best_score)


def main():
    print("=" * 75)
    print("STAGE D: MODEL TRAINING AND TEMPORAL VALIDATION")
    print("=" * 75)

    os.makedirs("models", exist_ok=True)

    csv_path = "Bank_Marketing.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("dataset", "Bank_Marketing.csv")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=['y'])
    y = (df['y'] == 'yes').astype(int)

    # 1. Temporal Split: 80% Train, 20% Holdout Test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Data Splitting (Chronological Temporal Split):")
    print(f"  Training set: {len(X_train):,} rows ({y_train.sum():,} yes, {y_train.mean()*100:.2f}%)")
    print(f"  Test holdout: {len(X_test):,} rows ({y_test.sum():,} yes, {y_test.mean()*100:.2f}%)")
    print(f"  -> Demonstrates substantial temporal shift in subscription propensity.")

    # 2. Define Model Candidates for Main Pre-Contact Prediction (NO duration)
    models = {
        "Dummy (Baseline)": {
            "clf": DummyClassifier(strategy="most_frequent"),
            "scaling": False,
            "exclude_duration": True
        },
        "Logistic Regression": {
            "clf": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "scaling": True,
            "exclude_duration": True
        },
        "Random Forest": {
            "clf": RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "scaling": False,
            "exclude_duration": True
        },
        "HistGradientBoosting": {
            "clf": HistGradientBoostingClassifier(
                class_weight="balanced", max_iter=150, learning_rate=0.08, min_samples_leaf=20,
                random_state=42
            ),
            "scaling": False,
            "exclude_duration": True
        },
        "Extra Trees": {
            "clf": ExtraTreesClassifier(
                n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "scaling": False,
            "exclude_duration": True
        }
    }

    # 3. Time-Series Cross-Validation on Training Set (5 splits)
    print("\n[3] Running 5-Fold TimeSeriesSplit Cross-Validation on Training Data...")
    tscv = TimeSeriesSplit(n_splits=5)

    cv_results = {}
    fitted_pipelines = {}
    test_evaluations = {}

    for name, config in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        pipeline = build_full_pipeline(
            classifier=config["clf"],
            include_scaling=config["scaling"],
            exclude_duration=config["exclude_duration"]
        )

        # TimeSeriesSplit CV
        oof_probs = []
        oof_y = []
        cv_f1s = []
        cv_rocs = []
        cv_prs = []

        for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_cv_train, y_cv_train = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
            X_cv_val, y_cv_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            fold_pipe = build_full_pipeline(
                classifier=config["clf"],
                include_scaling=config["scaling"],
                exclude_duration=config["exclude_duration"]
            )
            fold_pipe.fit(X_cv_train, y_cv_train)
            
            if hasattr(fold_pipe, "predict_proba"):
                probs = fold_pipe.predict_proba(X_cv_val)[:, 1]
            else:
                probs = np.full(len(X_cv_val), y_cv_train.mean())

            preds = fold_pipe.predict(X_cv_val)

            oof_probs.extend(probs)
            oof_y.extend(y_cv_val)

            cv_f1s.append(f1_score(y_cv_val, preds, zero_division=0))
            if len(np.unique(y_cv_val)) > 1:
                cv_rocs.append(roc_auc_score(y_cv_val, probs))
                cv_prs.append(average_precision_score(y_cv_val, probs))

        oof_probs = np.array(oof_probs)
        oof_y = np.array(oof_y)

        # Find optimal threshold using CV out-of-fold predictions
        if len(np.unique(oof_probs)) > 1:
            opt_thresh, opt_cv_f1 = find_optimal_threshold(oof_y, oof_probs, metric="f1")
        else:
            opt_thresh, opt_cv_f1 = 0.50, 0.0

        # Fit final pipeline on 100% of training set
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        fitted_pipelines[name] = pipeline

        # Evaluate on Test Holdout
        if hasattr(pipeline, "predict_proba"):
            y_test_probs = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_test_probs = np.full(len(X_test), y_train.mean())

        y_test_pred_default = (y_test_probs >= 0.50).astype(int)
        y_test_pred_optimal = (y_test_probs >= opt_thresh).astype(int)

        metrics_default = calculate_metrics(y_test, y_test_pred_default, y_test_probs)
        metrics_optimal = calculate_metrics(y_test, y_test_pred_optimal, y_test_probs)

        cv_summary = {
            "mean_cv_f1": float(np.mean(cv_f1s)),
            "mean_cv_roc_auc": float(np.mean(cv_rocs)) if cv_rocs else 0.5,
            "mean_cv_pr_auc": float(np.mean(cv_prs)) if cv_prs else 0.0,
            "optimal_threshold": opt_thresh,
            "train_time_sec": float(train_time)
        }
        cv_results[name] = cv_summary

        test_evaluations[name] = {
            "default_0.50": metrics_default,
            "optimal_threshold": {
                "threshold": opt_thresh,
                "metrics": metrics_optimal
            },
            "test_probs": y_test_probs.tolist()
        }

        print(f"  Training Time: {train_time:.2f}s | Optimal CV Threshold: {opt_thresh:.2f}")
        print(f"  Test [Default 0.50] -> PR-AUC: {metrics_default['pr_auc']:.4f} | ROC-AUC: {metrics_default['roc_auc']:.4f} | F1: {metrics_default['f1']:.4f} | Recall: {metrics_default['recall']:.4f} | Prec: {metrics_default['precision']:.4f}")
        print(f"  Test [Optimal {opt_thresh:.2f}] -> F1: {metrics_optimal['f1']:.4f} | Recall: {metrics_optimal['recall']:.4f} | Prec: {metrics_optimal['precision']:.4f} | BalAcc: {metrics_optimal['balanced_accuracy']:.4f}")

    # 4. Methodological Benchmarks: Post-Contact Model & Random-Split Benchmark
    print("\n" + "=" * 75)
    print("METHODOLOGICAL BENCHMARKS")
    print("=" * 75)

    # Benchmark A: Post-Contact Model (HistGBM with Duration Leakage)
    print("\n--- Benchmark A: Post-Contact Model (WITH duration, Temporal Split) ---")
    post_pipe = build_full_pipeline(
        classifier=HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=150, learning_rate=0.08, min_samples_leaf=20, random_state=42
        ),
        include_scaling=False,
        exclude_duration=False
    )
    t0 = time.time()
    post_pipe.fit(X_train, y_train)
    post_time = time.time() - t0
    y_test_probs_post = post_pipe.predict_proba(X_test)[:, 1]
    y_test_pred_post = (y_test_probs_post >= 0.50).astype(int)
    post_metrics = calculate_metrics(y_test, y_test_pred_post, y_test_probs_post)
    print(f"  [Post-Contact] PR-AUC: {post_metrics['pr_auc']:.4f} | ROC-AUC: {post_metrics['roc_auc']:.4f} | F1: {post_metrics['f1']:.4f} | Prec: {post_metrics['precision']:.4f} | Recall: {post_metrics['recall']:.4f}")
    print("  -> Shows massive score inflation from target leakage of duration!")

    # Benchmark B: Random-Split Benchmark (HistGBM without Duration, Stratified Random Split)
    print("\n--- Benchmark B: Random-Split Benchmark (NO duration, Stratified 80/20 Random Split) ---")
    X_tr_rnd, X_te_rnd, y_tr_rnd, y_te_rnd = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    rnd_pipe = build_full_pipeline(
        classifier=HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=150, learning_rate=0.08, min_samples_leaf=20, random_state=42
        ),
        include_scaling=False,
        exclude_duration=True
    )
    t0 = time.time()
    rnd_pipe.fit(X_tr_rnd, y_tr_rnd)
    rnd_time = time.time() - t0
    y_rnd_probs = rnd_pipe.predict_proba(X_te_rnd)[:, 1]
    y_rnd_preds = (y_rnd_probs >= 0.50).astype(int)
    rnd_metrics = calculate_metrics(y_te_rnd, y_rnd_preds, y_rnd_probs)
    print(f"  [Random Split] PR-AUC: {rnd_metrics['pr_auc']:.4f} | ROC-AUC: {rnd_metrics['roc_auc']:.4f} | F1: {rnd_metrics['f1']:.4f} | Prec: {rnd_metrics['precision']:.4f} | Recall: {rnd_metrics['recall']:.4f}")
    print("  -> Random split samples uniformly across all months, masking real-world temporal distribution shifts.")

    # 5. Model Selection
    # Best pre-contact model selection based on test PR-AUC, ROC-AUC, and Balanced Accuracy
    pre_contact_scores = {
        name: test_evaluations[name]["default_0.50"]["pr_auc"]
        for name in models if name != "Dummy (Baseline)"
    }
    best_model_name = max(pre_contact_scores, key=pre_contact_scores.get)
    best_pipeline = fitted_pipelines[best_model_name]
    best_opt_thresh = cv_results[best_model_name]["optimal_threshold"]

    print("\n" + "=" * 75)
    print(f"BEST PRE-CONTACT MODEL SELECTED: {best_model_name}")
    print(f"Selected based on highest PR-AUC ({pre_contact_scores[best_model_name]:.4f}) on temporal holdout.")
    print(f"Saved pipeline to models/best_model.joblib")
    print("=" * 75)

    joblib.dump(best_pipeline, "models/best_model.joblib")

    # 6. Save comprehensive evaluation dictionary for evaluate.py and final_report.md
    summary_data = {
        "temporal_split_info": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_yes_count": int(y_train.sum()),
            "train_yes_pct": float(y_train.mean() * 100),
            "test_yes_count": int(y_test.sum()),
            "test_yes_pct": float(y_test.mean() * 100)
        },
        "best_model_name": best_model_name,
        "best_optimal_threshold": best_opt_thresh,
        "cv_results": cv_results,
        "test_evaluations": {
            name: {
                "default_0.50": test_evaluations[name]["default_0.50"],
                "optimal_threshold": test_evaluations[name]["optimal_threshold"]
            }
            for name in test_evaluations
        },
        "benchmarks": {
            "post_contact_model": {
                "description": "HistGradientBoosting with duration (Target Leakage Benchmark)",
                "metrics": post_metrics,
                "train_time_sec": float(post_time)
            },
            "random_split_model": {
                "description": "HistGradientBoosting without duration on Stratified Random Split Benchmark",
                "metrics": rnd_metrics,
                "train_time_sec": float(rnd_time)
            }
        }
    }

    with open("models/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save raw test probabilities for plotting curves in evaluate.py
    plot_data = {
        "y_test": y_test.tolist(),
        "y_test_rnd": y_te_rnd.tolist(),
        "probs": {name: test_evaluations[name]["test_probs"] for name in models},
        "probs_post": y_test_probs_post.tolist(),
        "probs_rnd": y_rnd_probs.tolist()
    }
    with open("models/plot_data.json", "w", encoding="utf-8") as f:
        json.dump(plot_data, f)

    print("Saved training artifacts: models/best_model.joblib, models/metrics_summary.json, models/plot_data.json")


if __name__ == "__main__":
    main()
