"""
evaluate.py - Comprehensive Model Evaluation, Visualization Generation, and Feature Importance.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score
)
from sklearn.inspection import permutation_importance


def main():
    print("=" * 75)
    print("STAGE E & F: EVALUATION, VISUALIZATION, AND INTERPRETATION")
    print("=" * 75)

    os.makedirs("figures", exist_ok=True)

    with open("models/metrics_summary.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    with open("models/plot_data.json", "r", encoding="utf-8") as f:
        plot_data = json.load(f)

    best_pipe = joblib.load("models/best_model.joblib")
    best_model_name = metrics_data["best_model_name"]
    opt_thresh = metrics_data["best_optimal_threshold"]

    y_test = np.array(plot_data["y_test"])
    probs = {name: np.array(p) for name, p in plot_data["probs"].items()}
    y_test_rnd = np.array(plot_data["y_test_rnd"])
    probs_rnd = np.array(plot_data["probs_rnd"])
    probs_post = np.array(plot_data["probs_post"])

    print(f"Loaded evaluation data for Best Model: {best_model_name} (Optimal Threshold: {opt_thresh:.2f})")

    # -------------------------------------------------------------
    # 8. Model Comparison Plot
    # -------------------------------------------------------------
    print("Generating figures/model_comparison.png...")
    model_names = [m for m in metrics_data["test_evaluations"].keys() if m != "Dummy (Baseline)"]
    metrics_to_plot = ['pr_auc', 'roc_auc', 'balanced_accuracy', 'f1', 'recall', 'precision']
    metric_labels = ['PR-AUC', 'ROC-AUC', 'Balanced Acc', 'F1-Score', 'Recall', 'Precision']

    comp_rows = []
    for m_name in model_names:
        m_vals = metrics_data["test_evaluations"][m_name]["default_0.50"]
        for k, lbl in zip(metrics_to_plot, metric_labels):
            comp_rows.append({
                'Model': m_name,
                'Metric': lbl,
                'Score': m_vals[k]
            })
    comp_df = pd.DataFrame(comp_rows)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.barplot(data=comp_df, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax, edgecolor='black', linewidth=0.6)
    ax.set_title("Pre-Contact Model Comparison Across Key Performance Metrics (Threshold = 0.50)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", loc='upper right', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bars
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.015),
                        ha='center', va='bottom', fontsize=7, rotation=0, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 9. Confusion Matrix for Best Model (Default vs Optimal Threshold)
    # -------------------------------------------------------------
    print("Generating figures/confusion_matrix_best_model.png...")
    best_probs = probs[best_model_name]
    cm_default = confusion_matrix(y_test, (best_probs >= 0.50).astype(int))
    cm_optimal = confusion_matrix(y_test, (best_probs >= opt_thresh).astype(int))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    labels = ["No", "Yes"]
    sns.heatmap(cm_default, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=labels, yticklabels=labels, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f"{best_model_name} (Threshold = 0.50)\nRecall: {metrics_data['test_evaluations'][best_model_name]['default_0.50']['recall']:.2f} | Prec: {metrics_data['test_evaluations'][best_model_name]['default_0.50']['precision']:.2f} | F1: {metrics_data['test_evaluations'][best_model_name]['default_0.50']['f1']:.2f}",
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel("Predicted Label", fontsize=10)
    ax1.set_ylabel("True Label", fontsize=10)

    sns.heatmap(cm_optimal, annot=True, fmt=',d', cmap='Greens', cbar=False, ax=ax2,
                xticklabels=labels, yticklabels=labels, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title(f"{best_model_name} (Optimal CV Threshold = {opt_thresh:.2f})\nRecall: {metrics_data['test_evaluations'][best_model_name]['optimal_threshold']['metrics']['recall']:.2f} | Prec: {metrics_data['test_evaluations'][best_model_name]['optimal_threshold']['metrics']['precision']:.2f} | F1: {metrics_data['test_evaluations'][best_model_name]['optimal_threshold']['metrics']['f1']:.2f}",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Predicted Label", fontsize=10)
    ax2.set_ylabel("True Label", fontsize=10)

    fig.suptitle(f"Confusion Matrix Comparison for Best Model: {best_model_name}", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figures/confusion_matrix_best_model.png", dpi=200, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 10. ROC Curve Comparison (Including Benchmarks)
    # -------------------------------------------------------------
    print("Generating figures/roc_curve_comparison.png...")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

    colors = {'Logistic Regression': '#E67E22', 'Random Forest': '#2ECC71', 'HistGradientBoosting': '#3498DB', 'Extra Trees': '#9B59B6'}
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc_val = roc_auc_score(y_test, probs[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, '#333333'), linewidth=2)

    # Post contact benchmark
    fpr_post, tpr_post, _ = roc_curve(y_test, probs_post)
    auc_post = roc_auc_score(y_test, probs_post)
    ax.plot(fpr_post, tpr_post, label=f"Post-Contact Model (WITH duration, AUC = {auc_post:.3f}) [LEAKAGE]", color='#E74C3C', linestyle='--', linewidth=2.5)

    # Random split benchmark
    fpr_rnd, tpr_rnd, _ = roc_curve(y_test_rnd, probs_rnd)
    auc_rnd = roc_auc_score(y_test_rnd, probs_rnd)
    ax.plot(fpr_rnd, tpr_rnd, label=f"Random Split Benchmark (AUC = {auc_rnd:.3f}) [NO DRIFT]", color='#1ABC9C', linestyle=':', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.500)')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/roc_curve_comparison.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 11. Precision-Recall Curve Comparison
    # -------------------------------------------------------------
    print("Generating figures/precision_recall_curve.png...")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    test_prevalence = y_test.mean()

    for name in model_names:
        p_curve, r_curve, _ = precision_recall_curve(y_test, probs[name])
        pr_auc = average_precision_score(y_test, probs[name])
        ax.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=colors.get(name, '#333333'), linewidth=2)

    # Post contact benchmark
    p_post, r_post, _ = precision_recall_curve(y_test, probs_post)
    pr_post = average_precision_score(y_test, probs_post)
    ax.plot(r_post, p_post, label=f"Post-Contact Model (WITH duration, PR-AUC = {pr_post:.3f}) [LEAKAGE]", color='#E74C3C', linestyle='--', linewidth=2.5)

    ax.axhline(test_prevalence, color='black', linestyle='--', alpha=0.6, label=f"No-Skill Baseline ({test_prevalence*100:.1f}%)")
    ax.set_title("Precision-Recall Curve Comparison on Temporal Holdout Test Set", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/precision_recall_curve.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 12. Threshold Trade-off Curve for Best Model
    # -------------------------------------------------------------
    print("Generating figures/threshold_tradeoff.png...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    th_range = np.linspace(0.05, 0.95, 91)
    p_list, r_list, f1_list, f2_list = [], [], [], []

    for th in th_range:
        preds = (best_probs >= th).astype(int)
        p_list.append(precision_score(y_test, preds, zero_division=0))
        r_list.append(recall_score(y_test, preds, zero_division=0))
        f1_list.append(f1_score(y_test, preds, zero_division=0))
        f2_list.append(fbeta_score(y_test, preds, beta=2, zero_division=0))

    ax.plot(th_range, p_list, label="Precision", color='#E74C3C', linewidth=2)
    ax.plot(th_range, r_list, label="Recall", color='#3498DB', linewidth=2)
    ax.plot(th_range, f1_list, label="F1-Score", color='#2ECC71', linewidth=2.5)
    ax.plot(th_range, f2_list, label="F2-Score (Recall-Biased)", color='#9B59B6', linewidth=2, linestyle=':')

    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.8, label='Default Threshold (0.50)')
    ax.axvline(opt_thresh, color='darkgreen', linestyle='-', linewidth=2, label=f'Optimal CV Threshold ({opt_thresh:.2f})')

    ax.set_title(f"Decision Threshold Trade-Off Analysis for Best Model: {best_model_name}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Decision Probability Threshold", fontsize=11)
    ax.set_ylabel("Metric Value (0.0 to 1.0)", fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center right", fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/threshold_tradeoff.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 13. Permutation Feature Importance for Best Model
    # -------------------------------------------------------------
    print("Generating figures/feature_importance.png (Permutation Importance on Test Set)...")
    csv_path = "Bank_Marketing.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join("dataset", "Bank_Marketing.csv")
    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * 0.8)
    X_test_df = df.drop(columns=['y']).iloc[split_idx:]
    y_test_s = (df['y'] == 'yes').astype(int).iloc[split_idx:]

    # Compute permutation importance (on raw feature level)
    # Using roc_auc as scoring
    perm_result = permutation_importance(
        best_pipe, X_test_df, y_test_s,
        scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1
    )

    feature_names = X_test_df.columns.tolist()
    # If duration in df, remove duration from list since it was dropped in transformer
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': perm_result.importances_mean,
        'Importance_Std': perm_result.importances_std
    }).sort_values(by='Importance_Mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    bars = ax.barh(perm_df['Feature'], perm_df['Importance_Mean'], xerr=perm_df['Importance_Std'],
                   color='#2980B9', edgecolor='black', linewidth=0.6, capsize=3)
    ax.set_title(f"Permutation Feature Importance on Temporal Holdout Test Set\n(Model: {best_model_name}, Metric: ROC-AUC Decrease)",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Mean Decrease in ROC-AUC When Permuted", fontsize=11)
    ax.set_ylabel("Raw Feature Name", fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar, mean_val in zip(bars, perm_df['Importance_Mean']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{mean_val:.4f}",
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=200)
    plt.close()

    # Save feature importances to json
    perm_df.to_json("models/permutation_importance.json", orient='records', indent=2)

    print("All 6 evaluation figures successfully generated in figures/!")
    print("=" * 75)


if __name__ == "__main__":
    main()
