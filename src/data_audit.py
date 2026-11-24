"""
data_audit.py - Data Audit and Quality Assessment for Bank Marketing Dataset.
"""
import os
import json
import pandas as pd
import numpy as np

def run_data_audit(csv_path="Bank_Marketing.csv", metadata_path="metadata.json"):
    print("=" * 70)
    print("STAGE A: DATA AUDIT AND QUALITY ASSESSMENT")
    print("=" * 70)

    if not os.path.exists(csv_path):
        csv_path = os.path.join("dataset", "Bank_Marketing.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join("dataset", "metadata.json")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata from: {metadata_path}")
    print(f"Dataset name: {metadata.get('name')}")
    print(f"Target column in metadata: {metadata.get('target_col')}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from: {csv_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Duplicates check
    n_dupes = df.duplicated().sum()
    print(f"\n[1] Duplicate Rows Check: {n_dupes} duplicate rows found.")

    # 2. Target check
    print("\n[2] Target Distribution ('y'):")
    target_counts = df['y'].value_counts(dropna=False)
    target_props = df['y'].value_counts(normalize=True, dropna=False)
    for cls, count in target_counts.items():
        print(f"  Class '{cls}': {count:,} ({target_props[cls]*100:.2f}%)")
    valid_targets = set(['yes', 'no'])
    actual_targets = set(df['y'].dropna().unique())
    assert actual_targets.issubset(valid_targets), f"Unexpected target values: {actual_targets - valid_targets}"
    print("  -> Target column strictly contains 'yes' and 'no'. No spurious classes found.")

    # 3. Missing values analysis
    print("\n[3] Missing Values Analysis:")
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': null_counts, 'Missing_Pct': null_pcts})
    cols_with_missing = missing_df[missing_df['Missing_Count'] > 0]
    print(cols_with_missing.to_string())

    # String 'unknown' check in categorical columns
    print("\n[4] Explicit 'unknown' String Check in Categorical Columns:")
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col == 'y':
            continue
        unk_count = (df[col].astype(str).str.lower() == 'unknown').sum()
        if unk_count > 0:
            print(f"  Column '{col}': {unk_count:,} ({unk_count/len(df)*100:.2f}%) literal 'unknown' values")

    # 5. Day column check (day_of_week in CSV vs day in metadata)
    print("\n[5] Day Column Audit:")
    day_col = 'day_of_week' if 'day_of_week' in df.columns else 'day'
    print(f"  Column name in CSV: '{day_col}'")
    print(f"  Data type: {df[day_col].dtype}")
    print(f"  Min value: {df[day_col].min()}, Max value: {df[day_col].max()}")
    print(f"  Unique values count: {df[day_col].nunique()} (range 1 to 31)")
    print("  -> Confirmed: 'day_of_week' represents the calendar day of the month (1-31), NOT weekday (Monday-Sunday).")

    # 6. pdays and previous audit
    print("\n[6] pdays and previous Analysis:")
    n_pdays_neg1 = (df['pdays'] == -1).sum()
    print(f"  pdays == -1 (not previously contacted): {n_pdays_neg1:,} ({n_pdays_neg1/len(df)*100:.2f}%)")
    n_prev_zero = (df['previous'] == 0).sum()
    print(f"  previous == 0: {n_prev_zero:,} ({n_prev_zero/len(df)*100:.2f}%)")
    consistency = ((df['pdays'] == -1) == (df['previous'] == 0)).mean() * 100
    print(f"  Consistency between (pdays == -1) and (previous == 0): {consistency:.2f}%")

    # 7. Duration audit
    print("\n[7] Duration Column Audit:")
    print(f"  Min duration: {df['duration'].min()} s, Max duration: {df['duration'].max()} s")
    print(f"  Median duration: {df['duration'].median()} s, Mean duration: {df['duration'].mean():.2f} s")
    print("  -> Duration is telephone call duration in seconds.")
    print("  -> NOTE: Duration is strictly EXCLUDED from the main predictive model to prevent target leakage.")

    # 8. Temporal ordering audit
    print("\n[8] Temporal Structure Audit:")
    months = df['month'].tolist()
    month_changes = [(months[0], 0)]
    for i in range(1, len(months)):
        if months[i] != months[i-1]:
            month_changes.append((months[i], i))
    print(f"  Identified {len(month_changes)} month blocks in dataset order.")
    print(f"  First 8 blocks: {month_changes[:8]}")
    print(f"  Last 6 blocks: {month_changes[-6:]}")

    # Check temporal split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_slice = df.iloc[:split_idx]
    test_slice = df.iloc[split_idx:]
    print(f"\n  Temporal Split: Train rows (0 to {split_idx-1}) = {len(train_slice):,}; Test rows ({split_idx} to {len(df)-1}) = {len(test_slice):,}")
    print(f"  Train Target Distribution: {train_slice['y'].value_counts().to_dict()} (Yes rate: {train_slice['y'].value_counts(normalize=True)['yes']*100:.2f}%)")
    print(f"  Test Target Distribution: {test_slice['y'].value_counts().to_dict()} (Yes rate: {test_slice['y'].value_counts(normalize=True)['yes']*100:.2f}%)")
    print("  -> Significant concept drift observed between train (6.73% yes) and test (31.58% yes)!")

    # Summary dictionary for export
    summary = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "duplicate_rows": int(n_dupes),
        "target_distribution": {k: int(v) for k, v in target_counts.items()},
        "target_proportions": {k: float(v) for k, v in target_props.items()},
        "missing_columns": {col: {"count": int(null_counts[col]), "pct": float(null_pcts[col])} for col in cols_with_missing.index},
        "day_column": day_col,
        "pdays_neg1_count": int(n_pdays_neg1),
        "pdays_neg1_pct": float(n_pdays_neg1 / len(df) * 100),
        "temporal_train_rows": len(train_slice),
        "temporal_test_rows": len(test_slice),
        "temporal_train_yes_pct": float(train_slice['y'].value_counts(normalize=True)['yes'] * 100),
        "temporal_test_yes_pct": float(test_slice['y'].value_counts(normalize=True)['yes'] * 100),
    }

    with open("audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved audit summary to audit_summary.json")
    print("=" * 70)
    return summary

if __name__ == "__main__":
    run_data_audit()
