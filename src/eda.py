"""
eda.py - Exploratory Data Analysis & Visualizations for Bank Marketing Dataset.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("Bank_Marketing.csv")

# 1. Target Distribution
print("Generating figures/target_distribution.png...")
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
counts = df['y'].value_counts()
props = df['y'].value_counts(normalize=True) * 100
colors = ['#4A90E2', '#50E3C2']
bars = ax.bar(['No (Did not subscribe)', 'Yes (Subscribed)'], counts.values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

for bar, count, prop in zip(bars, counts.values, props.values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 700, f"{count:,}\n({prop:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Term Deposit Subscription Target Distribution ('y')", fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("Number of Clients", fontsize=11)
ax.set_ylim(0, 46000)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/target_distribution.png", dpi=200)
plt.close()

# 2. Missing Values
print("Generating figures/missing_values.png...")
fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
null_counts = df.isnull().sum()
null_pcts = (null_counts / len(df)) * 100
missing_data = pd.DataFrame({'Feature': df.columns, 'Missing_Pct': null_pcts, 'Missing_Count': null_counts})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(by='Missing_Pct', ascending=True)

bars = ax.barh(missing_data['Feature'], missing_data['Missing_Pct'], color='#E74C3C', edgecolor='black', linewidth=0.8)
for bar, count, pct in zip(bars, missing_data['Missing_Count'], missing_data['Missing_Pct']):
    ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{pct:.2f}% ({count:,} rows)", va='center', fontsize=10, fontweight='bold')

ax.set_title("Missing Values by Feature in Bank Marketing Dataset", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Missing Percentage (%)", fontsize=11)
ax.set_xlim(0, 100)
ax.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/missing_values.png", dpi=200)
plt.close()

# 3. Numeric Distributions
print("Generating figures/numeric_distributions.png...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=200)
num_cols = ['age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous']
titles = ['Age (Years)', 'Balance (Euros)', 'Day of Month (1-31)', 'Campaign Contacts', 'Pdays (Days Since Prior Contact)', 'Previous Contacts']

for ax, col, title in zip(axes.flatten(), num_cols, titles):
    data = df[col]
    if col == 'balance':
        # Clip extreme tails for visualization visibility
        data = np.clip(data, -2000, 15000)
    elif col == 'campaign':
        data = np.clip(data, 1, 15)
    elif col == 'previous':
        data = np.clip(data, 0, 15)
    
    sns.histplot(data, bins=30, ax=ax, kde=(col in ['age', 'day_of_week']), color='#3498DB', edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlabel(col, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("Distributions of Key Numeric Features", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/numeric_distributions.png", dpi=200, bbox_inches='tight')
plt.close()

# 4. Categorical Subscription Rate
print("Generating figures/categorical_subscription_rate.png...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
cat_cols = ['job', 'education', 'month', 'contact']
overall_rate = (df['y'] == 'yes').mean() * 100

for ax, col in zip(axes.flatten(), cat_cols):
    temp = df.copy()
    temp[col] = temp[col].fillna('Unknown')
    grouped = temp.groupby(col)['y'].apply(lambda s: (s == 'yes').mean() * 100).sort_values(ascending=False)
    counts_grp = temp.groupby(col).size()[grouped.index]
    
    bars = ax.bar(grouped.index, grouped.values, color='#2ECC71', edgecolor='black', linewidth=0.6)
    ax.axhline(overall_rate, color='red', linestyle='--', linewidth=1.2, label=f'Baseline ({overall_rate:.1f}%)')
    
    ax.set_title(f"Subscription Rate by {col.capitalize()}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Subscription Rate (%)", fontsize=10)
    ax.set_ylim(0, max(grouped.values) * 1.25)
    ax.tick_params(axis='x', rotation=45 if col in ['job', 'month'] else 0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle("Subscription Rate (%) across Categorical Attributes", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/categorical_subscription_rate.png", dpi=200, bbox_inches='tight')
plt.close()

# 5. Numeric by Target
print("Generating figures/numeric_by_target.png...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
box_cols = [('age', 'Age (Years)'), ('balance', 'Balance (€, clipped -1k to 10k)'), ('campaign', 'Campaign Contacts (clipped to 10)'), ('previous', 'Previous Contacts (clipped to 8)')]

for ax, (col, lbl) in zip(axes.flatten(), box_cols):
    plot_df = df[['y', col]].copy()
    if col == 'balance':
        plot_df[col] = np.clip(plot_df[col], -1000, 10000)
    elif col == 'campaign':
        plot_df[col] = np.clip(plot_df[col], 1, 10)
    elif col == 'previous':
        plot_df[col] = np.clip(plot_df[col], 0, 8)
        
    sns.boxplot(data=plot_df, x='y', y=col, ax=ax, palette=['#E74C3C', '#2ECC71'], width=0.4, boxprops=dict(alpha=0.8))
    ax.set_title(f"{lbl} by Subscription Status", fontsize=11, fontweight='bold')
    ax.set_xlabel("Subscribed ('y')", fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("Comparison of Numeric Features by Subscription Outcome", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/numeric_by_target.png", dpi=200, bbox_inches='tight')
plt.close()

# 6. Numeric Correlation Heatmap
print("Generating figures/numeric_correlation.png...")
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
num_df = df[['age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous']].copy()
# Note: Duration excluded from primary modeling correlation, but we can compute correlation for pre-contact features
corr = num_df.corr(method='spearman')

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Spearman Rank Correlation Heatmap of Numeric Features", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("figures/numeric_correlation.png", dpi=200)
plt.close()

# 7. Temporal Target Trend
print("Generating figures/temporal_target_trend.png...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=200, sharex=False)

# Rolling window of target along dataset row order
df['target_numeric'] = (df['y'] == 'yes').astype(int)
window_size = 1000
rolling_rate = df['target_numeric'].rolling(window=window_size, min_periods=100).mean() * 100

ax1.plot(df.index, rolling_rate, color='#8E44AD', linewidth=1.5, label=f'Rolling Subscription Rate (Window={window_size})')
split_idx = int(len(df) * 0.8)
ax1.axvline(split_idx, color='red', linestyle='--', linewidth=2, label=f'80/20 Temporal Split (Row {split_idx:,})')
ax1.axhline(6.73, color='blue', linestyle=':', label='Train Period Mean (6.73%)')
ax1.axhline(31.58, color='green', linestyle=':', label='Test Period Mean (31.58%)')

ax1.set_title("Rolling Subscription Rate Across Chronological Dataset Rows", fontsize=12, fontweight='bold')
ax1.set_ylabel("Subscription Rate (%)", fontsize=10)
ax1.set_xlabel("Dataset Row Index (Chronological Order: May 2008 -> Nov 2010)", fontsize=10)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(0, 50)

# Monthly sequence subscription rates
months_seq = []
curr_m = df['month'].iloc[0]
start_i = 0
for i, m in enumerate(df['month']):
    if m != curr_m:
        slice_df = df.iloc[start_i:i]
        months_seq.append({
            'month': curr_m,
            'start_idx': start_i,
            'end_idx': i,
            'rate': (slice_df['y'] == 'yes').mean() * 100,
            'count': len(slice_df)
        })
        curr_m = m
        start_i = i
slice_df = df.iloc[start_i:]
months_seq.append({
    'month': curr_m,
    'start_idx': start_i,
    'end_idx': len(df),
    'rate': (slice_df['y'] == 'yes').mean() * 100,
    'count': len(slice_df)
})

seq_df = pd.DataFrame(months_seq)
colors = ['#3498DB' if row['end_idx'] <= split_idx else '#E67E22' for _, row in seq_df.iterrows()]
bars = ax2.bar(range(len(seq_df)), seq_df['rate'], color=colors, edgecolor='black', linewidth=0.5)
ax2.axvline(18.5, color='red', linestyle='--', linewidth=2, label='Train / Holdout Boundary (~Row 36,168)')
ax2.set_xticks(range(len(seq_df)))
ax2.set_xticklabels([f"{r['month']}\n(n={r['count']})" for _, r in seq_df.iterrows()], rotation=60, fontsize=8)
ax2.set_title("Sequential Campaign Batches: Subscription Rate by Month Block", fontsize=12, fontweight='bold')
ax2.set_ylabel("Subscription Rate (%)", fontsize=10)
ax2.set_xlabel("Chronological Campaign Blocks (Blue = Train Set 80%, Orange = Temporal Holdout Test 20%)", fontsize=10)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/temporal_target_trend.png", dpi=200)
plt.close()

print("All 7 EDA figures successfully generated!")
