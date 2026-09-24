"""
Automated Time-Series Diagram Generator for BD-FPP Models
CSE 4112: Machine Learning Laboratory Final Project

Generates Price vs. Time evaluation diagrams comparing Actual Ground Truth vs. Predicted
prices with explicit Train Period (80%) vs. Test Period (20%) boundaries on Temporal split data.

Usage:
    python Evaluation/temporal/generate_timeseries_plots.py
"""

import os
import io
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# Set up paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
BUFFER_DIR = os.path.join(PROJECT_DIR, 'Buffer', 'temporal')
PREPROC_DIR = os.path.join(PROJECT_DIR, 'Preprocessing')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'timeseries_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_MAP = {
    'rf': {'name': 'Random Forest', 'color': '#2ca02c'},
    'xgbr': {'name': 'XGBoost', 'color': '#9467bd'},
    'gbr': {'name': 'Gradient Boosting', 'color': '#d62728'},
    'lr': {'name': 'Linear Regression', 'color': '#1f77b4'},
    'm5p': {'name': 'M5P Model Tree', 'color': '#ff7f0e'}
}

def parse_buffer_file(path):
    """Extracts predictions CSV table from raw Weka run logs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Buffer file not found: {path}")
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('inst#,actual,predicted,error'):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Could not find predictions header in {path}")
    data_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('==='):
            break
        data_lines.append(line)
    df = pd.read_csv(io.StringIO(''.join(data_lines)))
    df.columns = df.columns.str.strip()
    df['commodity_name'] = df['commodity_name'].astype(str).str.strip().str.strip("'")
    return df

# Load reference datasets
train_ref = pd.read_csv(os.path.join(PREPROC_DIR, 'moa_train_80_lag.csv'))
test_ref = pd.read_csv(os.path.join(PREPROC_DIR, 'moa_test_20_lag.csv'))
train_ref['date'] = pd.to_datetime(train_ref[['year', 'month', 'day']])
test_ref['date'] = pd.to_datetime(test_ref[['year', 'month', 'day']])
SPLIT_DATE = test_ref['date'].min()

def get_model_timeseries_df(model_key, horizon=7):
    train_folder = f"target_{horizon}" if horizon != 30 else "target-30"
    test_folder = f"target_{horizon}"
    train_path = os.path.join(BUFFER_DIR, 'train', train_folder, f"{model_key}_buffer_train_{horizon}")
    test_path = os.path.join(BUFFER_DIR, 'test', test_folder, f"{model_key}_buffer_{horizon}")
    train_buf = parse_buffer_file(train_path)
    test_buf = parse_buffer_file(test_path)
    target_col = f"target_{horizon}d"
    df_train = train_ref.copy()
    df_train['predicted'] = train_buf['predicted']
    df_train['actual_target'] = df_train[target_col]
    df_train['split'] = 'Train'
    df_test = test_ref.copy()
    df_test['predicted'] = test_buf['predicted']
    df_test['actual_target'] = df_test[target_col]
    df_test['split'] = 'Test'
    return pd.concat([df_train, df_test], ignore_index=True)

def plot_commodity_timeseries(model_key='rf', horizon=7, commodity='Rice - Medium', division=None, save=True):
    model_info = MODEL_MAP[model_key]
    model_name = model_info['name']
    model_color = model_info['color']
    full_df = get_model_timeseries_df(model_key, horizon)
    sub = full_df[full_df['commodity_name'] == commodity].copy()
    if sub.empty: return None
    unit = sub['retail_unit'].iloc[0]
    
    if division is not None:
        sub = sub[sub['division'] == division]
        title_region = f"[{division} Division]"
    else:
        title_region = "[National Average across Divisions]"
        
    daily = sub.groupby(['date', 'split'], as_index=False).agg({
        'actual_target': 'mean',
        'predicted': 'mean'
    }).sort_values('date')
    
    test_data = daily[daily['split'] == 'Test']
    if len(test_data) > 1:
        act_te = test_data['actual_target'].values
        pred_te = test_data['predicted'].values
        rmse = np.sqrt(np.mean((act_te - pred_te) ** 2))
        r_val, _ = pearsonr(act_te, pred_te) if np.std(act_te) > 0 and np.std(pred_te) > 0 else (np.nan, 0)
        mape = np.mean(np.abs((act_te - pred_te) / act_te)) * 100
        metric_str = f"Test Metrics:\nPearson $R$: {r_val:.4f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
    else:
        metric_str = "Test Metrics: N/A"
        
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    ax.plot(daily['date'], daily['actual_target'], label='Actual Price (Ground Truth)', 
            color='#1f77b4', linewidth=2.0, alpha=0.9)
    ax.plot(daily['date'], daily['predicted'], label=f'Predicted Price ({model_name})', 
            color=model_color, linewidth=1.8, linestyle='--', alpha=0.95)
    
    min_date = daily['date'].min()
    max_date = daily['date'].max()
    ax.axvspan(min_date, SPLIT_DATE, color='#2ca02c', alpha=0.10, label='Train Period (Earliest 80% Chronological)')
    ax.axvspan(SPLIT_DATE, max_date, color='#ff7f0e', alpha=0.12, label='Test Period (Recent 20% Out-of-Sample)')
    ax.axvline(SPLIT_DATE, color='#333333', linestyle=':', linewidth=1.6, 
               label=f'Temporal Split Boundary ({SPLIT_DATE.strftime("%Y-%m-%d")})')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bbbbbb')
    ax.text(0.985, 0.05, metric_str, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax.set_title(f'Time-Series Forecasting: Price vs. Time\n{commodity} ({unit}) -- {model_name} ({horizon}-Day Forecast Horizon) {title_region}',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Time (Calendar Date)', fontsize=10.5, labelpad=8)
    ax.set_ylabel(f'Price (BDT / {unit})', fontsize=10.5, labelpad=8)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    fig.tight_layout()
    
    if save:
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', commodity)
        save_folder = os.path.join(OUTPUT_DIR, model_key, f"target_{horizon}")
        os.makedirs(save_folder, exist_ok=True)
        filename = f"{model_key}_{clean_name}_{horizon}d.png"
        out_file = os.path.join(save_folder, filename)
        fig.savefig(out_file, bbox_inches='tight')
        print(f"Saved: {out_file}")
    plt.close(fig)

def generate_all_staples():
    staples = [
        'Rice - Medium',
        'Ata (Packet)',
        'Broiler chicken',
        'Egg Farm-Red',
        'Soybean Oil(loose)',
        'Onion (local)'
    ]
    print("\n--- 1. Generating Representative Staple Diagrams for Winning Model (Random Forest) ---")
    for s in staples:
        plot_commodity_timeseries('rf', 7, s)

def generate_cross_model_comparison(commodity='Rice - Medium', horizon=7):
    print(f"\n--- 2. Generating Cross-Model 5-Panel Comparison for {commodity} ---")
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True, dpi=140)
    for idx, (m_key, m_info) in enumerate(MODEL_MAP.items()):
        ax = axes[idx]
        full_df = get_model_timeseries_df(m_key, horizon)
        sub = full_df[full_df['commodity_name'] == commodity]
        unit = sub['retail_unit'].iloc[0]
        daily = sub.groupby(['date', 'split'], as_index=False).agg({
            'actual_target': 'mean',
            'predicted': 'mean'
        }).sort_values('date')
        test_data = daily[daily['split'] == 'Test']
        r_val, _ = pearsonr(test_data['actual_target'], test_data['predicted'])
        rmse = np.sqrt(np.mean((test_data['actual_target'] - test_data['predicted']) ** 2))
        ax.plot(daily['date'], daily['actual_target'], label='Actual Price', color='#1f77b4', linewidth=1.8)
        ax.plot(daily['date'], daily['predicted'], label=f'{m_info["name"]}', color=m_info['color'], 
                linewidth=1.6, linestyle='--')
        ax.axvspan(daily['date'].min(), SPLIT_DATE, color='#2ca02c', alpha=0.08)
        ax.axvspan(SPLIT_DATE, daily['date'].max(), color='#ff7f0e', alpha=0.10)
        ax.axvline(SPLIT_DATE, color='#333333', linestyle=':', linewidth=1.2)
        ax.set_ylabel(f'BDT / {unit}', fontsize=9.5)
        ax.set_title(f'{m_info["name"]} (Test $R = {r_val:.3f}$, RMSE = {rmse:.2f})', 
                     fontsize=10.5, fontweight='bold', loc='left')
        ax.legend(loc='upper left', fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time (Calendar Date)', fontsize=10.5, labelpad=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    fig.suptitle(f'Cross-Model Trajectory Comparison: {commodity} ({horizon}-Day Target Horizon)',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', commodity)
    cross_out = os.path.join(OUTPUT_DIR, f"cross_model_comparison_{clean_name}_{horizon}d.png")
    fig.savefig(cross_out, bbox_inches='tight')
    plt.close(fig)
    print("Saved cross-model grid to:", cross_out)

def generate_multi_horizon_comparison(model_key='rf', commodity='Soybean Oil(loose)'):
    print(f"\n--- 3. Generating Multi-Horizon Comparison for {commodity} ---")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, dpi=140)
    for idx, h in enumerate([7, 14, 30]):
        ax = axes[idx]
        full_df = get_model_timeseries_df(model_key, h)
        sub = full_df[full_df['commodity_name'] == commodity]
        unit = sub['retail_unit'].iloc[0]
        daily = sub.groupby(['date', 'split'], as_index=False).agg({
            'actual_target': 'mean',
            'predicted': 'mean'
        }).sort_values('date')
        test_data = daily[daily['split'] == 'Test']
        r_val, _ = pearsonr(test_data['actual_target'], test_data['predicted'])
        rmse = np.sqrt(np.mean((test_data['actual_target'] - test_data['predicted']) ** 2))
        ax.plot(daily['date'], daily['actual_target'], label='Actual Price', color='#1f77b4', linewidth=1.8)
        ax.plot(daily['date'], daily['predicted'], label=f'Predicted ({h}-day horizon)', color='#2ca02c', 
                linewidth=1.6, linestyle='--')
        ax.axvspan(daily['date'].min(), SPLIT_DATE, color='#2ca02c', alpha=0.08)
        ax.axvspan(SPLIT_DATE, daily['date'].max(), color='#ff7f0e', alpha=0.10)
        ax.axvline(SPLIT_DATE, color='#333333', linestyle=':', linewidth=1.2)
        ax.set_ylabel(f'BDT / {unit}', fontsize=9.5)
        ax.set_title(f'{h}-Day Forecast Horizon (Test $R = {r_val:.3f}$, RMSE = {rmse:.2f})', 
                     fontsize=10.5, fontweight='bold', loc='left')
        ax.legend(loc='upper left', fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time (Calendar Date)', fontsize=10.5, labelpad=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    fig.suptitle(f'Random Forest: Multi-Horizon Forecasting Comparison on {commodity}',
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', commodity)
    multi_out = os.path.join(OUTPUT_DIR, f"{model_key}_multi_horizon_{clean_name}.png")
    fig.savefig(multi_out, bbox_inches='tight')
    plt.close(fig)
    print("Saved multi-horizon grid to:", multi_out)

if __name__ == '__main__':
    generate_all_staples()
    generate_cross_model_comparison('Rice - Medium', horizon=7)
    generate_multi_horizon_comparison('rf', 'Soybean Oil(loose)')
    print("\nAll default time-series diagrams successfully generated!")

