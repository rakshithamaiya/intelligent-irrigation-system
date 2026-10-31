"""
Intelligent Irrigation System Using Fuzzy Logic
Evaluation, Baseline Comparison, and Optional ANN Model Comparison

This module provides:
1. Evaluation of the Mamdani Fuzzy System against the synthetic reference benchmark.
2. Comparison against a Traditional Fixed-Threshold Irrigation Baseline.
3. Simulated Water Savings and over/under-watering analysis.
4. Optional simple Artificial Neural Network (ANN) comparative regression model.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_threshold_baseline(moisture: np.ndarray, temp: np.ndarray, hum: np.ndarray) -> np.ndarray:
    """
    Conventional fixed-threshold timer controller (bang-bang irrigation controller):
    - Moisture < 50%: Fixed 85% pumping cycle regardless of weather
    - Moisture >= 50%: 0% (Off)
    Lacks weather awareness; wastes water during cool or humid conditions and
    causes thermal shock during scorching afternoons.
    """
    baseline = np.where(moisture < 50.0, 85.0, 0.0)
    return baseline


def evaluate_irrigation_systems(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive performance evaluation of Fuzzy Logic vs Benchmark and Threshold Baseline.
    """
    y_ref = df['reference_intensity'].values
    y_fuzzy = df['fuzzy_intensity'].values
    m_arr = df['soil_moisture'].values
    t_arr = df['temperature'].values
    h_arr = df['humidity'].values

    # 1. Fuzzy Logic Evaluation
    fuzzy_mae = float(mean_absolute_error(y_ref, y_fuzzy))
    fuzzy_rmse = float(np.sqrt(mean_squared_error(y_ref, y_fuzzy)))
    fuzzy_r2 = float(r2_score(y_ref, y_fuzzy))

    cat_accuracy = float(accuracy_score(df['reference_category'], df['fuzzy_category']))
    cat_macro_f1 = float(f1_score(df['reference_category'], df['fuzzy_category'], average='macro'))

    labels = ['Off', 'Low', 'Medium', 'High']
    cm = confusion_matrix(df['reference_category'], df['fuzzy_category'], labels=labels)

    # 2. Threshold Baseline Evaluation
    y_baseline = run_threshold_baseline(m_arr, t_arr, h_arr)
    base_mae = float(mean_absolute_error(y_ref, y_baseline))
    base_rmse = float(np.sqrt(mean_squared_error(y_ref, y_baseline)))
    base_r2 = float(r2_score(y_ref, y_baseline))

    # Baseline Categories
    base_cats = []
    for b in y_baseline:
        if b < 20.0:
            base_cats.append('Off')
        elif b < 45.0:
            base_cats.append('Low')
        elif b < 70.0:
            base_cats.append('Medium')
        else:
            base_cats.append('High')
    base_accuracy = float(accuracy_score(df['reference_category'], base_cats))

    # 3. Simulated Water Management & Water Savings Analysis
    sim_fuzzy_water = float(np.sum(y_fuzzy))
    sim_base_water = float(np.sum(y_baseline))
    sim_ref_water = float(np.sum(y_ref))

    water_diff_pct = float(((sim_base_water - sim_fuzzy_water) / sim_base_water) * 100.0)

    # Over-watering episodes: Irrigating significantly (>25%) when soil is already wet (>=55%)
    fuzzy_overwater = int(np.sum((m_arr >= 55.0) & (y_fuzzy > 25.0)))
    base_overwater = int(np.sum((m_arr >= 55.0) & (y_baseline > 25.0)))

    # Under-watering episodes: Irrigating < 20% when soil is dry (<30%) and temperature is hot (>30°C)
    fuzzy_underwater = int(np.sum((m_arr < 30.0) & (t_arr > 30.0) & (y_fuzzy < 20.0)))
    base_underwater = int(np.sum((m_arr < 30.0) & (t_arr > 30.0) & (y_baseline < 20.0)))

    # 4. Optional ANN Model (MLP Regressor)
    ann_results = None
    try:
        X = df[['soil_moisture', 'temperature', 'humidity']].values
        y = y_ref

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True
        )
        mlp.fit(X_train_scaled, y_train)
        y_ann_pred = mlp.predict(X_test_scaled)
        y_ann_pred = np.clip(y_ann_pred, 0.0, 100.0)

        ann_mae = float(mean_absolute_error(y_test, y_ann_pred))
        ann_rmse = float(np.sqrt(mean_squared_error(y_test, y_ann_pred)))
        ann_r2 = float(r2_score(y_test, y_ann_pred))

        ann_results = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_mae': round(ann_mae, 2),
            'test_rmse': round(ann_rmse, 2),
            'test_r2': round(ann_r2, 3),
            'architecture': 'MLPRegressor (3 -> 16 -> 8 -> 1)'
        }
    except Exception as err:
        ann_results = {'error': str(err)}

    results = {
        'fuzzy': {
            'mae': round(fuzzy_mae, 2),
            'rmse': round(fuzzy_rmse, 2),
            'r2': round(fuzzy_r2, 3),
            'accuracy': round(cat_accuracy * 100.0, 1),
            'macro_f1': round(cat_macro_f1, 3),
            'confusion_matrix': cm,
            'labels': labels,
            'simulated_water_volume': round(sim_fuzzy_water, 1),
            'overwater_episodes': fuzzy_overwater,
            'underwater_episodes': fuzzy_underwater,
        },
        'baseline': {
            'name': 'Fixed-Threshold Controller (Moisture < 50%)',
            'mae': round(base_mae, 2),
            'rmse': round(base_rmse, 2),
            'r2': round(base_r2, 3),
            'accuracy': round(base_accuracy * 100.0, 1),
            'simulated_water_volume': round(sim_base_water, 1),
            'overwater_episodes': base_overwater,
            'underwater_episodes': base_underwater,
        },
        'water_savings': {
            'simulated_savings_pct': round(water_diff_pct, 1),
            'reference_volume': round(sim_ref_water, 1)
        },
        'ann_comparison': ann_results
    }

    return results


def plot_evaluation_charts(df: pd.DataFrame, eval_results: Dict[str, Any]) -> Tuple[plt.Figure, plt.Figure]:
    """Create publication-quality evaluation charts for Streamlit and project reports."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: Predicted vs Reference & Error Distribution
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig1.patch.set_facecolor('#0f172a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

    # Scatter
    y_ref = df['reference_intensity']
    y_fuzzy = df['fuzzy_intensity']
    sc = ax1.scatter(y_ref, y_fuzzy, c=df['soil_moisture'], cmap='viridis', alpha=0.6, s=25, edgecolors='none')
    cbar = fig1.colorbar(sc, ax=ax1)
    cbar.set_label('Soil Moisture (%)', color='#94a3b8', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')

    ax1.plot([0, 100], [0, 100], '--', color='#f43f5e', lw=2, label='Ideal 1:1 Identity')
    ax1.set_title(f"Fuzzy vs Reference Intensity (R² = {eval_results['fuzzy']['r2']:.2f})", fontsize=11, fontweight='bold', color='#f8fafc')
    ax1.set_xlabel('Reference Benchmark Intensity (%)', color='#94a3b8', fontsize=9.5)
    ax1.set_ylabel('Fuzzy Predicted Intensity (%)', color='#94a3b8', fontsize=9.5)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)

    # Error Histogram
    errors = y_fuzzy - y_ref
    ax2.hist(errors, bins=30, color='#38bdf8', edgecolor='#0f172a', alpha=0.85)
    ax2.axvline(0, color='#f43f5e', linestyle='--', lw=2, label='Zero Error')
    ax2.axvline(np.mean(errors), color='#fbbf24', linestyle='-', lw=1.8, label=f'Mean Error = {np.mean(errors):.2f}%')
    ax2.set_title(f"Prediction Error Distribution (MAE = {eval_results['fuzzy']['mae']:.2f}%)", fontsize=11, fontweight='bold', color='#f8fafc')
    ax2.set_xlabel('Residual Error: Fuzzy - Reference (%)', color='#94a3b8', fontsize=9.5)
    ax2.set_ylabel('Frequency', color='#94a3b8', fontsize=9.5)
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
    plt.tight_layout()

    # Chart 2: Baseline vs Fuzzy Logic Comparison
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.patch.set_facecolor('#0f172a')
    for ax in (ax3, ax4):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

    # Water Volume Comparison Bar Chart
    models = ['Reference Benchmark', 'Fuzzy Logic System', 'Threshold Baseline']
    volumes = [
        eval_results['water_savings']['reference_volume'],
        eval_results['fuzzy']['simulated_water_volume'],
        eval_results['baseline']['simulated_water_volume']
    ]
    colors = ['#94a3b8', '#38bdf8', '#f87171']
    bars = ax3.bar(models, volumes, color=colors, width=0.55, edgecolor='#0f172a')
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., h + 600, f'{h:,.0f}', ha='center', va='bottom', color='#f8fafc', fontsize=9, fontweight='bold')
    ax3.set_title('Simulated Cumulative Water Consumption', fontsize=11, fontweight='bold', color='#f8fafc')
    ax3.set_ylabel('Simulated Relative Units', color='#94a3b8', fontsize=9.5)
    ax3.set_ylim(0, max(volumes) * 1.18)

    # MAE & Accuracy Comparison
    x_indices = np.arange(2)
    width = 0.35
    ax4.bar(x_indices - width / 2, [eval_results['fuzzy']['mae'], eval_results['baseline']['mae']], width, label='MAE (lower is better)', color='#38bdf8')
    ax4.bar(x_indices + width / 2, [eval_results['fuzzy']['rmse'], eval_results['baseline']['rmse']], width, label='RMSE (lower is better)', color='#fbbf24')
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(['Fuzzy Logic', 'Threshold Baseline'], color='#94a3b8', fontsize=9.5)
    ax4.set_title('Error Metrics Comparison (MAE vs RMSE)', fontsize=11, fontweight='bold', color='#f8fafc')
    ax4.set_ylabel('Error (%)', color='#94a3b8', fontsize=9.5)
    ax4.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5)
    plt.tight_layout()

    return fig1, fig2


if __name__ == '__main__':
    from pathlib import Path
    data_file = Path(__file__).resolve().parent / 'data' / 'irrigation_dataset.csv'
    if not data_file.exists():
        from generate_dataset import generate_irrigation_dataset
        df = generate_irrigation_dataset(output_path=str(data_file))
    else:
        df = pd.read_csv(data_file)

    eval_out = evaluate_irrigation_systems(df)
    print("=== FUZZY LOGIC EVALUATION RESULTS ===")
    print(f"MAE: {eval_out['fuzzy']['mae']}%")
    print(f"RMSE: {eval_out['fuzzy']['rmse']}%")
    print(f"R^2 Score: {eval_out['fuzzy']['r2']}")
    print(f"Category Accuracy: {eval_out['fuzzy']['accuracy']}%")
    print("\n=== BASELINE COMPARISON ===")
    print(f"Threshold MAE: {eval_out['baseline']['mae']}%")
    print(f"Threshold RMSE: {eval_out['baseline']['rmse']}%")
    print(f"Threshold Accuracy: {eval_out['baseline']['accuracy']}%")
    print(f"Simulated Water Savings: {eval_out['water_savings']['simulated_savings_pct']}%")
    print(f"Fuzzy Overwater Episodes: {eval_out['fuzzy']['overwater_episodes']} vs Baseline: {eval_out['baseline']['overwater_episodes']}")
    if 'ann_comparison' in eval_out and 'test_mae' in eval_out['ann_comparison']:
        print("\n=== OPTIONAL ANN COMPARISON ===")
        print(f"ANN Test MAE: {eval_out['ann_comparison']['test_mae']}%")
        print(f"ANN Test RMSE: {eval_out['ann_comparison']['test_rmse']}%")
        print(f"ANN Test R^2: {eval_out['ann_comparison']['test_r2']}")

