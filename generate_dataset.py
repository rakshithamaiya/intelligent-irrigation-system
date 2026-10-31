"""
Dataset Generator for Intelligent Irrigation System Using Fuzzy Logic

Generates a reproducible synthetic dataset of environmental sensor readings
(Soil Moisture, Ambient Temperature, Relative Humidity) alongside:
1. An agronomic benchmark reference (Crop Evapotranspiration Deficit model with sensor noise)
2. Live Mamdani Fuzzy Logic predictions computed by FuzzyIrrigationSystem

Seed: 42 (Reproducible)
Sample Size: 1,200 records
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fuzzy_model import FuzzyIrrigationSystem


def generate_irrigation_dataset(
    n_samples: int = 1200,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic environmental dataset and execute fuzzy inference.

    Agronomic Evapotranspiration Deficit Model (FAO-56 Simplified Benchmark):
    - Soil Moisture (%): Realistic mixture of dry conditions, moderate field capacity, and saturated soil.
    - Temperature (°C): 5°C to 48°C diurnal range.
    - Relative Humidity (%): 10% to 95% inversely correlated with temperature.
    - Reference Intensity (%):
      * Depletion ratio: D = clip((60.0 - Moisture) / 60.0, 0, 1)
      * Vapor pressure deficit (VPD) factor: V = (Temp / 50.0) * (1.15 - 0.75 * (Humidity / 100.0))
      * Reference = clip(100.0 * (0.50 * D + 0.50 * D * V) + Noise, 0, 100)
      * If Soil Moisture >= 55.0%, Reference = 0.0% (saturation / root aeration protection).
    """
    np.random.seed(random_seed)

    # 1. Soil Moisture (%) - Multimodal realistic soil distribution
    n_dry = int(n_samples * 0.35)
    n_mod = int(n_samples * 0.40)
    n_wet = n_samples - n_dry - n_mod

    m_dry = np.random.uniform(5.0, 32.0, n_dry)
    m_mod = np.random.uniform(30.0, 65.0, n_mod)
    m_wet = np.random.uniform(62.0, 98.0, n_wet)
    soil_moisture = np.concatenate([m_dry, m_mod, m_wet])

    # 2. Temperature (°C)
    temperature = np.random.normal(loc=28.0, scale=8.0, size=n_samples)
    temperature = np.clip(temperature, 6.0, 48.0)

    # 3. Relative Humidity (%) - Inverse diurnal relationship with temperature
    base_humidity = 96.0 - (temperature * 1.25) + np.random.normal(0, 10.0, size=n_samples)
    humidity = np.clip(base_humidity, 10.0, 96.0)

    # Permute
    indices = np.random.permutation(n_samples)
    soil_moisture = np.round(soil_moisture[indices], 2)
    temperature = np.round(temperature[indices], 2)
    humidity = np.round(humidity[indices], 2)

    # 4. Reference Benchmark Calculation (Agronomic Soil-Water Deficit Model)
    depletion = np.clip((60.0 - soil_moisture) / 60.0, 0.0, 1.0)
    vpd_factor = (temperature / 50.0) * (1.15 - 0.75 * (humidity / 100.0))
    vpd_factor = np.clip(vpd_factor, 0.05, 1.0)

    ref_intensity = 100.0 * (0.50 * depletion + 0.50 * depletion * vpd_factor)
    # Saturation rule: if soil moisture >= 55%, zero irrigation needed
    ref_intensity = np.where(soil_moisture >= 55.0, 0.0, ref_intensity)

    # Realistic environmental sensor noise
    noise = np.random.normal(0.0, 2.5, size=n_samples)
    ref_intensity = np.where(ref_intensity == 0.0, 0.0, np.clip(ref_intensity + noise, 0.0, 100.0))
    ref_intensity = np.round(ref_intensity, 2)

    # Reference Categories
    fis = FuzzyIrrigationSystem()
    ref_categories = [fis.classify_intensity(val) for val in ref_intensity]

    # 5. Live Fuzzy Inference Execution
    print(f"[Dataset Generator] Running Mamdani Fuzzy System across {n_samples} samples...")
    fuzzy_intensities, fuzzy_categories = fis.predict_batch(soil_moisture, temperature, humidity)

    df = pd.DataFrame({
        'sample_id': np.arange(1, n_samples + 1),
        'soil_moisture': soil_moisture,
        'temperature': temperature,
        'humidity': humidity,
        'reference_intensity': ref_intensity,
        'reference_category': ref_categories,
        'fuzzy_intensity': np.round(fuzzy_intensities, 2),
        'fuzzy_category': fuzzy_categories
    })

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Dataset Generator] Successfully saved {len(df)} rows to: {output_path}")

    return df


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    out_file = base_dir / 'data' / 'irrigation_dataset.csv'
    dataset = generate_irrigation_dataset(n_samples=1200, random_seed=42, output_path=str(out_file))
    print(dataset.head())
    print("\nFuzzy vs Reference Category Distribution:")
    print("Fuzzy Categories:\n", dataset['fuzzy_category'].value_counts())
    print("Reference Categories:\n", dataset['reference_category'].value_counts())
