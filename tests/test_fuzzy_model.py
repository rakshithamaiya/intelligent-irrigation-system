"""
Automated Pytest Suite for Intelligent Irrigation Fuzzy Inference System
Tests cover:
- Core agronomic scenarios
- Environmental edge cases and boundary conditions
- Monotonicity and Soft Computing sanity checks
- Membership coverage (no dead zones)
- Categorical consistency
- Robust error handling for invalid/out-of-bounds inputs
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fuzzy_model import FuzzyIrrigationSystem


@pytest.fixture(scope="module")
def fis():
    """Instantiate the Fuzzy Irrigation System once for tests."""
    return FuzzyIrrigationSystem()


# =====================================================================
# 1. Core Environmental Scenarios (Scenarios 1 - 8)
# =====================================================================

def test_scenario_1_very_dry_soil_high_temp_low_humidity(fis):
    """Scenario 1: Critical drought condition -> High irrigation."""
    res = fis.predict(moisture=10.0, temperature=45.0, humidity=15.0)
    assert res['intensity'] >= 70.0, f"Expected High irrigation, got {res['intensity']}"
    assert res['category'] == 'High'
    assert 'high' in res['summary'].lower()


def test_scenario_2_very_dry_soil_cool_temp(fis):
    """Scenario 2: Dry soil in cool/humid weather -> Low to Medium irrigation."""
    res = fis.predict(moisture=10.0, temperature=10.0, humidity=80.0)
    assert 20.0 <= res['intensity'] < 70.0
    assert res['category'] in ['Low', 'Medium']


def test_scenario_3_moderate_soil_high_temp_low_humidity(fis):
    """Scenario 3: Moderate soil moisture with high evapotranspiration demand -> Medium irrigation."""
    res = fis.predict(moisture=50.0, temperature=42.0, humidity=15.0)
    assert 45.0 <= res['intensity'] < 70.0
    assert res['category'] == 'Medium'


def test_scenario_4_moderate_soil_normal_weather(fis):
    """Scenario 4: Moderate soil moisture in mild weather -> Low irrigation."""
    res = fis.predict(moisture=50.0, temperature=25.0, humidity=50.0)
    assert 20.0 <= res['intensity'] < 45.0
    assert res['category'] == 'Low'


def test_scenario_5_wet_soil_hot_weather(fis):
    """Scenario 5: Wet soil under hot conditions -> Off (root aeration priority)."""
    res = fis.predict(moisture=85.0, temperature=42.0, humidity=25.0)
    assert res['intensity'] < 20.0
    assert res['category'] == 'Off'


def test_scenario_6_wet_soil_cool_humid_weather(fis):
    """Scenario 6: Saturated soil in cool humid weather -> Off."""
    res = fis.predict(moisture=90.0, temperature=12.0, humidity=85.0)
    assert res['intensity'] < 20.0
    assert res['category'] == 'Off'


def test_scenario_7_low_vs_high_humidity_effect(fis):
    """Scenario 7: Lower humidity should increase or maintain evaporative irrigation demand."""
    res_dry_air = fis.predict(moisture=20.0, temperature=30.0, humidity=15.0)
    res_humid_air = fis.predict(moisture=20.0, temperature=30.0, humidity=85.0)
    assert res_dry_air['intensity'] >= res_humid_air['intensity']


def test_scenario_8_cool_vs_hot_temperature_effect(fis):
    """Scenario 8: Higher temperature should increase or maintain irrigation demand."""
    res_cool = fis.predict(moisture=20.0, temperature=10.0, humidity=50.0)
    res_hot = fis.predict(moisture=20.0, temperature=42.0, humidity=50.0)
    assert res_hot['intensity'] >= res_cool['intensity']


# =====================================================================
# 2. Boundary and Domain Limits (Scenarios 9 - 11)
# =====================================================================

def test_scenario_9_minimum_boundary_values(fis):
    """Scenario 9: Absolute minimum domain limits (0%, 0°C, 0%)."""
    res = fis.predict(moisture=0.0, temperature=0.0, humidity=0.0)
    assert np.isfinite(res['intensity'])
    assert 0.0 <= res['intensity'] <= 100.0
    assert res['category'] in ['Off', 'Low', 'Medium', 'High']


def test_scenario_10_maximum_boundary_values(fis):
    """Scenario 10: Absolute maximum domain limits (100%, 50°C, 100%)."""
    res = fis.predict(moisture=100.0, temperature=50.0, humidity=100.0)
    assert np.isfinite(res['intensity'])
    assert 0.0 <= res['intensity'] <= 100.0
    # Saturated soil (100%) must be Off
    assert res['category'] == 'Off'


def test_scenario_11_intermediate_crossover_boundaries(fis):
    """Scenario 11: Set overlap boundaries (e.g. 25%, 50%, 75%)."""
    crossover_inputs = [
        (25.0, 20.0, 50.0),
        (50.0, 25.0, 50.0),
        (75.0, 35.0, 75.0),
        (35.0, 28.0, 45.0)
    ]
    for m, t, h in crossover_inputs:
        res = fis.predict(m, t, h)
        assert np.isfinite(res['intensity'])
        assert 0.0 <= res['intensity'] <= 100.0


# =====================================================================
# 3. Soft Computing & Monotonicity Properties (Scenarios 12 - 14)
# =====================================================================

def test_scenario_12_soil_moisture_monotonicity(fis):
    """Scenario 12: Dry soil receives strictly more irrigation than Wet soil under identical weather."""
    res_dry = fis.predict(moisture=15.0, temperature=32.0, humidity=40.0)
    res_wet = fis.predict(moisture=85.0, temperature=32.0, humidity=40.0)
    assert res_dry['intensity'] > res_wet['intensity']


def test_scenario_13_categorical_consistency(fis):
    """Scenario 13: Category mapping strictly adheres to defined numerical ranges."""
    test_grid = [
        (m, t, h)
        for m in [5.0, 25.0, 50.0, 75.0, 95.0]
        for t in [8.0, 25.0, 42.0]
        for h in [15.0, 50.0, 85.0]
    ]
    for m, t, h in test_grid:
        res = fis.predict(m, t, h)
        val = res['intensity']
        cat = res['category']
        if val < 20.0:
            assert cat == 'Off'
        elif val < 45.0:
            assert cat == 'Low'
        elif val < 70.0:
            assert cat == 'Medium'
        else:
            assert cat == 'High'


def test_scenario_14_membership_domain_coverage_no_dead_zones(fis):
    """Scenario 14: Verify that membership functions cover the full domain without dead zones."""
    # Test 200 points across Soil Moisture
    for x in np.linspace(0.0, 100.0, 200):
        mu_sum = (
            fis.eval_membership('moisture', 'dry', x) +
            fis.eval_membership('moisture', 'moderate', x) +
            fis.eval_membership('moisture', 'wet', x)
        )
        assert mu_sum >= 0.3, f"Dead zone detected in soil moisture at x={x}"

    # Test 100 points across Temperature
    for x in np.linspace(0.0, 50.0, 100):
        mu_sum = (
            fis.eval_membership('temperature', 'cool', x) +
            fis.eval_membership('temperature', 'normal', x) +
            fis.eval_membership('temperature', 'hot', x)
        )
        assert mu_sum >= 0.3, f"Dead zone detected in temperature at x={x}"

    # Test 200 points across Humidity
    for x in np.linspace(0.0, 100.0, 200):
        mu_sum = (
            fis.eval_membership('humidity', 'low', x) +
            fis.eval_membership('humidity', 'medium', x) +
            fis.eval_membership('humidity', 'high', x)
        )
        assert mu_sum >= 0.3, f"Dead zone detected in humidity at x={x}"


# =====================================================================
# 4. Error Handling & Rejection of Invalid Inputs (Scenario 15)
# =====================================================================

@pytest.mark.parametrize("invalid_m, invalid_t, invalid_h", [
    (-5.0, 25.0, 50.0),    # Negative moisture
    (105.0, 25.0, 50.0),   # Moisture > 100
    (50.0, -2.0, 50.0),    # Negative temperature
    (50.0, 55.0, 50.0),    # Temperature > 50
    (50.0, 25.0, -10.0),   # Negative humidity
    (50.0, 25.0, 110.0),   # Humidity > 100
    (np.nan, 25.0, 50.0),  # NaN
    (50.0, np.inf, 50.0),  # Infinity
])
def test_scenario_15_invalid_inputs_raise_value_error(fis, invalid_m, invalid_t, invalid_h):
    """Scenario 15: Out of range or invalid numeric inputs must be rejected with helpful messages."""
    with pytest.raises(ValueError) as excinfo:
        fis.predict(invalid_m, invalid_t, invalid_h)
    assert "must be between" in str(excinfo.value) or "cannot be NaN" in str(excinfo.value) or "finite" in str(excinfo.value)


def test_scenario_16_non_numeric_input_rejection(fis):
    """Scenario 16: String or None inputs must raise ValueError."""
    with pytest.raises(ValueError):
        fis.predict("very_dry", 25.0, 50.0)


def test_scenario_17_rule_firing_explanations(fis):
    """Scenario 17: Fired rules should contain index, linguistic terms, and positive alpha."""
    res = fis.predict(15.0, 38.0, 20.0)
    assert len(res['active_rules']) > 0
    top_rule = res['active_rules'][0]
    assert 'alpha' in top_rule
    assert top_rule['alpha'] > 0.0
    assert 'rule' in top_rule
    assert 'action' in res


def test_scenario_18_batch_prediction(fis):
    """Scenario 18: Vectorized batch prediction returns arrays with correct length."""
    m = np.array([10.0, 50.0, 90.0])
    t = np.array([40.0, 25.0, 15.0])
    h = np.array([20.0, 50.0, 80.0])
    intensities, categories = fis.predict_batch(m, t, h)
    assert len(intensities) == 3
    assert len(categories) == 3
    assert categories[0] == 'High'
    assert categories[2] == 'Off'
