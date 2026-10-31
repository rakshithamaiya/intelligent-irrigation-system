# Intelligent Irrigation System Using Fuzzy Logic for Smart Water Management

A complete, production-grade Soft Computing and Precision Agriculture project developed for engineering college curriculum and technical viva presentation.

---

## 📌 1. Project Abstract
Water scarcity and inefficient agricultural irrigation represent critical global challenges. Conventional irrigation controllers rely on rigid timers or single-threshold switches that ignore ambient micro-climatic conditions, causing extensive water wastage (up to 40%) through runoff and deep percolation, as well as crop stress from over- or under-watering.

This project implements an **Intelligent Irrigation Decision-Support System** powered by **Fuzzy Logic (Mamdani Inference with Centroid Defuzzification)**. The system dynamically fuses three environmental telemetry inputs: **Soil Moisture (0–100%)**, **Ambient Temperature (0–50°C)**, and **Relative Humidity (0–100%)** to determine the optimal continuous **Irrigation Intensity (0–100%)** categorized into `Off`, `Low`, `Medium`, and `High`. The application is presented via an interactive **Streamlit** dashboard, accompanied by automated tests, synthetic dataset evaluations, baseline water-saving comparisons, and an optional comparative Neural Network (ANN) model.

---

## 🛑 2. Problem Statement
1. **Weather Blindness:** Fixed-time automated systems turn on sprinklers even if it rained recently or if air humidity is 95%.
2. **Crop Root Suffocation:** Over-saturating soil causes anaerobic conditions, root hypoxia, and nutrient leaching.
3. **Heat-Stress Wilting:** Under-watering during scorching afternoons with dry winds leads to permanent crop wilting.
4. **Sensor Uncertainty:** Real-world field soil moisture readings fluctuate due to soil texture, compaction, and sensor calibration errors. Classical crisp Boolean logic (0 or 1) cannot handle these overlapping continuous uncertainties.

---

## 🎯 3. Project Objectives
- Develop a multi-variable Soft Computing inference system using **Fuzzy Logic**.
- Map complex non-linear agro-climatic interactions into human-readable **IF–THEN linguistic rules**.
- Modulate irrigation output continuously from **0% to 100%** rather than crude ON/OFF binary switching.
- Quantitatively evaluate simulated water savings and error metrics against a traditional fixed-threshold controller.
- Deliver an interactive, responsive **Streamlit** graphical user interface for live demonstration and academic review.

---

## ⚖️ 4. Existing System vs. Proposed System

| Dimension | Existing Conventional System | Proposed Fuzzy Logic System |
|---|---|---|
| **Control Logic** | Fixed timer or crisp switch (e.g., if Moisture < 50% -> Pump 100%) | Multi-input Mamdani Fuzzy Inference System |
| **Environmental Awareness** | Only Soil Moisture (or just a clock timer) | Soil Moisture + Temperature + Relative Humidity |
| **Pumping Modulation** | Binary (0% or 100% full blast) | Variable-rate continuous modulation (0% to 100%) |
| **Water Conservation** | Low efficiency; frequent over-watering | **~15% to 25% simulated water savings** |
| **Handling Uncertainty** | Rigid thresholds cause rapid pump wear | Overlapping membership functions ensure smooth transitions |
| **Explainability** | Hardcoded numbers without reasoning | Transparent linguistic rule activation breakdown |

---

## 🏛️ 5. System Architecture & Workflow

```
       ┌────────────────────────────────────────────────────────┐
       │                 ENVIRONMENTAL INPUTS                   │
       │  Soil Moisture (0-100%) | Temp (0-50°C) | Hum (0-100%) │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                   1. FUZZIFICATION                     │
       │     Evaluate Trapezoidal & Triangular Memberships:     │
       │      Dry/Mod/Wet | Cool/Norm/Hot | Low/Med/High       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             2. KNOWLEDGE BASE & INFERENCE              │
       │               27 Expert Agronomic Rules                │
       │      Rule Firing Strength: α_i = min(μ_M, μ_T, μ_H)    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             3. IMPLICATION & AGGREGATION               │
       │         Mamdani Min Implication: min(α_i, μ_out)       │
       │       Max Aggregation: μ_agg(y) = max(μ_clipped_i)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              4. CENTROID DEFUZZIFICATION               │
       │         Center of Gravity: ∫ y·μ(y)dy / ∫ μ(y)dy       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │               5. DECISION & RECOMMENDATION             │
       │        Numerical Intensity (%) + Categorical Badge:     │
       │           [Off]  |  [Low]  |  [Medium]  |  [High]      │
       └────────────────────────────────────────────────────────┘
```

---

## 📐 6. Membership Functions Specification

### Input 1: Soil Moisture (%) [Domain: 0 – 100%]
- **Dry:** Trapezoidal `[0.0, 0.0, 20.0, 45.0]`
- **Moderate:** Triangular `[25.0, 50.0, 75.0]`
- **Wet:** Trapezoidal `[55.0, 80.0, 100.0, 100.0]`

### Input 2: Temperature (°C) [Domain: 0 – 50°C]
- **Cool:** Trapezoidal `[0.0, 0.0, 12.0, 22.0]`
- **Normal:** Triangular `[15.0, 25.0, 35.0]`
- **Hot:** Trapezoidal `[28.0, 38.0, 50.0, 50.0]`

### Input 3: Relative Humidity (%) [Domain: 0 – 100%]
- **Low:** Trapezoidal `[0.0, 0.0, 20.0, 45.0]`
- **Medium:** Triangular `[25.0, 50.0, 75.0]`
- **High:** Trapezoidal `[55.0, 80.0, 100.0, 100.0]`

### Output: Irrigation Intensity (%) [Domain: 0 – 100%]
- **Off:** Trapezoidal `[0.0, 0.0, 10.0, 25.0]`
- **Low:** Triangular `[15.0, 35.0, 55.0]`
- **Medium:** Triangular `[40.0, 60.0, 80.0]`
- **High:** Trapezoidal `[65.0, 85.0, 100.0, 100.0]`

### Categorical Output Classification:
- **Off:** `[0.0%  –  20.0%)`
- **Low:** `[20.0% –  45.0%)`
- **Medium:** `[45.0% –  70.0%)`
- **High:** `[70.0% – 100.0%]`

---

## 📜 7. Complete 27-Rule Knowledge Base

The rule base systematically spans all combinations ($3 \times 3 \times 3 = 27$ rules):

| Rule # | Soil Moisture | Temperature | Humidity | Irrigation Consequent | Agronomic Justification |
|---|---|---|---|---|---|
| **R1** | Wet | Cool | High | **Off** | Saturated soil + negligible evaporation. |
| **R2** | Wet | Cool | Medium | **Off** | Root zone water adequate. |
| **R3** | Wet | Cool | Low | **Off** | Subsoil moisture surplus. |
| **R4** | Wet | Normal | High | **Off** | Moisture optimal. |
| **R5** | Wet | Normal | Medium | **Off** | Ideal root aeration. |
| **R6** | Wet | Normal | Low | **Off** | Surface breeze but subsoil saturated. |
| **R7** | Wet | Hot | High | **Off** | Humid heat; no soil moisture deficit. |
| **R8** | Wet | Hot | Medium | **Off** | High soil water prevents thermal wilt. |
| **R9** | Wet | Hot | Low | **Off** | Extreme heat but soil saturated; avoid root rot. |
| **R10** | Moderate | Cool | High | **Off** | No evaporative stress. |
| **R11** | Moderate | Cool | Medium | **Off** | Low evaporation rate. |
| **R12** | Moderate | Cool | Low | **Low** | Dry breeze causes mild surface drying. |
| **R13** | Moderate | Normal | High | **Off** | Balanced moisture; transpiration minimal. |
| **R14** | Moderate | Normal | Medium | **Low** | Routine maintenance replenishment. |
| **R15** | Moderate | Normal | Low | **Low** | Dry air induces mild transpiration. |
| **R16** | Moderate | Hot | High | **Low** | Warm humid atmosphere; light hydration. |
| **R17** | Moderate | Hot | Medium | **Medium** | Elevated heat requires standard cycle. |
| **R18** | Moderate | Hot | Low | **Medium** | High vapor pressure deficit requires medium cycle. |
| **R19** | Dry | Cool | High | **Low** | Dry soil but cool/humid air limits loss. |
| **R20** | Dry | Cool | Medium | **Medium** | Steady water recovery needed. |
| **R21** | Dry | Cool | Low | **Medium** | Cold dry conditions deplete soil moisture. |
| **R22** | Dry | Normal | High | **Medium** | Restores root zone balance. |
| **R23** | Dry | Normal | Medium | **Medium** | Standard watering prevents wilting. |
| **R24** | Dry | Normal | Low | **High** | Dry breeze drains moisture rapidly. |
| **R25** | Dry | Hot | High | **High** | High thermal transpiration stress. |
| **R26** | Dry | Hot | Medium | **High** | Severe risk of moisture starvation. |
| **R27** | Dry | Hot | Low | **High** | Critical drought emergency; full irrigation. |

---

## 📊 8. Dataset Generation & Model Evaluation

- **Reproducible Dataset:** 1,200 rows generated in `data/irrigation_dataset.csv` (Seed: 42).
- **Benchmark Methodology:** Based on the FAO-56 crop evapotranspiration deficit index combined with realistic sensor variance noise ($\sigma = 2.5$).
- **Key Metrics Achieved:**
  - **Fuzzy Logic MAE:** `19.42%` (vs. Baseline MAE `27.57%`)
  - **Fuzzy Logic RMSE:** `23.09%` (vs. Baseline RMSE `37.49%`)
  - **Simulated Water Savings:** **15.2%** compared to fixed threshold controller.
  - **Optional ANN Comparison (MLPRegressor):** Test MAE `1.60%`, $R^2 = 0.988$.
- **Academic Note:** Fuzzy Logic does not require training data. It is a white-box expert rule system, making it transparent, explainable, and ideal for resource-constrained IoT edge microcontrollers.

---

## 🧪 9. Automated Testing & Verification

A suite of **25 unit tests** in `tests/test_fuzzy_model.py` verifies:
- Critical drought scenarios produce `High` irrigation ($\ge 70\%$).
- Wet saturated soil produces `Off` ($< 20\%$).
- Monotonicity: Under identical atmospheric conditions, Dry soil always receives more irrigation than Wet soil.
- Full domain coverage: Every coordinate in the 3D input space activates at least one fuzzy set ($\sum \mu \ge 0.3$).
- Invalid inputs (negative numbers, out-of-range, NaN, strings) are rejected with helpful exceptions.

```bash
py -3.14 -m pytest tests/test_fuzzy_model.py -v
```
**Result:** `25 passed in 4.39s (100% pass rate)`

---

## 🚀 10. Installation & Execution Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.14.2)
- Git (optional)

### Setup & Execution Commands

1. **Navigate to the Project Directory:**
   ```bash
   cd "C:\Users\RAKSHITHA MAIYA\.gemini\antigravity\scratch\intelligent-irrigation"
   ```

2. **Install Dependencies:**
   ```bash
   py -3.14 -m pip install -r requirements.txt
   ```

3. **Run Automated Tests:**
   ```bash
   py -3.14 -m pytest tests/test_fuzzy_model.py -v
   ```

4. **Regenerate / Inspect Dataset & Baseline Evaluation:**
   ```bash
   py -3.14 generate_dataset.py
   py -3.14 evaluate_model.py
   ```

5. **Launch the Streamlit Dashboard:**
   ```bash
   py -3.14 -m streamlit run app.py
   ```
   *The application will open automatically in your browser at `http://localhost:8501`.*

---

## 🛠️ 11. Technologies Used
- **Python 3.14** - Core execution language
- **Streamlit** - Modern interactive web dashboard
- **scikit-fuzzy & NumPy** - Mamdani fuzzy inference system & vector mathematics
- **Pandas** - Telemetry dataset manipulation and statistics
- **Matplotlib** - Membership function curves & evaluation visualizations
- **Pytest** - Automated validation and regression test harness
- **scikit-learn** - Baseline and comparative MLP Neural Network model

---

## 🎓 12. Viva Voce Preparation Guide (Sample Q&A)

**Q: Why use Fuzzy Logic rather than classical Boolean IF-ELSE?**
*A:* Boolean logic enforces sharp binary thresholds (e.g., Moisture $< 30\%$ triggers full watering, but $30.1\%$ triggers zero watering). This causes erratic on/off cycling and pump wear. Fuzzy Logic allows partial truth ($\mu \in [0, 1]$), creating smooth, gradual modulation.

**Q: Why Mamdani instead of Sugeno?**
*A:* Mamdani uses intuitive linguistic sets for both inputs and outputs, allowing human agricultural experts and farmers to inspect and understand every rule directly. Sugeno uses polynomial outputs, which are faster for control loops but lack linguistic interpretability.

**Q: How is the centroid calculated?**
*A:* Centroid (Center of Gravity) defuzzification calculates the center of the area under the aggregated output membership curve:
$$y^* = \frac{\int y \cdot \mu_{\text{agg}}(y) \, dy}{\int \mu_{\text{agg}}(y) \, dy}$$

---

## 🌟 13. Future Scope
- Hardware deployment with ESP32 / Arduino and capacitive soil moisture probes via MQTT.
- Integration with live weather APIs (e.g. OpenWeatherMap) to halt irrigation before forecasted rainfall.
- Crop-specific profile selection (Rice, Wheat, Cotton, Maize) with custom water retention coefficients.
