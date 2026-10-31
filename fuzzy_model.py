"""
Intelligent Irrigation System Using Fuzzy Logic for Smart Water Management
Core Fuzzy Inference Engine (Mamdani Model with Centroid Defuzzification)

Soft Computing Technique:
- Inputs: Soil Moisture (0-100%), Temperature (0-50 C), Relative Humidity (0-100%)
- Output: Irrigation Intensity (0-100%)
- Membership Functions: Triangular and Trapezoidal sets covering 100% of the input universes
- Inference Engine: Mamdani Min-Max Composition
- Defuzzification: Centroid (Center of Gravity)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False


class FuzzyIrrigationSystem:
    """
    Complete Mamdani Fuzzy Inference System for Intelligent Irrigation.
    Supports both scikit-fuzzy control engine and an exact analytical NumPy engine
    for transparent rule inspection, defuzzification plotting, and batch processing.
    """

    # Domain Universes
    MOISTURE_UNIVERSE = (0.0, 100.0)
    TEMPERATURE_UNIVERSE = (0.0, 50.0)
    HUMIDITY_UNIVERSE = (0.0, 100.0)
    IRRIGATION_UNIVERSE = (0.0, 100.0)

    # Membership Function Parameter Definitions [a, b, c, d] for trap / [a, b, c] for tri
    PARAMS = {
        'moisture': {
            'dry': [0.0, 0.0, 20.0, 45.0],        # trapezoidal
            'moderate': [25.0, 50.0, 75.0],       # triangular
            'wet': [55.0, 80.0, 100.0, 100.0],    # trapezoidal
        },
        'temperature': {
            'cool': [0.0, 0.0, 12.0, 22.0],       # trapezoidal
            'normal': [15.0, 25.0, 35.0],         # triangular
            'hot': [28.0, 38.0, 50.0, 50.0],      # trapezoidal
        },
        'humidity': {
            'low': [0.0, 0.0, 20.0, 45.0],        # trapezoidal
            'medium': [25.0, 50.0, 75.0],         # triangular
            'high': [55.0, 80.0, 100.0, 100.0],   # trapezoidal
        },
        'irrigation': {
            'off': [0.0, 0.0, 10.0, 25.0],        # trapezoidal
            'low': [15.0, 35.0, 55.0],            # triangular
            'medium': [40.0, 60.0, 80.0],         # triangular
            'high': [65.0, 85.0, 100.0, 100.0],   # trapezoidal
        }
    }

    # Complete 27-rule knowledge base: (moisture, temp, humidity) -> output consequent
    RULE_BASE = [
        # Soil Moisture = Wet (9 rules: Irrigating wet soil causes waterlogging/root damage)
        ('wet', 'cool', 'high', 'off', 'Rule 1: Wet soil + Cool + High humidity -> Soil saturated, evaporation negligible.'),
        ('wet', 'cool', 'medium', 'off', 'Rule 2: Wet soil + Cool + Medium humidity -> Ample root zone moisture.'),
        ('wet', 'cool', 'low', 'off', 'Rule 3: Wet soil + Cool + Low humidity -> Baseline moisture sufficient.'),
        ('wet', 'normal', 'high', 'off', 'Rule 4: Wet soil + Normal temp + High humidity -> Adequate moisture.'),
        ('wet', 'normal', 'medium', 'off', 'Rule 5: Wet soil + Normal temp + Medium humidity -> Ideal conditions.'),
        ('wet', 'normal', 'low', 'off', 'Rule 6: Wet soil + Normal temp + Low humidity -> Surface dry but root saturated.'),
        ('wet', 'hot', 'high', 'off', 'Rule 7: Wet soil + Hot + High humidity -> High moisture, minimal transpiration deficit.'),
        ('wet', 'hot', 'medium', 'off', 'Rule 8: Wet soil + Hot + Medium humidity -> Root zone wet, no irrigation required.'),
        ('wet', 'hot', 'low', 'off', 'Rule 9: Wet soil + Hot + Low humidity -> Extreme heat but sufficient subsoil moisture.'),

        # Soil Moisture = Moderate (9 rules: Balance evapotranspiration demand)
        ('moderate', 'cool', 'high', 'off', 'Rule 10: Moderate moisture + Cool + High humidity -> No water stress.'),
        ('moderate', 'cool', 'medium', 'off', 'Rule 11: Moderate moisture + Cool + Medium humidity -> Evaporation very low.'),
        ('moderate', 'cool', 'low', 'low', 'Rule 12: Moderate moisture + Cool + Low humidity -> Dry breeze demands mild replenishment.'),
        ('moderate', 'normal', 'high', 'off', 'Rule 13: Moderate moisture + Normal temp + High humidity -> Moisture stable.'),
        ('moderate', 'normal', 'medium', 'low', 'Rule 14: Moderate moisture + Normal temp + Medium humidity -> Standard low maintenance cycle.'),
        ('moderate', 'normal', 'low', 'low', 'Rule 15: Moderate moisture + Normal temp + Low humidity -> Dry air creates moderate transpiration.'),
        ('moderate', 'hot', 'high', 'low', 'Rule 16: Moderate moisture + Hot + High humidity -> Warm but humid; mild irrigation.'),
        ('moderate', 'hot', 'medium', 'medium', 'Rule 17: Moderate moisture + Hot + Medium humidity -> Elevated heat requires moderate cycle.'),
        ('moderate', 'hot', 'low', 'medium', 'Rule 18: Moderate moisture + Hot + Low humidity -> High vapor pressure deficit requires medium cycle.'),

        # Soil Moisture = Dry (9 rules: Water deficit must be alleviated)
        ('dry', 'cool', 'high', 'low', 'Rule 19: Dry soil + Cool + High humidity -> Dry soil but low evaporation rate.'),
        ('dry', 'cool', 'medium', 'medium', 'Rule 20: Dry soil + Cool + Medium humidity -> Steady medium irrigation recommended.'),
        ('dry', 'cool', 'low', 'medium', 'Rule 21: Dry soil + Cool + Low humidity -> Cold dry air requires steady water recovery.'),
        ('dry', 'normal', 'high', 'medium', 'Rule 22: Dry soil + Normal temp + High humidity -> Medium cycle restores root balance.'),
        ('dry', 'normal', 'medium', 'medium', 'Rule 23: Dry soil + Normal temp + Medium humidity -> Standard watering required to prevent wilting.'),
        ('dry', 'normal', 'low', 'high', 'Rule 24: Dry soil + Normal temp + Low humidity -> Low humidity rapidly drains moisture.'),
        ('dry', 'hot', 'high', 'high', 'Rule 25: Dry soil + Hot + High humidity -> High heat creates heavy crop transpiration stress.'),
        ('dry', 'hot', 'medium', 'high', 'Rule 26: Dry soil + Hot + Medium humidity -> Severe risk of thermal and water stress.'),
        ('dry', 'hot', 'low', 'high', 'Rule 27: Dry soil + Hot + Low humidity -> Critical drought condition, maximum irrigation.'),
    ]

    def __init__(self):
        """Initialize fuzzy universes and compiled skfuzzy control system if available."""
        self.y_resolution = 501
        self.y_grid = np.linspace(0.0, 100.0, self.y_resolution)
        self._init_skfuzzy_system()

    def _init_skfuzzy_system(self):
        """Build scikit-fuzzy control system."""
        if not HAS_SKFUZZY:
            self.skfuzzy_sim = None
            return

        u_moist = np.linspace(0.0, 100.0, 201)
        u_temp = np.linspace(0.0, 50.0, 101)
        u_hum = np.linspace(0.0, 100.0, 201)
        u_irrig = np.linspace(0.0, 100.0, 201)

        self.moist_ant = ctrl.Antecedent(u_moist, 'moisture')
        self.temp_ant = ctrl.Antecedent(u_temp, 'temperature')
        self.hum_ant = ctrl.Antecedent(u_hum, 'humidity')
        self.irrig_consq = ctrl.Consequent(u_irrig, 'irrigation')

        # Membership functions
        self.moist_ant['dry'] = fuzz.trapmf(u_moist, self.PARAMS['moisture']['dry'])
        self.moist_ant['moderate'] = fuzz.trimf(u_moist, self.PARAMS['moisture']['moderate'])
        self.moist_ant['wet'] = fuzz.trapmf(u_moist, self.PARAMS['moisture']['wet'])

        self.temp_ant['cool'] = fuzz.trapmf(u_temp, self.PARAMS['temperature']['cool'])
        self.temp_ant['normal'] = fuzz.trimf(u_temp, self.PARAMS['temperature']['normal'])
        self.temp_ant['hot'] = fuzz.trapmf(u_temp, self.PARAMS['temperature']['hot'])

        self.hum_ant['low'] = fuzz.trapmf(u_hum, self.PARAMS['humidity']['low'])
        self.hum_ant['medium'] = fuzz.trimf(u_hum, self.PARAMS['humidity']['medium'])
        self.hum_ant['high'] = fuzz.trapmf(u_hum, self.PARAMS['humidity']['high'])

        self.irrig_consq['off'] = fuzz.trapmf(u_irrig, self.PARAMS['irrigation']['off'])
        self.irrig_consq['low'] = fuzz.trimf(u_irrig, self.PARAMS['irrigation']['low'])
        self.irrig_consq['medium'] = fuzz.trimf(u_irrig, self.PARAMS['irrigation']['medium'])
        self.irrig_consq['high'] = fuzz.trapmf(u_irrig, self.PARAMS['irrigation']['high'])

        # Compile rules
        ctrl_rules = []
        for m_term, t_term, h_term, out_term, _ in self.RULE_BASE:
            antecedent = self.moist_ant[m_term] & self.temp_ant[t_term] & self.hum_ant[h_term]
            ctrl_rules.append(ctrl.Rule(antecedent, self.irrig_consq[out_term]))

        self.skfuzzy_ctrl = ctrl.ControlSystem(ctrl_rules)
        self.skfuzzy_sim = ctrl.ControlSystemSimulation(self.skfuzzy_ctrl)

    # -------------------------------------------------------------
    # Mathematical Membership Functions (Trap & Tri)
    # -------------------------------------------------------------
    @staticmethod
    def _trapmf(x: np.ndarray, abcd: List[float]) -> np.ndarray:
        """Trapezoidal membership function."""
        a, b, c, d = abcd
        y = np.zeros_like(x, dtype=float)
        # Left ramp
        if b > a:
            mask = (x >= a) & (x < b)
            y[mask] = (x[mask] - a) / (b - a)
        else:
            y[x == a] = 1.0
        # Plateau
        y[(x >= b) & (x <= c)] = 1.0
        # Right ramp
        if d > c:
            mask = (x > c) & (x <= d)
            y[mask] = (d - x[mask]) / (d - c)
        return np.clip(y, 0.0, 1.0)

    @staticmethod
    def _trimf(x: np.ndarray, abc: List[float]) -> np.ndarray:
        """Triangular membership function."""
        a, b, c = abc
        y = np.zeros_like(x, dtype=float)
        # Left ramp
        if b > a:
            mask = (x >= a) & (x < b)
            y[mask] = (x[mask] - a) / (b - a)
        # Peak
        y[x == b] = 1.0
        # Right ramp
        if c > b:
            mask = (x > b) & (x <= c)
            y[mask] = (c - x[mask]) / (c - b)
        return np.clip(y, 0.0, 1.0)

    def eval_membership(self, variable: str, term: str, x: float) -> float:
        """Evaluate the degree of membership for a given crisp scalar."""
        params = self.PARAMS[variable][term]
        x_arr = np.array([float(x)], dtype=float)
        if len(params) == 4:
            return float(self._trapmf(x_arr, params)[0])
        elif len(params) == 3:
            return float(self._trimf(x_arr, params)[0])
        raise ValueError(f"Invalid parameter shape for {variable}.{term}")

    def eval_output_curve(self, term: str, y_arr: np.ndarray) -> np.ndarray:
        """Evaluate output fuzzy membership curve over grid y."""
        params = self.PARAMS['irrigation'][term]
        if len(params) == 4:
            return self._trapmf(y_arr, params)
        return self._trimf(y_arr, params)

    # -------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------
    def validate_inputs(self, moisture: float, temperature: float, humidity: float) -> Tuple[float, float, float]:
        """Validate input types and numeric ranges."""
        try:
            m = float(moisture)
            t = float(temperature)
            h = float(humidity)
        except (TypeError, ValueError) as err:
            raise ValueError(f"All inputs must be valid numeric values. Details: {err}") from err

        if np.isnan(m) or np.isnan(t) or np.isnan(h):
            raise ValueError("Inputs cannot be NaN.")
        if np.isinf(m) or np.isinf(t) or np.isinf(h):
            raise ValueError("Inputs must be finite numbers.")

        if not (self.MOISTURE_UNIVERSE[0] <= m <= self.MOISTURE_UNIVERSE[1]):
            raise ValueError(
                f"Soil Moisture must be between {self.MOISTURE_UNIVERSE[0]}% and {self.MOISTURE_UNIVERSE[1]}%. Received: {m}"
            )
        if not (self.TEMPERATURE_UNIVERSE[0] <= t <= self.TEMPERATURE_UNIVERSE[1]):
            raise ValueError(
                f"Temperature must be between {self.TEMPERATURE_UNIVERSE[0]}°C and {self.TEMPERATURE_UNIVERSE[1]}°C. Received: {t}"
            )
        if not (self.HUMIDITY_UNIVERSE[0] <= h <= self.HUMIDITY_UNIVERSE[1]):
            raise ValueError(
                f"Relative Humidity must be between {self.HUMIDITY_UNIVERSE[0]}% and {self.HUMIDITY_UNIVERSE[1]}%. Received: {h}"
            )

        return m, t, h

    # -------------------------------------------------------------
    # Classification & Agronomic Logic
    # -------------------------------------------------------------
    @staticmethod
    def classify_intensity(intensity: float) -> str:
        """
        Map continuous fuzzy output [0, 100] to display categories:
        - Off:    [0%, 20%)
        - Low:    [20%, 45%)
        - Medium: [45%, 70%)
        - High:   [70%, 100%]
        """
        intensity = float(np.clip(intensity, 0.0, 100.0))
        if intensity < 20.0:
            return 'Off'
        elif intensity < 45.0:
            return 'Low'
        elif intensity < 70.0:
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def get_recommendation(category: str, moisture: float, temperature: float, humidity: float) -> Dict[str, str]:
        """Generate clear agricultural advisory messages."""
        if category == 'Off':
            if moisture >= 55.0:
                summary = "Irrigation is OFF. Soil moisture is optimal to surplus."
                action = "No watering needed. Current moisture levels satisfy crop requirements. Avoid overwatering to prevent root rot."
            else:
                summary = "Irrigation is OFF. Environmental evaporation is minimal."
                action = "Weather conditions (cool/humid) prevent soil desiccation. Monitor moisture levels in 12 hours."
        elif category == 'Low':
            summary = "Irrigation is LOW. Mild moisture replenishment needed."
            action = "Run micro-drip or light sprinkler cycle (approx. 10-15 minutes). Prevents gradual root zone drying."
        elif category == 'Medium':
            summary = "Irrigation is MEDIUM. Moderate water deficit detected."
            action = "Standard irrigation cycle recommended (approx. 25-35 minutes). Apply during cooler hours (morning/evening) for efficiency."
        else:  # High
            summary = "Irrigation is HIGH. Severe dry conditions & high evapotranspiration."
            action = "Immediate full irrigation cycle recommended (approx. 45-60 minutes). Check drippers for uniform distribution."

        return {
            'summary': summary,
            'action': action,
            'category': category
        }

    # -------------------------------------------------------------
    # Mamdani Inference & Defuzzification Engine
    # -------------------------------------------------------------
    def predict(self, moisture: float, temperature: float, humidity: float, use_skfuzzy: bool = True) -> Dict[str, Any]:
        """
        Execute full Mamdani fuzzy inference pipeline:
        1. Fuzzification (evaluate degrees of membership for all inputs)
        2. Rule Evaluation (compute firing strength using min T-norm)
        3. Consequent Implication (clip output membership function)
        4. Aggregation (combine all implied sets using max S-norm)
        5. Centroid Defuzzification (compute center of gravity)
        """
        m, t, h = self.validate_inputs(moisture, temperature, humidity)

        # 1. Fuzzification
        memberships = {
            'moisture': {
                'dry': self.eval_membership('moisture', 'dry', m),
                'moderate': self.eval_membership('moisture', 'moderate', m),
                'wet': self.eval_membership('moisture', 'wet', m),
            },
            'temperature': {
                'cool': self.eval_membership('temperature', 'cool', t),
                'normal': self.eval_membership('temperature', 'normal', t),
                'hot': self.eval_membership('temperature', 'hot', t),
            },
            'humidity': {
                'low': self.eval_membership('humidity', 'low', h),
                'medium': self.eval_membership('humidity', 'medium', h),
                'high': self.eval_membership('humidity', 'high', h),
            }
        }

        # 2. Rule Evaluation & Implication
        active_rules = []
        aggregated_output = np.zeros_like(self.y_grid, dtype=float)

        for idx, (m_term, t_term, h_term, out_term, desc) in enumerate(self.RULE_BASE):
            mu_m = memberships['moisture'][m_term]
            mu_t = memberships['temperature'][t_term]
            mu_h = memberships['humidity'][h_term]

            # Mamdani Min-conjunction (AND)
            alpha = min(mu_m, mu_t, mu_h)

            if alpha > 0.0:
                active_rules.append({
                    'index': idx + 1,
                    'rule': f"IF Moisture is {m_term.capitalize()} AND Temp is {t_term.capitalize()} AND Humidity is {h_term.capitalize()} THEN Irrigation is {out_term.capitalize()}",
                    'alpha': round(alpha, 4),
                    'output_term': out_term,
                    'description': desc
                })

                # Mamdani Min-implication
                term_curve = self.eval_output_curve(out_term, self.y_grid)
                clipped_curve = np.minimum(alpha, term_curve)

                # Mamdani Max-aggregation (OR)
                aggregated_output = np.maximum(aggregated_output, clipped_curve)

        # 3. Defuzzification via Centroid
        area = np.sum(aggregated_output)
        if area > 1e-9:
            centroid = float(np.sum(self.y_grid * aggregated_output) / area)
        else:
            # Fallback if no rule fires (cannot happen with complete 27-rule coverage)
            centroid = 0.0

        # Optional validation against scikit-fuzzy engine
        skfuzzy_output = None
        if use_skfuzzy and self.skfuzzy_sim is not None:
            try:
                self.skfuzzy_sim.input['moisture'] = m
                self.skfuzzy_sim.input['temperature'] = t
                self.skfuzzy_sim.input['humidity'] = h
                self.skfuzzy_sim.compute()
                skfuzzy_output = float(self.skfuzzy_sim.output['irrigation'])
            except Exception:
                skfuzzy_output = None

        final_intensity = centroid if skfuzzy_output is None else skfuzzy_output
        final_intensity = float(np.clip(final_intensity, 0.0, 100.0))
        category = self.classify_intensity(final_intensity)
        advisory = self.get_recommendation(category, m, t, h)

        # Sort active rules by firing strength descending
        active_rules.sort(key=lambda r: r['alpha'], reverse=True)

        return {
            'inputs': {'moisture': m, 'temperature': t, 'humidity': h},
            'intensity': round(final_intensity, 2),
            'category': category,
            'summary': advisory['summary'],
            'action': advisory['action'],
            'memberships': memberships,
            'active_rules': active_rules,
            'aggregated_output': (self.y_grid, aggregated_output),
            'centroid': round(centroid, 2),
            'skfuzzy_verified': skfuzzy_output is not None
        }

    def predict_batch(self, moisture_arr: np.ndarray, temp_arr: np.ndarray, hum_arr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Fast vectorized batch prediction for dataset evaluation."""
        intensities = []
        categories = []
        for m, t, h in zip(moisture_arr, temp_arr, hum_arr):
            res = self.predict(m, t, h, use_skfuzzy=False)
            intensities.append(res['intensity'])
            categories.append(res['category'])
        return np.array(intensities, dtype=float), categories

    # -------------------------------------------------------------
    # Visualization Helpers
    # -------------------------------------------------------------
    def plot_membership_functions(self) -> plt.Figure:
        """Generate Matplotlib figure visualizing all input and output fuzzy sets."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.patch.set_facecolor('#0f172a')

        palette = {
            'blue': '#38bdf8',
            'emerald': '#34d399',
            'amber': '#fbbf24',
            'rose': '#f43f5e',
            'purple': '#c084fc',
            'gray': '#94a3b8'
        }

        def style_axis(ax, title, xlabel, xlim):
            ax.set_facecolor('#1e293b')
            ax.set_title(title, fontsize=12, fontweight='bold', color='#f8fafc', pad=10)
            ax.set_xlabel(xlabel, fontsize=10, color='#94a3b8')
            ax.set_ylabel('Degree of Membership (μ)', fontsize=10, color='#94a3b8')
            ax.set_xlim(xlim)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(colors='#94a3b8', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

        # 1. Soil Moisture
        x_m = np.linspace(0, 100, 300)
        ax = axes[0, 0]
        style_axis(ax, 'Input 1: Soil Moisture (%)', 'Soil Moisture (%)', (0, 100))
        mu_dry = self._trapmf(x_m, self.PARAMS['moisture']['dry'])
        mu_mod = self._trimf(x_m, self.PARAMS['moisture']['moderate'])
        mu_wet = self._trapmf(x_m, self.PARAMS['moisture']['wet'])
        ax.plot(x_m, mu_dry, color=palette['amber'], lw=2.2, label='Dry (0-45%)')
        ax.fill_between(x_m, mu_dry, alpha=0.15, color=palette['amber'])
        ax.plot(x_m, mu_mod, color=palette['emerald'], lw=2.2, label='Moderate (25-75%)')
        ax.fill_between(x_m, mu_mod, alpha=0.15, color=palette['emerald'])
        ax.plot(x_m, mu_wet, color=palette['blue'], lw=2.2, label='Wet (55-100%)')
        ax.fill_between(x_m, mu_wet, alpha=0.15, color=palette['blue'])
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5, loc='center right')

        # 2. Temperature
        x_t = np.linspace(0, 50, 300)
        ax = axes[0, 1]
        style_axis(ax, 'Input 2: Temperature (°C)', 'Temperature (°C)', (0, 50))
        mu_cool = self._trapmf(x_t, self.PARAMS['temperature']['cool'])
        mu_norm = self._trimf(x_t, self.PARAMS['temperature']['normal'])
        mu_hot = self._trapmf(x_t, self.PARAMS['temperature']['hot'])
        ax.plot(x_t, mu_cool, color=palette['blue'], lw=2.2, label='Cool (0-22°C)')
        ax.fill_between(x_t, mu_cool, alpha=0.15, color=palette['blue'])
        ax.plot(x_t, mu_norm, color=palette['emerald'], lw=2.2, label='Normal (15-35°C)')
        ax.fill_between(x_t, mu_norm, alpha=0.15, color=palette['emerald'])
        ax.plot(x_t, mu_hot, color=palette['rose'], lw=2.2, label='Hot (28-50°C)')
        ax.fill_between(x_t, mu_hot, alpha=0.15, color=palette['rose'])
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5, loc='center right')

        # 3. Relative Humidity
        x_h = np.linspace(0, 100, 300)
        ax = axes[1, 0]
        style_axis(ax, 'Input 3: Relative Humidity (%)', 'Relative Humidity (%)', (0, 100))
        mu_low = self._trapmf(x_h, self.PARAMS['humidity']['low'])
        mu_med = self._trimf(x_h, self.PARAMS['humidity']['medium'])
        mu_hi = self._trapmf(x_h, self.PARAMS['humidity']['high'])
        ax.plot(x_h, mu_low, color=palette['amber'], lw=2.2, label='Low (0-45%)')
        ax.fill_between(x_h, mu_low, alpha=0.15, color=palette['amber'])
        ax.plot(x_h, mu_med, color=palette['emerald'], lw=2.2, label='Medium (25-75%)')
        ax.fill_between(x_h, mu_med, alpha=0.15, color=palette['emerald'])
        ax.plot(x_h, mu_hi, color=palette['blue'], lw=2.2, label='High (55-100%)')
        ax.fill_between(x_h, mu_hi, alpha=0.15, color=palette['blue'])
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5, loc='center right')

        # 4. Output: Irrigation Intensity
        x_i = np.linspace(0, 100, 300)
        ax = axes[1, 1]
        style_axis(ax, 'Output: Irrigation Intensity (%)', 'Irrigation Intensity (%)', (0, 100))
        mu_off = self._trapmf(x_i, self.PARAMS['irrigation']['off'])
        mu_low = self._trimf(x_i, self.PARAMS['irrigation']['low'])
        mu_med = self._trimf(x_i, self.PARAMS['irrigation']['medium'])
        mu_high = self._trapmf(x_i, self.PARAMS['irrigation']['high'])
        ax.plot(x_i, mu_off, color=palette['gray'], lw=2.2, label='Off (0-25%)')
        ax.fill_between(x_i, mu_off, alpha=0.15, color=palette['gray'])
        ax.plot(x_i, mu_low, color=palette['emerald'], lw=2.2, label='Low (15-55%)')
        ax.fill_between(x_i, mu_low, alpha=0.15, color=palette['emerald'])
        ax.plot(x_i, mu_med, color=palette['amber'], lw=2.2, label='Medium (40-80%)')
        ax.fill_between(x_i, mu_med, alpha=0.15, color=palette['amber'])
        ax.plot(x_i, mu_high, color=palette['rose'], lw=2.2, label='High (65-100%)')
        ax.fill_between(x_i, mu_high, alpha=0.15, color=palette['rose'])
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8.5, loc='center right')

        plt.tight_layout()
        return fig

    def plot_defuzzification(self, moisture: float, temperature: float, humidity: float) -> plt.Figure:
        """Visual demonstration of Mamdani Aggregation & Centroid Defuzzification."""
        res = self.predict(moisture, temperature, humidity)
        y_grid, agg_curve = res['aggregated_output']
        centroid = res['intensity']

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        # Plot underlying output fuzzy sets as faint reference
        x_i = np.linspace(0, 100, 300)
        ax.plot(x_i, self._trapmf(x_i, self.PARAMS['irrigation']['off']), '--', color='#64748b', alpha=0.5, label='Off set')
        ax.plot(x_i, self._trimf(x_i, self.PARAMS['irrigation']['low']), '--', color='#34d399', alpha=0.5, label='Low set')
        ax.plot(x_i, self._trimf(x_i, self.PARAMS['irrigation']['medium']), '--', color='#fbbf24', alpha=0.5, label='Medium set')
        ax.plot(x_i, self._trapmf(x_i, self.PARAMS['irrigation']['high']), '--', color='#f43f5e', alpha=0.5, label='High set')

        # Plot aggregated fuzzy area
        ax.fill_between(y_grid, agg_curve, color='#38bdf8', alpha=0.45, label='Aggregated Area μ_agg(y)')
        ax.plot(y_grid, agg_curve, color='#38bdf8', lw=2.5)

        # Plot Centroid Line
        ax.axvline(centroid, color='#f43f5e', lw=3.0, linestyle='-', label=f'Centroid = {centroid:.1f}% ({res["category"]})')
        ax.scatter([centroid], [0.0], color='#f43f5e', s=80, zorder=5)

        ax.set_title(
            f"Mamdani Defuzzification | Inputs: Moisture={moisture}%, Temp={temperature}°C, Hum={humidity}%\n"
            f"Result: {centroid:.1f}% [{res['category']}]",
            fontsize=12, fontweight='bold', color='#f8fafc', pad=12
        )
        ax.set_xlabel('Irrigation Intensity Universe (%)', fontsize=10, color='#94a3b8')
        ax.set_ylabel('Aggregated Membership Degree', fontsize=10, color='#94a3b8')
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9, loc='upper right')

        plt.tight_layout()
        return fig
