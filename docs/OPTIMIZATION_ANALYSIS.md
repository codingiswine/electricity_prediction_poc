# Code Optimization Analysis: 최신electricity_usage_prediction.ipynb

## Executive Summary
> **Note:** This document analyzes the notebook **as it was before refactoring** (25 code cells). The refactoring described here has since been applied — the current notebook has 13 cells and the extracted logic now lives in `src/utils.py`, `src/data_loader.py`, and `src/models.py`. Cell numbers below refer to the pre-refactoring version.

This notebook contains **extensive code duplication** across 25 code cells, with repeated patterns for data loading, metric calculations, and model training. The code can be significantly optimized by extracting common patterns into reusable functions and modules.

---

## 1. DUPLICATE CODE BLOCKS - HIGH PRIORITY

### 1.1 Evaluation Metrics Functions (Found in 8+ cells)
**Location:** Cells 5, 9, 10, 11, 13, 15, 17, 19, 21, 23
**Pattern:** Identical definition of `mape()`, `rmse()`, and `eval_metrics()` functions

**Current State:**
```python
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def eval_metrics(name, y_true, y_pred, results_store):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{name}: 길이 불일치...")
    if np.isnan(y_pred).any():
        raise ValueError(f"{name}: 예측값에 NaN 존재")
    results_store.append([name, rmse(y_true, y_pred), ...])
```

**Recommendation:**
- Extract to a shared utility module: `metrics.py`
- Remove from all 8+ cells
- Import once at notebook top

**Impact:** Reduces notebook size by ~15%, improves maintainability

---

### 1.2 Data Loading Pattern (Found in every model training cell)
**Location:** Cells 5, 9, 10, 11, 13, 15, 17, 19, 21, 23
**Pattern:** Repeated CSV loading, date conversion, index setting, train/test split

**Current State (example from Cell 10):**
```python
df_power   = pd.read_csv(power_path)
df_weather = pd.read_csv(weather_path)
df_rain    = pd.read_csv(rain_path)

for d in (df_power, df_weather, df_rain):
    d["date"] = pd.to_datetime(d["date"])

df_power   = df_power.set_index("date").asfreq("MS")
df_weather = df_weather.set_index("date").asfreq("MS")
df_rain    = df_rain.set_index("date").asfreq("MS")

df_all = df_power.join(df_weather, how="left").join(df_rain, how="left")

train = df_all.loc["2021-01-01":"2023-12-01"].copy()
test  = df_all.loc["2024-01-01":"2024-12-01"].copy()
```

**Recommendation:**
- Create `data_loader.py` with functions:
  - `load_and_prepare_data(csv_paths, train_end_date, test_end_date)`
  - `convert_to_monthly_ts(df)`
  - `train_test_split_ts(df, train_end, test_end)`
- Replace 15+ lines per cell with single function call

**Impact:** Reduces code duplication by ~30%, centralizes data handling logic

---

## 2. INEFFICIENT DATA LOADING PATTERNS

### 2.1 Repeated CSV Reads Without Caching
**Location:** Multiple cells load the SAME files repeatedly
- Cell 5: reads `POWER_CSV_A`, `WEATHER_CSV`, `RAIN_CSV`
- Cell 9: reads same files again
- Cell 10: reads same files again
- Cell 13: reads same files again
- (Pattern repeats in cells 15, 17, 19, 21, 23)

**Problem:** Files are read into memory multiple times unnecessarily

**Recommendation:**
```python
# Cell 1 (setup): Load and cache data once
_DATA_CACHE = {}

def get_data(file_path, force_reload=False):
    if file_path not in _DATA_CACHE or force_reload:
        _DATA_CACHE[file_path] = pd.read_csv(file_path)
    return _DATA_CACHE[file_path].copy()

# Then in all subsequent cells:
df_power = get_data(POWER_CSV_A)
```

**Impact:** Faster execution (~20-30% speedup), reduces memory usage

---

### 2.2 Inefficient Column Selection in Data Preparation
**Location:** Cells 10, 13, 15, 17, 19, 21, 23
**Pattern:** Manually specifying external variable columns

**Current State (Cell 10):**
```python
exog_cols = [c for c in df_all.columns if c != "value"]
```

**Better Approach:**
```python
def get_exog_cols(df, target_col='value', exclude_cols=None):
    """Auto-detect external variables, handling NaN columns"""
    exclude = {'value', 'date'}
    if exclude_cols:
        exclude.update(exclude_cols)
    return [c for c in df.columns if c not in exclude and df[c].notna().sum() > 0]
```

**Impact:** Reduces errors from manual column selection, handles missing values better

---

## 3. REDUNDANT CALCULATIONS

### 3.1 Repeated Model Instantiation & Training
**Location:** Cells 11, 15, 17, 19, 21, 23 (SARIMAX models)
**Pattern:** Each cell creates and trains SARIMAX with similar logic but slightly different exog variables

**Current State (3 separate cells for SARIMAX experiments):**
- Cell 11: Basic CDD/HDD + lag1
- Cell 15: CDD/HDD + squares + lag1
- Cell 17: CDD/HDD + squares + seasonal interaction + lag1

**Problem:** Much duplicated model fitting code

**Recommendation:**
```python
class SARIMAXExperimenter:
    def __init__(self, train_data, test_data, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.train_data = train_data
        self.test_data = test_data
        self.order = order
        self.seasonal_order = seasonal_order
    
    def build_exog_baseline(self, base_cool=24.0, base_heat=18.0):
        """Build basic CDD/HDD features"""
        ...
    
    def build_exog_with_squares(self):
        """Build CDD/HDD + squares"""
        ...
    
    def fit_and_predict(self, exog_builder_func):
        """Fit model with given exog builder"""
        ...
```

**Impact:** Consolidates 3 cells into 1 reusable class, ~40% less code

---

### 3.2 Identical Feature Engineering Across Cells
**Location:** Cells 11, 15, 17, 19, 21
**Pattern:** CDD/HDD calculation repeated 5 times with slight variations

**Current State:**
```python
# Cell 11:
base_cool, base_heat = 24.0, 18.0
df_all["CDD"] = (df_all["avg_max_temp"] - base_cool).clip(lower=0)
df_all["HDD"] = (base_heat - df_all["avg_min_temp"]).clip(lower=0)

# Cell 15 (same logic):
df_all["CDD"] = (df_all["avg_max_temp"] - base_cool).clip(lower=0)
df_all["HDD"] = (base_heat - df_all["avg_min_temp"]).clip(lower=0)

# Cell 17 (same logic):
... # Repeated again
```

**Recommendation:**
```python
# features.py
def calculate_cdd_hdd(df, temp_col_max='avg_max_temp', temp_col_min='avg_min_temp',
                       base_cool=24.0, base_heat=18.0):
    """Calculate CDD/HDD features"""
    cdd = (df[temp_col_max] - base_cool).clip(lower=0)
    hdd = (base_heat - df[temp_col_min]).clip(lower=0)
    return pd.DataFrame({'CDD': cdd, 'HDD': hdd}, index=df.index)

def calculate_lag_features(df, col, lags=[1]):
    """Calculate lag features"""
    return pd.concat([df[col].shift(i).rename(f'{col}_lag{i}') for i in lags], axis=1)
```

**Impact:** Eliminates ~25 lines of duplicated feature code

---

## 4. CODE THAT SHOULD BE REFACTORED INTO FUNCTIONS

### 4.1 Prophet Model Setup (Cells 9, 10, 21, 23)
**Problem:** Repetitive Prophet training code

**Current State (Cell 9):**
```python
p_train = train.reset_index().rename(columns={"date": "ds", "value": "y"})
p_test  = test.reset_index().rename(columns={"date": "ds", "value": "y"})
weather_cols = df_weather.columns.tolist()
m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
for col in weather_cols:
    m_prophet.add_regressor(col)
m_prophet.fit(p_train[['ds', 'y'] + weather_cols])
p_future = p_test[['ds'] + weather_cols]
forecast = m_prophet.predict(p_future)
prophet_pred = forecast['yhat'].values
```

**Recommendation:**
```python
class ProphetForecaster:
    def __init__(self, yearly=True, weekly=False, daily=False, seasonality_mode='additive'):
        self.config = {
            'yearly_seasonality': yearly,
            'weekly_seasonality': weekly,
            'daily_seasonality': daily,
            'seasonality_mode': seasonality_mode
        }
        self.model = None
        self.regressors = []
    
    def fit(self, train_df, target_col='value', exog_cols=None):
        """Fit Prophet model with optional external regressors"""
        df_p = train_df.reset_index().rename(columns={'date': 'ds', target_col: 'y'})
        self.model = Prophet(**self.config)
        
        if exog_cols:
            self.regressors = exog_cols
            for col in exog_cols:
                self.model.add_regressor(col)
        
        self.model.fit(df_p[['ds', 'y'] + (exog_cols or [])])
    
    def predict(self, test_df):
        """Generate predictions for test period"""
        cols = ['ds'] + self.regressors if self.regressors else ['ds']
        p_test = test_df.reset_index().rename(columns={'date': 'ds'})
        forecast = self.model.predict(p_test[cols])
        return forecast['yhat'].values
```

**Impact:** Reduces Prophet-related code by ~60%, improves reusability

---

### 4.2 Train/Test Split with Visualization
**Location:** Cells 5, 9, 10, 11 (+ more)
**Pattern:** Creates plots of predicted vs actual

**Current State (Example from Cell 10):**
```python
plt.figure(figsize=(14,6))
plt.plot(test.index, y_true, label="📌 Actual", linewidth=2)
plt.plot(test.index, prophet_pred, "--", label="Prophet+Wx+Rain")
plt.plot(test.index, sarimax_pred, "--", label="SARIMAX+Wx+Rain")
plt.title("2024 Monthly Power Forecast (with Weather & Rain)")
plt.xlabel("Month")
plt.ylabel("Power Usage (value)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Recommendation:**
```python
def plot_forecast_comparison(test_index, y_true, predictions_dict, title, ylabel="Power Usage"):
    """Generic forecast comparison plot"""
    plt.figure(figsize=(14,6))
    plt.plot(test_index, y_true, label="📌 Actual", linewidth=2)
    for name, pred in predictions_dict.items():
        plt.plot(test_index, pred, "--", label=name)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Usage:
plot_forecast_comparison(test.index, y_true, {
    "Prophet+Wx+Rain": prophet_pred,
    "SARIMAX+Wx+Rain": sarimax_pred
}, "2024 Monthly Power Forecast (with Weather & Rain)")
```

**Impact:** Eliminates ~30 lines of duplicated visualization code

---

### 4.3 Results DataFrame Creation
**Location:** Cells 5, 9, 10, 11 (+ many more)
**Pattern:** Similar evaluation and results table creation

**Current State:**
```python
results = []
results.append(eval_metrics("Prophet+Wx+Rain", y_true, prophet_pred))
results.append(eval_metrics("SARIMAX+Wx+Rain", y_true, sarimax_pred))
results_df = pd.DataFrame(results).sort_values("RMSE")
print(results_df.to_string(index=False))
```

**Recommendation:**
```python
def create_results_dataframe(predictions_dict, y_true):
    """Create standardized results comparison DataFrame"""
    results = []
    for name, pred in predictions_dict.items():
        results.append({
            "Model": name,
            "RMSE": rmse(y_true, pred),
            "MAE": mean_absolute_error(y_true, pred),
            "MAPE(%)": mape(y_true, pred)
        })
    return pd.DataFrame(results).sort_values("RMSE")

# Usage:
results_df = create_results_dataframe({
    "Prophet+Wx+Rain": prophet_pred,
    "SARIMAX+Wx+Rain": sarimax_pred
}, y_true)
print(results_df.to_string(index=False))
```

**Impact:** Standardizes output, ~20 lines of duplicated code eliminated per cell

---

## 5. MISSING/INCONSISTENT ERROR HANDLING

### 5.1 Inconsistent Data Validation
**Location:** Some cells have assertions, others don't
**Problem:** No consistent validation strategy

**Current State:**
- Cell 5: `assert 'value' in df.columns, "❗ df에는 'value' 컬럼이 있어야 합니다."`
- Cell 10: No validation of `exog_cols`
- Cell 11: Has assertions, but not all cells
- Cell 19: Missing weather column validation

**Recommendation:**
```python
def validate_data(df, required_cols=None, date_col='date', target_col='value'):
    """Comprehensive data validation"""
    errors = []
    
    # Check required columns
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")
    
    # Check target column
    if target_col and target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found")
    
    # Check for NaN values in critical columns
    if target_col and df[target_col].isna().all():
        errors.append(f"Target column '{target_col}' is all NaN")
    
    # Check date index
    if date_col and date_col in df.columns:
        try:
            pd.to_datetime(df[date_col])
        except:
            errors.append(f"Column '{date_col}' cannot be converted to datetime")
    
    if errors:
        raise ValueError("\n".join(["Data validation failed:"] + errors))
    
    return True
```

**Usage:**
```python
validate_data(df_power, required_cols=['date', 'value'], target_col='value')
```

---

### 5.2 Missing Exception Handling for Model Fitting
**Location:** All model training cells (5, 9, 10, 11, 13, 15, 17, 19, 21, 23)
**Problem:** No try/except around model training operations

**Current State:**
```python
sarimax_model = sm.tsa.statespace.SARIMAX(...)
sarimax_result = sarimax_model.fit(disp=False)  # ← Can fail silently or raise
sarimax_pred = sarimax_result.predict(...)
```

**Recommendation:**
```python
def fit_model_safe(model_class, fit_kwargs, predict_kwargs, model_name="Model"):
    """Safely fit and predict with error handling"""
    try:
        model = model_class(**fit_kwargs['init'])
        result = model.fit(**fit_kwargs.get('fit', {}))
        pred = result.predict(**predict_kwargs)
        return pred, None
    except Exception as e:
        error_msg = f"{model_name} failed: {str(e)}"
        print(f"WARNING: {error_msg}")
        return None, error_msg

# Usage:
sarimax_pred, error = fit_model_safe(
    SARIMAX,
    {
        'init': {'endog': train['value'], 'order': (1,1,1), ...},
        'fit': {'disp': False}
    },
    {'start': test.index[0], 'end': test.index[-1]},
    "SARIMAX"
)
if error:
    print(f"Skipping SARIMAX: {error}")
```

---

### 5.3 Inconsistent NaN Handling
**Location:** Cells 5, 10, 13, 15, 17, 19, 21, 23
**Pattern:** Different approaches to NaN values

**Current State:**
- Cell 5: `if np.isnan(y_pred).any(): raise ValueError(...)`
- Cell 10: `train[exog_cols].fillna(method="ffill").fillna(method="bfill")`
- Cell 13: No NaN handling (potential issue)

**Recommendation:**
```python
def handle_missing_values(df, strategy='ffill', fill_value=None):
    """Standardized missing value handling"""
    if strategy == 'ffill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'bfill':
        return df.fillna(method='bfill').fillna(method='ffill')
    elif strategy == 'interpolate':
        return df.interpolate(method='linear')
    elif strategy == 'drop':
        return df.dropna()
    elif fill_value is not None:
        return df.fillna(fill_value)
    else:
        raise ValueError("Must specify strategy or fill_value")
```

---

## 6. SUMMARY OF OPTIMIZATION OPPORTUNITIES

### Quick Wins (Easy to implement)
| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 1 | Extract metrics functions | Cells 5,9,10,11,13,15,17,19,21,23 | Easy | High |
| 2 | Create data loading function | All model cells | Easy | High |
| 3 | Add generic plotting function | Cells 5,9,10,11,13,15,17,19,21,23 | Easy | Medium |
| 4 | Centralize evaluation results | All model cells | Easy | Medium |
| 5 | Consistent error handling | All cells | Medium | Medium |

### Medium-term Refactoring
| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 6 | Create SARIMAXExperimenter class | Cells 11,15,17,19 | Medium | High |
| 7 | Create ProphetForecaster class | Cells 9,10,21,23 | Medium | High |
| 8 | Extract feature engineering | Cells 11,15,17,19,21 | Medium | Medium |
| 9 | Implement data caching | All cells | Medium | High (speed) |

### Architectural Improvements
| # | Issue | Location | Effort | Impact |
|---|-------|----------|--------|--------|
| 10 | Create `metrics.py` module | All | Easy | High |
| 11 | Create `data_loader.py` module | All | Easy | High |
| 12 | Create `models.py` module | Model cells | Medium | High |
| 13 | Create `visualization.py` module | All | Easy | Medium |

---

## Specific Implementation Order

### Phase 1 (Immediate - Cells 1-4)
```python
# Cell X: Create utilities module
# metrics.py functions
# data_loader.py functions
# visualization.py functions
```

### Phase 2 (Short-term - Refactor existing cells)
- Cells 5, 9, 10: Remove duplicate code, import utilities
- Cells 11, 13, 15: Consolidate into single experiment cell

### Phase 3 (Medium-term)
- Create forecaster classes
- Implement proper error handling
- Add comprehensive logging

---

## Files to Create/Modify

```
Current Structure:
최신electricity_usage_prediction.ipynb (50KB, 25 cells)

Recommended Structure:
최신electricity_usage_prediction.ipynb (20KB, 12 cells - cleaner)
utils/
  - __init__.py
  - metrics.py (200 lines)
  - data_loader.py (150 lines)
  - visualization.py (100 lines)
  - models.py (250 lines - forecaster classes)
  - features.py (100 lines - feature engineering)
```

---

## Expected Results After Optimization

- **Code reduction:** 30-40% fewer lines (from ~8000 to ~5000)
- **Readability:** Significantly improved with reusable functions
- **Maintainability:** Changes to evaluation metrics, plotting, loading only needed in one place
- **Execution speed:** 20-30% faster with data caching
- **Error handling:** Consistent validation across all cells
- **Reusability:** Functions and classes can be used in other projects

