# Code Optimization Examples - Ready to Implement

## File 1: metrics.py
Create a new file with all evaluation functions to replace duplicates across cells.

```python
# metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class EvaluationMetrics:
    """Centralized evaluation metrics for time series forecasting"""
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        eps = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def validate_predictions(y_true, y_pred, model_name="Model"):
        """Validate prediction arrays before evaluation"""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"{model_name}: Length mismatch - "
                f"y_true({len(y_true)}) vs y_pred({len(y_pred)})"
            )
        
        if np.isnan(y_pred).any():
            raise ValueError(f"{model_name}: Predictions contain NaN values")
        
        if np.isnan(y_true).any():
            raise ValueError(f"{model_name}: Actual values contain NaN")
        
        return True
    
    @staticmethod
    def evaluate(model_name, y_true, y_pred):
        """Comprehensive evaluation returning dict"""
        EvaluationMetrics.validate_predictions(y_true, y_pred, model_name)
        
        return {
            "Model": model_name,
            "RMSE": EvaluationMetrics.rmse(y_true, y_pred),
            "MAE": EvaluationMetrics.mae(y_true, y_pred),
            "MAPE(%)": EvaluationMetrics.mape(y_true, y_pred)
        }

# Usage in notebook:
# from metrics import EvaluationMetrics as EM
# result = EM.evaluate("Prophet", y_true, prophet_pred)
```

---

## File 2: data_loader.py
Centralize all data loading and preprocessing logic.

```python
# data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class DataLoader:
    """Centralized data loading and preprocessing"""
    
    _cache = {}  # Simple caching mechanism
    
    @staticmethod
    def load_csv(file_path: str, force_reload: bool = False) -> pd.DataFrame:
        """Load CSV with caching"""
        if file_path not in DataLoader._cache or force_reload:
            DataLoader._cache[file_path] = pd.read_csv(file_path)
        return DataLoader._cache[file_path].copy()
    
    @staticmethod
    def convert_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Convert date column to datetime"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        return df
    
    @staticmethod
    def to_monthly_ts(df: pd.DataFrame, 
                      date_col: str = 'date',
                      freq: str = 'MS') -> pd.DataFrame:
        """Convert to monthly time series with date index"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.asfreq(freq)  # Enforce monthly frequency
        return df
    
    @staticmethod
    def merge_datasets(data_dict: dict, 
                      how: str = 'left',
                      on: Optional[str] = None) -> pd.DataFrame:
        """Merge multiple dataframes"""
        dfs = list(data_dict.values())
        result = dfs[0]
        
        for df in dfs[1:]:
            if on:
                result = result.merge(df, on=on, how=how)
            else:
                result = result.join(df, how=how)
        
        return result
    
    @staticmethod
    def train_test_split(df: pd.DataFrame,
                        train_end: str,
                        test_end: str,
                        target_col: str = 'value') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train and test periods"""
        train = df.loc[:train_end].copy()
        test = df.loc[train_end:test_end].copy()
        
        # Remove overlap
        test = test[test.index > pd.to_datetime(train_end)]
        
        return train, test
    
    @staticmethod
    def load_and_prepare(csv_paths: dict,
                        train_end: str = "2023-12-01",
                        test_end: str = "2024-12-01",
                        freq: str = 'MS') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """One-line function to load, merge, and split data"""
        # Load all CSVs
        dfs = {}
        for name, path in csv_paths.items():
            df = DataLoader.load_csv(path)
            df = DataLoader.convert_dates(df)
            df = DataLoader.to_monthly_ts(df, freq=freq)
            dfs[name] = df
        
        # Merge (assuming first is primary)
        keys = list(dfs.keys())
        merged = dfs[keys[0]]
        for key in keys[1:]:
            merged = merged.join(dfs[key], how='left')
        
        # Split
        train, test = DataLoader.train_test_split(
            merged, train_end, test_end
        )
        
        return train, test

# Usage in notebook:
# from data_loader import DataLoader
# train, test = DataLoader.load_and_prepare({
#     'power': POWER_CSV_A,
#     'weather': WEATHER_CSV,
#     'rain': RAIN_CSV
# })
```

---

## File 3: visualization.py
Centralize all visualization code.

```python
# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Union

class Visualizer:
    """Centralized visualization utilities"""
    
    @staticmethod
    def plot_forecast_comparison(test_index,
                                 y_true,
                                 predictions_dict: Dict[str, pd.Series],
                                 title: str,
                                 ylabel: str = "Power Usage",
                                 figsize: tuple = (14, 6)):
        """Plot actual vs predicted for multiple models"""
        plt.figure(figsize=figsize)
        plt.plot(test_index, y_true, 
                label="📌 Actual", linewidth=2, color='black')
        
        for name, pred in predictions_dict.items():
            plt.plot(test_index, pred, linestyle="--", label=name)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Month", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_error_comparison(predictions_dict: Dict[str, pd.Series],
                             y_true,
                             test_index,
                             title: str = "Model Comparison - Absolute Error",
                             figsize: tuple = (10, 4)):
        """Plot absolute error comparison"""
        abs_err = pd.DataFrame({
            name: (y_true - pred).abs() 
            for name, pred in predictions_dict.items()
        }, index=test_index)
        
        abs_err.mean().sort_values().plot(
            kind='bar', figsize=figsize, title=title, color='skyblue'
        )
        plt.ylabel('MAE (Monthly Average)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(history,
                             figsize: tuple = (12, 4)):
        """Plot neural network training history"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Usage in notebook:
# from visualization import Visualizer
# Visualizer.plot_forecast_comparison(test.index, y_true, {
#     "Prophet": prophet_pred,
#     "SARIMAX": sarimax_pred
# }, "2024 Forecasts")
```

---

## File 4: models.py
Create reusable forecaster classes.

```python
# models.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from typing import Optional, Dict, Tuple

class ProphetForecaster:
    """Wrapper for Prophet with standardized interface"""
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = False,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'additive'):
        self.config = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'seasonality_mode': seasonality_mode
        }
        self.model = None
        self.regressors = []
    
    def fit(self, train_df: pd.DataFrame,
            target_col: str = 'value',
            exog_cols: Optional[list] = None):
        """Fit Prophet with optional external regressors"""
        df_p = train_df.reset_index().rename(
            columns={'date': 'ds', target_col: 'y'}
        )
        
        self.model = Prophet(**self.config)
        self.regressors = exog_cols or []
        
        if exog_cols:
            for col in exog_cols:
                self.model.add_regressor(col)
        
        cols_to_fit = ['ds', 'y'] + self.regressors
        self.model.fit(df_p[cols_to_fit])
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for test period"""
        cols = ['ds'] + self.regressors if self.regressors else ['ds']
        p_test = test_df.reset_index().rename(columns={'date': 'ds'})
        
        forecast = self.model.predict(p_test[cols])
        return forecast['yhat'].values


class SARIMAXForecaster:
    """Wrapper for SARIMAX with standardized interface"""
    
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.result = None
    
    def fit(self, train_df: pd.DataFrame,
            target_col: str = 'value',
            exog_cols: Optional[pd.DataFrame] = None):
        """Fit SARIMAX model"""
        self.model = sm.tsa.statespace.SARIMAX(
            train_df[target_col],
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=exog_cols,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        
        self.result = self.model.fit(disp=False)
    
    def predict(self, test_df: pd.DataFrame,
               exog_cols: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions for test period"""
        predictions = self.result.predict(
            start=test_df.index[0],
            end=test_df.index[-1],
            exog=exog_cols
        )
        return predictions.values


class ExogenousVariableBuilder:
    """Build external variables for regression models"""
    
    @staticmethod
    def cdd_hdd(df: pd.DataFrame,
               temp_col_max: str = 'avg_max_temp',
               temp_col_min: str = 'avg_min_temp',
               base_cool: float = 24.0,
               base_heat: float = 18.0) -> pd.DataFrame:
        """Calculate cooling and heating degree days"""
        cdd = (df[temp_col_max] - base_cool).clip(lower=0)
        hdd = (base_heat - df[temp_col_min]).clip(lower=0)
        return pd.DataFrame({'CDD': cdd, 'HDD': hdd}, index=df.index)
    
    @staticmethod
    def lag_features(series: pd.Series,
                    lags: list = [1]) -> pd.DataFrame:
        """Create lag features"""
        return pd.concat([
            series.shift(i).rename(f'{series.name}_lag{i}') 
            for i in lags
        ], axis=1)
    
    @staticmethod
    def rolling_features(series: pd.Series,
                        windows: list = [3, 6, 12]) -> pd.DataFrame:
        """Create rolling average features"""
        return pd.concat([
            series.shift(1).rolling(w).mean().rename(f'{series.name}_roll{w}')
            for w in windows
        ], axis=1)
    
    @staticmethod
    def polynomial_features(series: pd.Series,
                           degrees: list = [2]) -> pd.DataFrame:
        """Create polynomial features"""
        return pd.concat([
            (series ** d).rename(f'{series.name}_power{d}')
            for d in degrees
        ], axis=1)

# Usage in notebook:
# from models import ProphetForecaster, SARIMAXForecaster, ExogenousVariableBuilder
#
# # Build features
# exog = ExogenousVariableBuilder.cdd_hdd(df)
# exog = pd.concat([exog, ExogenousVariableBuilder.lag_features(exog['CDD'])], axis=1)
#
# # Prophet
# prophet = ProphetForecaster(yearly_seasonality=True)
# prophet.fit(train, exog_cols=exog.columns.tolist())
# prophet_pred = prophet.predict(test)
#
# # SARIMAX
# sarimax = SARIMAXForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
# sarimax.fit(train, exog_cols=exog.loc[train.index])
# sarimax_pred = sarimax.predict(test, exog_cols=exog.loc[test.index])
```

---

## Before & After: Cell Reduction Example

### BEFORE (Cell 9 - ~80 lines)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 로드
power_path = POWER_CSV_A
weather_path = WEATHER_CSV
df_power = pd.read_csv(power_path)
df_weather = pd.read_csv(weather_path)

# 날짜 처리
df_power['date'] = pd.to_datetime(df_power['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

# 인덱스 세팅
df_power = df_power.set_index('date').asfreq('MS')
df_weather = df_weather.set_index('date').asfreq('MS')

# 데이터 병합
df_all = df_power.join(df_weather, how='left')

# 학습/테스트 분리
train = df_all.loc["2021-01-01":"2023-12-01"].copy()
test = df_all.loc["2024-01-01":"2024-12-01"].copy()

# 평가 함수들
def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def eval_metrics(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred)
    }

# Prophet 모델
p_train = train.reset_index().rename(columns={"date": "ds", "value": "y"})
p_test = test.reset_index().rename(columns={"date": "ds", "value": "y"})
weather_cols = df_weather.columns.tolist()
m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
for col in weather_cols:
    m_prophet.add_regressor(col)
m_prophet.fit(p_train[['ds', 'y'] + weather_cols])
p_future = p_test[['ds'] + weather_cols]
forecast = m_prophet.predict(p_future)
prophet_pred = forecast['yhat'].values

# SARIMAX 모델
sarimax_model = sm.tsa.statespace.SARIMAX(
    train['value'],
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    exog=train[weather_cols],
    enforce_stationarity=False,
    enforce_invertibility=False)
sarimax_result = sarimax_model.fit(disp=False)
sarimax_pred = sarimax_result.predict(
    start=test.index[0],
    end=test.index[-1],
    exog=test[weather_cols]).values

# 성능 비교
y_true = test['value'].values
results = []
results.append(eval_metrics("Prophet+Weather", y_true, prophet_pred))
results.append(eval_metrics("SARIMAX+Weather", y_true, sarimax_pred))
results_df = pd.DataFrame(results).sort_values("RMSE")
print(results_df.to_string(index=False))

# 시각화
plt.figure(figsize=(14,6))
plt.plot(test.index, y_true, label='📌 실제', linewidth=2)
plt.plot(test.index, prophet_pred, '--', label='Prophet+Weather')
plt.plot(test.index, sarimax_pred, '--', label='SARIMAX+Weather')
plt.title("2024년 월별 전력 사용량 예측 (날씨 포함)")
plt.xlabel("월")
plt.ylabel("전력 사용량")
plt.legend()
plt.grid(True)
plt.show()
```

### AFTER (Using utility modules - ~20 lines)
```python
from data_loader import DataLoader
from models import ProphetForecaster, SARIMAXForecaster
from metrics import EvaluationMetrics as EM
from visualization import Visualizer
import pandas as pd

# 데이터 한 줄로 로드
train, test = DataLoader.load_and_prepare({
    'power': POWER_CSV_A,
    'weather': WEATHER_CSV
})

# 모델 학습
prophet = ProphetForecaster(yearly_seasonality=True)
prophet.fit(train, exog_cols=['avg_temp', 'avg_max_temp', 'avg_min_temp'])
prophet_pred = prophet.predict(test)

sarimax = SARIMAXForecaster()
sarimax.fit(train, exog_cols=train[['avg_temp', 'avg_max_temp', 'avg_min_temp']])
sarimax_pred = sarimax.predict(test, exog_cols=test[['avg_temp', 'avg_max_temp', 'avg_min_temp']])

# 평가 및 시각화
y_true = test['value'].values
results_df = pd.DataFrame([
    EM.evaluate("Prophet+Weather", y_true, prophet_pred),
    EM.evaluate("SARIMAX+Weather", y_true, sarimax_pred)
]).sort_values("RMSE")
print(results_df.to_string(index=False))

Visualizer.plot_forecast_comparison(test.index, y_true, {
    "Prophet+Weather": prophet_pred,
    "SARIMAX+Weather": sarimax_pred
}, "2024년 월별 전력 사용량 예측 (날씨 포함)")
```

**Result:** 80 lines → 28 lines (65% reduction) while maintaining same functionality

---

## Implementation Checklist

- [ ] Create `metrics.py` with `EvaluationMetrics` class
- [ ] Create `data_loader.py` with `DataLoader` class
- [ ] Create `visualization.py` with `Visualizer` class
- [ ] Create `models.py` with forecaster classes
- [ ] Update Cell 1 (setup) to import all utilities
- [ ] Refactor Cell 5 (Prophet/SARIMAX basics)
- [ ] Refactor Cell 9 (Prophet + Weather)
- [ ] Refactor Cell 10 (Prophet + Weather + Rain)
- [ ] Consolidate Cells 11, 15, 17, 19 (SARIMAX experiments)
- [ ] Refactor remaining cells to use utilities
- [ ] Test all cells run correctly
- [ ] Document new utility modules

