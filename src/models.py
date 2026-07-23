"""
시계열 예측 모델 모듈

SARIMAX, Prophet 등 시계열 예측 모델 관련 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')


def fit_sarimax(endog, exog=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), **kwargs):
    """
    SARIMAX 모델 학습

    Args:
        endog: 종속 변수 (시계열 데이터)
        exog: 외생 변수 (선택)
        order: (p, d, q) 파라미터 (기본값: (1,1,1))
        seasonal_order: (P, D, Q, m) 파라미터 (기본값: (1,1,1,12))
        **kwargs: SARIMAX 추가 파라미터

    Returns:
        SARIMAXResults: 학습된 SARIMAX 모델
    """
    model = SARIMAX(
        endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        **kwargs
    )

    result = model.fit(disp=False)
    return result


def predict_with_sarimax(model, steps, exog=None):
    """
    SARIMAX 모델로 예측

    Args:
        model: 학습된 SARIMAX 모델
        steps: 예측 기간
        exog: 예측용 외생 변수 (선택)

    Returns:
        pd.Series: 예측 값
    """
    if exog is not None:
        forecast = model.forecast(steps=steps, exog=exog)
    else:
        forecast = model.forecast(steps=steps)

    return forecast


def fit_prophet(df, y_col='value', date_col='date', **kwargs):
    """
    Prophet 모델 학습

    Args:
        df: 데이터프레임 (date, value 컬럼 필요)
        y_col: 예측 대상 컬럼명 (기본값: 'value')
        date_col: 날짜 컬럼명 (기본값: 'date')
        **kwargs: Prophet 추가 파라미터

    Returns:
        Prophet: 학습된 Prophet 모델
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet가 설치되지 않았습니다. 'pip install prophet' 실행하세요.")

    # Prophet 형식으로 데이터 변환
    prophet_df = df[[date_col, y_col]].rename(
        columns={date_col: 'ds', y_col: 'y'}
    )

    model = Prophet(**kwargs)
    model.fit(prophet_df)

    return model


def predict_with_prophet(model, periods, freq='MS'):
    """
    Prophet 모델로 예측

    Args:
        model: 학습된 Prophet 모델
        periods: 예측 기간
        freq: 주기 ('MS': 월초, 'D': 일별 등)

    Returns:
        pd.DataFrame: 예측 결과
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast


def sarimax_grid_search(endog, exog=None, pdq_list=None, seasonal_pdq_list=None, seasonal_period=12):
    """
    SARIMAX 하이퍼파라미터 그리드 서치

    Args:
        endog: 종속 변수 (시계열 데이터)
        exog: 외생 변수 (선택)
        pdq_list: (p, d, q) 조합 리스트
        seasonal_pdq_list: (P, D, Q) 조합 리스트
        seasonal_period: 계절 주기 (기본값: 12)

    Returns:
        list: 각 조합의 AIC/BIC를 포함한 결과 리스트
    """
    if pdq_list is None:
        pdq_list = [(0,1,0), (1,1,0), (0,1,1), (1,1,1)]

    if seasonal_pdq_list is None:
        seasonal_pdq_list = [(0,1,0), (1,1,0), (0,1,1), (1,1,1)]

    results = []

    for order in pdq_list:
        for sorder in seasonal_pdq_list:
            try:
                seasonal_order = (*sorder, seasonal_period)
                model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order)
                fit = model.fit(disp=False)

                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fit.aic,
                    'bic': fit.bic,
                    'params': len(fit.params)
                })
            except Exception as e:
                # 수렴 실패 등의 경우 건너뛰기
                continue

    # AIC 기준 정렬
    results = sorted(results, key=lambda x: x['aic'])

    return results


def calculate_exog_features(df, temp_col='avg_max_temp', min_temp_col='avg_min_temp'):
    """
    SARIMAX용 외생변수(CDD, HDD, lag 등) 계산

    Args:
        df: 데이터프레임
        temp_col: 평균 최고 기온 컬럼명
        min_temp_col: 평균 최저 기온 컬럼명

    Returns:
        pd.DataFrame: 외생변수가 추가된 데이터프레임
    """
    df = df.copy()

    # 평균 기온 계산 (최고+최저)/2
    if temp_col in df.columns and min_temp_col in df.columns:
        df['avg_temp'] = (df[temp_col] + df[min_temp_col]) / 2
    elif temp_col in df.columns:
        df['avg_temp'] = df[temp_col]

    # CDD/HDD 계산 (기준 온도 18°C)
    base_temp = 18
    df['CDD'] = np.maximum(0, df['avg_temp'] - base_temp)
    df['HDD'] = np.maximum(0, base_temp - df['avg_temp'])

    # Lag 변수 (이전 달 전력 사용량)
    if 'value' in df.columns:
        df['lag1'] = df['value'].shift(1)

    # 제곱항
    df['CDD_sq'] = df['CDD'] ** 2
    df['HDD_sq'] = df['HDD'] ** 2
    if 'lag1' in df.columns:
        df['lag1_sq'] = df['lag1'] ** 2

    # 계절 상호작용 (월 정보 필요)
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['CDD_month'] = df['CDD'] * df['month']
        df['HDD_month'] = df['HDD'] * df['month']

    return df
