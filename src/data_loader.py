"""
데이터 로딩 모듈

전력 사용량 및 기상 데이터를 로드하고 전처리하는 함수들을 제공합니다.
"""

import pandas as pd
from pathlib import Path


# 데이터 디렉토리 경로
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _read_csv(filename: str):
    """
    CSV 파일을 읽고 BOM 제거 및 컬럼명 정규화

    Args:
        filename: CSV 파일명

    Returns:
        pd.DataFrame: 정규화된 데이터프레임
    """
    path = DATA_DIR / filename
    # BOM 제거를 위해 utf-8-sig 사용
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 컬럼명 정규화 (소문자 + 공백 제거)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_monthly_data(filename="monthly_data.csv", parse_dates=True):
    """
    월별 전력 사용량 데이터 로드

    Args:
        filename: CSV 파일명 (기본값: monthly_data.csv)
        parse_dates: 날짜 컬럼을 datetime으로 변환할지 여부

    Returns:
        pd.DataFrame: 전력 사용량 데이터프레임
    """
    df = _read_csv(filename)

    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    return df


# 하위 호환성을 위한 별칭
load_electricity_data = load_monthly_data


def load_weather_data(filename="monthly_weather.csv", parse_dates=True):
    """
    월별 기상 데이터 로드

    Args:
        filename: CSV 파일명 (기본값: monthly_weather.csv)
        parse_dates: 날짜 컬럼을 datetime으로 변환할지 여부

    Returns:
        pd.DataFrame: 기상 데이터프레임
    """
    df = _read_csv(filename)

    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    return df


def load_rain_data(filename="monthly_rain.csv", parse_dates=True):
    """
    월별 강수량 데이터 로드

    Args:
        filename: CSV 파일명 (기본값: monthly_rain.csv)
        parse_dates: 날짜 컬럼을 datetime으로 변환할지 여부

    Returns:
        pd.DataFrame: 강수량 데이터프레임
    """
    df = _read_csv(filename)

    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    return df


def load_daily_weather_data(filename="daily_weather.csv", parse_dates=True):
    """
    일별 기상 데이터 로드

    Args:
        filename: CSV 파일명 (기본값: daily_weather.csv)
        parse_dates: 날짜 컬럼을 datetime으로 변환할지 여부

    Returns:
        pd.DataFrame: 일별 기상 데이터프레임
    """
    df = _read_csv(filename)

    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    return df


def merge_power_weather(power_df, weather_df, on='date'):
    """
    전력 데이터와 기상 데이터 병합

    Args:
        power_df: 전력 데이터프레임
        weather_df: 기상 데이터프레임
        on: 병합 기준 컬럼 (기본값: 'date')

    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    merged = pd.merge(power_df, weather_df, on=on, how='left')
    return merged


def train_test_split_by_date(df, test_start_date='2024-01-01'):
    """
    날짜 기준으로 train/test 분할

    Args:
        df: 데이터프레임
        test_start_date: 테스트 시작 날짜 (str or datetime)

    Returns:
        tuple: (train_df, test_df)
    """
    test_start = pd.to_datetime(test_start_date)
    train = df[df['date'] < test_start].copy()
    test = df[df['date'] >= test_start].copy()

    return train, test
