"""
공통 유틸리티 함수 모듈

평가 지표 및 헬퍼 함수들을 제공합니다.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) 계산

    Args:
        y_true: 실제 값 (array-like)
        y_pred: 예측 값 (array-like)

    Returns:
        float: MAPE 값 (백분율)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 0으로 나누기 방지
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) 계산

    Args:
        y_true: 실제 값 (array-like)
        y_pred: 예측 값 (array-like)

    Returns:
        float: RMSE 값
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE) 계산

    Args:
        y_true: 실제 값 (array-like)
        y_pred: 예측 값 (array-like)

    Returns:
        float: MAE 값
    """
    return mean_absolute_error(y_true, y_pred)


def eval_metrics(y_true, y_pred, model_name="Model"):
    """
    여러 평가 지표를 한 번에 계산하고 출력

    Args:
        y_true: 실제 값 (array-like)
        y_pred: 예측 값 (array-like)
        model_name: 모델 이름 (str)

    Returns:
        dict: 평가 지표 딕셔너리
    """
    mape_score = mape(y_true, y_pred)
    rmse_score = rmse(y_true, y_pred)
    mae_score = mae(y_true, y_pred)

    metrics = {
        "MAPE": mape_score,
        "RMSE": rmse_score,
        "MAE": mae_score
    }

    print(f"=== {model_name} Evaluation ===")
    print(f"MAPE: {mape_score:.2f}%")
    print(f"RMSE: {rmse_score:.3f}")
    print(f"MAE:  {mae_score:.3f}")

    return metrics


def calculate_cdd_hdd(avg_temp, base_temp=18):
    """
    냉방도일(CDD)와 난방도일(HDD) 계산

    Args:
        avg_temp: 평균 기온 (float or Series)
        base_temp: 기준 온도 (기본값: 18°C)

    Returns:
        tuple: (CDD, HDD)
    """
    cdd = np.maximum(0, avg_temp - base_temp)
    hdd = np.maximum(0, base_temp - avg_temp)
    return cdd, hdd


def save_figure(fig, filename, dpi=300):
    """
    Figure를 figures/ 폴더에 자동 저장

    Args:
        fig: matplotlib figure 객체
        filename: 저장할 파일명 (예: 'sarimax_prediction.png')
        dpi: 이미지 해상도 (기본값: 300)
    """
    from pathlib import Path

    figures_dir = Path(__file__).resolve().parents[1] / 'figures'
    figures_dir.mkdir(exist_ok=True)

    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✅ Figure saved: {filepath}")


def save_results_csv(df, filename):
    """
    결과 DataFrame을 figures/ 폴더에 CSV로 저장

    Args:
        df: 저장할 DataFrame
        filename: 저장할 파일명 (예: 'sarimax_results.csv')
    """
    from pathlib import Path

    figures_dir = Path(__file__).resolve().parents[1] / 'figures'
    figures_dir.mkdir(exist_ok=True)

    filepath = figures_dir / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"✅ Results saved: {filepath}")
