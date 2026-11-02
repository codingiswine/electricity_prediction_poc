"""CDD/HDD 기준온도 선택 사후 검증

냉난방도일의 기준온도는 표준이 하나로 정해져 있지 않다. 국내만 봐도
한국가스공사 통계는 냉방 24°C / 난방 18°C를 쓰고, 연구에 따라 냉방은
18~28°C 범위에서 고른다. 이 저장소는 평균기온에 18°C 단일 기준을 쓴다.

docs/OPTIMIZATION_ANALYSIS.md가 냉방 24°C(최고기온) / 난방 18°C(최저기온)를
권고하고 있어, 어느 쪽이 실제로 낮은 오차를 내는지 측정한다.

주의: 아래 민감도 표는 2024년 테스트셋 오차를 보고 기준온도를 고르는
형태다. 최저값을 그대로 채택하면 테스트셋 누출이므로 운영 설정은
바꾸지 않았다. 기준온도를 조정하려면 검증 구간을 따로 떼야 한다.

실행: python -m experiments.cdd_base_sweep   (저장소 루트에서)
"""
import warnings

import numpy as np

from src.data_loader import (
    load_monthly_data,
    load_weather_data,
    merge_power_weather,
    train_test_split_by_date,
)
from src.models import calculate_exog_features, fit_sarimax

warnings.filterwarnings("ignore")

# README "Python 모듈 사용 예시"의 Step 2 설정
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)
TEST_START = "2024-01-01"


def evaluate(merged, override=None):
    """기준온도 변형을 적용해 Step 2 설정으로 MAPE/RMSE를 측정한다."""
    df = calculate_exog_features(merged)
    if override is not None:
        df["CDD"], df["HDD"] = override(df)

    train, test = train_test_split_by_date(df, TEST_START)
    train, test = train.dropna(), test.dropna()

    model = fit_sarimax(
        train["value"],
        exog=train[["CDD", "HDD"]],
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
    )
    pred = model.forecast(steps=len(test), exog=test[["CDD", "HDD"]])

    actual = test["value"].values
    mape = 100 * np.mean(np.abs((actual - pred.values) / actual))
    rmse = np.sqrt(np.mean((actual - pred.values) ** 2))
    return mape, rmse


def main():
    merged = merge_power_weather(load_monthly_data(), load_weather_data())

    def report(label, override=None):
        mape, rmse = evaluate(merged, override)
        print(f"  {label:38} MAPE {mape:5.2f}%   RMSE {rmse:7,.0f}")
        return mape

    print("[A] 현재 구현 — README Step 2 재현")
    current = report("CDD 18(평균) / HDD 18(평균)")
    assert abs(current - 5.65) < 0.01, (
        f"Step 2 재현 실패: {current:.2f}% (README 표는 5.65%). "
        "calculate_exog_features / fit_sarimax 변경 여부를 확인할 것."
    )

    print("\n[B] docs/OPTIMIZATION_ANALYSIS.md 권고 (가스공사 통계 기준)")
    report(
        "CDD 24(최고) / HDD 18(최저)",
        lambda d: (
            (d["avg_max_temp"] - 24).clip(lower=0),
            (18 - d["avg_min_temp"]).clip(lower=0),
        ),
    )

    print("\n[C] 냉방 기준온도 민감도 (난방 18°C/평균 고정)")
    print("    ※ 테스트셋을 보고 고르는 비교 — 운영 설정에 반영하지 않는다")
    for base in (18, 20, 22, 24, 26):
        report(
            f"CDD {base}(평균) / HDD 18(평균)",
            lambda d, b=base: (
                np.maximum(0, d["avg_temp"] - b),
                np.maximum(0, 18 - d["avg_temp"]),
            ),
        )


if __name__ == "__main__":
    main()
