# -*- coding: utf-8 -*-
"""
ARIMA + Holt-Winters 小样本时间序列预测

设置：
1. 总样本数：20
2. 前15个作为训练集
3. 后5个作为测试集
4. column = 0/1/2 选择不同数据列

模型：
1. ARIMA
   - 使用训练集内部滚动一步预测
   - 以MAPE最小选择最优(p,d,q)

2. Holt-Winters
   - 只考虑趋势模型
   - Holt Additive Trend
   - Holt Damped Additive Trend
   - 避免退化成Simple Exponential Smoothing

输出：
1. 训练拟合值
2. 测试预测值
3. 训练误差
4. 测试误差
5. 最优参数
6. 拟合和预测图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# ============================================================
# 1. 输入数据
# ============================================================

# ------------------------------------------------------------
# 选择预测的数据列
# ------------------------------------------------------------
# column = 0：第1列
# column = 1：第2列
# column = 2：第3列
# ------------------------------------------------------------

column = 1


# ------------------------------------------------------------
# 原始数据
# ------------------------------------------------------------

raw_data = np.array([

    [607.8,  278.2, 42.2],
    [619.7,  282.6, 36.4],
    [640.3,  290.9, 37.9],
    [663.7,  297.7, 39.4],
    [687.4,  304.9, 41.4],
    [712.2,  312.2, 45.6],
    [739.4,  320.9, 50.1],
    [773.6,  326.3, 53.0],
    [810.4,  332.9, 50.4],
    [849.3,  339.1, 50.0],
    [889.2,  346.0, 52.0],
    [923.9,  357.9, 59.1],
    [945.3,  376.0, 60.0],
    [961.9,  383.2, 55.1],
    [984.9,  390.0, 58.7],
    [1007.3, 399.2, 61.4],
    [1031.8, 408.3, 64.5],
    [1200.4, 409.6, 67.1],
    [1249.4, 420.2, 77.1],
    [1303.2, 431.6, 81.0]

], dtype=float)


# ------------------------------------------------------------
# 当前使用的数据
# ------------------------------------------------------------

data = raw_data[:, column]


# ------------------------------------------------------------
# 最后5个作为测试集
# ------------------------------------------------------------

test_size = 5


# ============================================================
# 2. 划分训练集和测试集
# ============================================================

train = data[:-test_size]

test = data[-test_size:]


print("\n")
print("=" * 80)
print("数据划分")
print("=" * 80)

print(
    "当前选择第 {} 列".format(
        column + 1
    )
)

print(
    "总样本数：",
    len(data)
)

print(
    "训练集样本数：",
    len(train)
)

print(
    "测试集样本数：",
    len(test)
)


print("\n训练集：")

print(train)


print("\n测试集：")

print(test)


# ============================================================
# 3. 评价指标
# ============================================================

def MAPE(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    mask = np.abs(
        y_true
    ) > 1e-10

    return np.mean(

        np.abs(

            (
                y_true[mask]
                -
                y_pred[mask]
            )

            /

            y_true[mask]

        )

    ) * 100


def RMSE(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return np.sqrt(

        np.mean(

            (
                y_true
                -
                y_pred
            ) ** 2

        )

    )


def MAE(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return np.mean(

        np.abs(

            y_true
            -
            y_pred

        )

    )


# ============================================================
# 4. ARIMA滚动一步预测
# ============================================================

def arima_rolling_mape(
        train,
        order,
        min_train_size=8
):

    """
    训练集内部滚动一步预测

    前8个预测第9个
    前9个预测第10个
    ...
    前14个预测第15个

    使用MAPE作为参数优化目标
    """

    actual = []

    prediction = []


    for i in range(
        min_train_size,
        len(train)
    ):

        history = train[:i]

        try:

            model = ARIMA(

                history,

                order=order,

                enforce_stationarity=False,

                enforce_invertibility=False

            )


            model_fit = model.fit()


            pred = model_fit.forecast(
                steps=1
            )[0]


            if np.isfinite(pred):

                actual.append(
                    train[i]
                )

                prediction.append(
                    pred
                )


        except:

            return np.inf


    if len(actual) == 0:

        return np.inf


    return MAPE(
        actual,
        prediction
    )


# ============================================================
# 5. ARIMA参数优化
# ============================================================

def optimize_arima(train):

    print("\n")
    print("=" * 80)
    print("ARIMA参数优化")
    print("=" * 80)


    best_mape = np.inf

    best_order = None

    results = []


    # --------------------------------------------------------
    # 搜索范围
    # --------------------------------------------------------

    p_values = range(
        0,
        4
    )

    d_values = range(
        0,
        3
    )

    q_values = range(
        0,
        4
    )


    for p in p_values:

        for d in d_values:

            for q in q_values:


                # 排除(0,0,0)
                if (
                    p == 0
                    and
                    d == 0
                    and
                    q == 0
                ):

                    continue


                order = (
                    p,
                    d,
                    q
                )


                try:

                    mape = arima_rolling_mape(

                        train,

                        order

                    )


                    if np.isfinite(mape):

                        results.append([

                            p,

                            d,

                            q,

                            mape

                        ])


                        print(

                            "ARIMA{}    Rolling MAPE = {:.4f}%".format(

                                order,

                                mape

                            )

                        )


                        if mape < best_mape:

                            best_mape = mape

                            best_order = order


                except:

                    pass


    if best_order is None:

        raise ValueError(
            "ARIMA参数搜索失败"
        )


    results = pd.DataFrame(

        results,

        columns=[

            "p",

            "d",

            "q",

            "Rolling_MAPE"

        ]

    )


    results = results.sort_values(

        by="Rolling_MAPE"

    ).reset_index(
        drop=True
    )


    print("\n")

    print(
        "最优ARIMA参数：",
        best_order
    )

    print(

        "最优训练滚动MAPE：{:.4f}%".format(
            best_mape
        )

    )


    return (
        best_order,
        results
    )


# ============================================================
# 6. Holt-Winters滚动预测
# ============================================================

def hw_rolling_mape(
        train,
        trend,
        damped_trend,
        min_train_size=8
):

    actual = []

    prediction = []


    for i in range(
        min_train_size,
        len(train)
    ):

        history = train[:i]


        try:

            model = ExponentialSmoothing(

                history,

                trend=trend,

                damped_trend=damped_trend,

                seasonal=None,

                initialization_method="estimated"

            )


            model_fit = model.fit(
                optimized=True
            )


            pred = model_fit.forecast(
                1
            )[0]


            if np.isfinite(pred):

                actual.append(
                    train[i]
                )

                prediction.append(
                    pred
                )


        except:

            return np.inf


    if len(actual) == 0:

        return np.inf


    return MAPE(
        actual,
        prediction
    )


# ============================================================
# 7. Holt-Winters模型优化
# ============================================================

def optimize_hw(train):

    print("\n")
    print("=" * 80)
    print("Holt-Winters趋势模型优化")
    print("=" * 80)


    # --------------------------------------------------------
    # 这里只保留具有趋势项的模型
    # 不再使用Simple Exponential Smoothing
    # --------------------------------------------------------

    candidates = [

        {

            "name":
                "Holt Additive Trend",

            "trend":
                "add",

            "damped":
                False

        },

        {

            "name":
                "Holt Damped Additive Trend",

            "trend":
                "add",

            "damped":
                True

        }

    ]


    best_mape = np.inf

    best_config = None

    results = []


    for config in candidates:

        try:

            mape = hw_rolling_mape(

                train,

                trend=
                    config["trend"],

                damped_trend=
                    config["damped"]

            )


            if np.isfinite(mape):

                results.append([

                    config["name"],

                    config["trend"],

                    config["damped"],

                    mape

                ])


                print(

                    "{:<35s} Rolling MAPE = {:.4f}%".format(

                        config["name"],

                        mape

                    )

                )


                if mape < best_mape:

                    best_mape = mape

                    best_config = config


        except:

            pass


    if best_config is None:

        raise ValueError(
            "Holt-Winters模型优化失败"
        )


    results = pd.DataFrame(

        results,

        columns=[

            "Model",

            "Trend",

            "Damped",

            "Rolling_MAPE"

        ]

    )


    results = results.sort_values(

        by="Rolling_MAPE"

    ).reset_index(
        drop=True
    )


    print("\n")

    print(

        "最优Holt-Winters模型：",
        best_config["name"]

    )


    print(

        "最优训练滚动MAPE：{:.4f}%".format(
            best_mape
        )

    )


    return (
        best_config,
        results
    )


# ============================================================
# 8. 优化ARIMA
# ============================================================

best_arima_order, arima_search_results = optimize_arima(
    train
)


# ============================================================
# 9. 优化Holt-Winters
# ============================================================

best_hw_config, hw_search_results = optimize_hw(
    train
)


# ============================================================
# 10. 建立最终ARIMA模型
# ============================================================

arima_model = ARIMA(

    train,

    order=best_arima_order,

    enforce_stationarity=False,

    enforce_invertibility=False

)


arima_fit = arima_model.fit()


# ------------------------------------------------------------
# 训练集拟合
# ------------------------------------------------------------

arima_train_pred = arima_fit.predict(

    start=0,

    end=len(train)-1

)


arima_train_pred = np.asarray(

    arima_train_pred,

    dtype=float

)


# ------------------------------------------------------------
# 测试集预测
# ------------------------------------------------------------

arima_test_pred = arima_fit.forecast(

    steps=test_size

)


arima_test_pred = np.asarray(

    arima_test_pred,

    dtype=float

)


# ============================================================
# 11. 建立最终Holt-Winters模型
# ============================================================

hw_model = ExponentialSmoothing(

    train,

    trend=
        best_hw_config["trend"],

    damped_trend=
        best_hw_config["damped"],

    seasonal=None,

    initialization_method="estimated"

)


hw_fit = hw_model.fit(
    optimized=True
)


# ------------------------------------------------------------
# 训练拟合
# ------------------------------------------------------------

hw_train_pred = np.asarray(

    hw_fit.fittedvalues,

    dtype=float

)


# ------------------------------------------------------------
# 测试预测
# ------------------------------------------------------------

hw_test_pred = np.asarray(

    hw_fit.forecast(
        test_size
    ),

    dtype=float

)


# ============================================================
# 12. 训练集误差
# ============================================================

# ------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------

arima_train_mask = np.isfinite(
    arima_train_pred
)


arima_train_mape = MAPE(

    train[arima_train_mask],

    arima_train_pred[arima_train_mask]

)


arima_train_rmse = RMSE(

    train[arima_train_mask],

    arima_train_pred[arima_train_mask]

)


arima_train_mae = MAE(

    train[arima_train_mask],

    arima_train_pred[arima_train_mask]

)


# ------------------------------------------------------------
# Holt-Winters
# ------------------------------------------------------------

hw_train_mask = np.isfinite(
    hw_train_pred
)


hw_train_mape = MAPE(

    train[hw_train_mask],

    hw_train_pred[hw_train_mask]

)


hw_train_rmse = RMSE(

    train[hw_train_mask],

    hw_train_pred[hw_train_mask]

)


hw_train_mae = MAE(

    train[hw_train_mask],

    hw_train_pred[hw_train_mask]

)


# ============================================================
# 13. 测试集误差
# ============================================================

# ------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------

arima_test_mape = MAPE(

    test,

    arima_test_pred

)


arima_test_rmse = RMSE(

    test,

    arima_test_pred

)


arima_test_mae = MAE(

    test,

    arima_test_pred

)


# ------------------------------------------------------------
# Holt-Winters
# ------------------------------------------------------------

hw_test_mape = MAPE(

    test,

    hw_test_pred

)


hw_test_rmse = RMSE(

    test,

    hw_test_pred

)


hw_test_mae = MAE(

    test,

    hw_test_pred

)


# ============================================================
# 14. 合并训练拟合 + 测试预测
# ============================================================

arima_all_pred = np.concatenate([

    arima_train_pred,

    arima_test_pred

])


hw_all_pred = np.concatenate([

    hw_train_pred,

    hw_test_pred

])


set_type = (

    ["Train"] * len(train)

    +

    ["Test"] * len(test)

)


# ============================================================
# 15. 输出完整拟合和预测结果
# ============================================================

result = pd.DataFrame({

    "Sample":
        np.arange(
            1,
            len(data)+1
        ),

    "Set":
        set_type,

    "Actual":
        data,

    "ARIMA":
        arima_all_pred,

    "Holt-Winters":
        hw_all_pred

})


print("\n")
print("=" * 100)
print("完整拟合与预测结果")
print("=" * 100)


print(

    result
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 16. 输出训练误差 + 测试误差
# ============================================================

metrics = pd.DataFrame({

    "Model": [

        "ARIMA",

        "Holt-Winters"

    ],

    "Train_MAPE(%)": [

        arima_train_mape,

        hw_train_mape

    ],

    "Train_RMSE": [

        arima_train_rmse,

        hw_train_rmse

    ],

    "Train_MAE": [

        arima_train_mae,

        hw_train_mae

    ],

    "Test_MAPE(%)": [

        arima_test_mape,

        hw_test_mape

    ],

    "Test_RMSE": [

        arima_test_rmse,

        hw_test_rmse

    ],

    "Test_MAE": [

        arima_test_mae,

        hw_test_mae

    ]

})


print("\n")
print("=" * 120)
print("训练集和测试集误差")
print("=" * 120)


print(

    metrics
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 17. 输出模型参数
# ============================================================

print("\n")
print("=" * 80)
print("最优模型参数")
print("=" * 80)


print(

    "ARIMA最优参数 (p,d,q) = {}".format(
        best_arima_order
    )

)


print(

    "Holt-Winters最优模型 = {}".format(
        best_hw_config["name"]
    )

)


print(

    "Holt-Winters trend = {}".format(
        best_hw_config["trend"]
    )

)


print(

    "Holt-Winters damped_trend = {}".format(
        best_hw_config["damped"]
    )

)


# ============================================================
# 18. ARIMA搜索结果
# ============================================================

print("\n")
print("=" * 80)
print("ARIMA参数搜索前10名")
print("=" * 80)


print(

    arima_search_results
    .head(10)
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 19. Holt-Winters搜索结果
# ============================================================

print("\n")
print("=" * 80)
print("Holt-Winters模型搜索结果")
print("=" * 80)


print(

    hw_search_results
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 20. 单独输出预测结果
#     方便复制到Excel
# ============================================================

copy_result = pd.DataFrame({

    "Actual":

        data,

    "ARIMA":

        arima_all_pred,

    "Holt-Winters":

        hw_all_pred

})


print("\n")
print("=" * 80)
print("真实值 + 模型结果")
print("=" * 80)


print(

    copy_result
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 21. 单独输出误差
#     方便复制到论文
# ============================================================

print("\n")
print("=" * 120)
print("模型误差结果")
print("=" * 120)


print(

    metrics
    .round(4)
    .to_string(
        index=False
    )

)


# ============================================================
# 22. 绘图
# ============================================================

x = np.arange(
    1,
    len(data)+1
)


plt.figure(
    figsize=(10, 6)
)


# ------------------------------------------------------------
# 真实值
# ------------------------------------------------------------

plt.plot(

    x,

    data,

    marker="o",

    linewidth=2,

    label="Actual"

)


# ------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------

plt.plot(

    x,

    arima_all_pred,

    marker="s",

    linestyle="--",

    label="ARIMA"

)


# ------------------------------------------------------------
# Holt-Winters
# ------------------------------------------------------------

plt.plot(

    x,

    hw_all_pred,

    marker="^",

    linestyle="--",

    label="Holt-Winters"

)


# ------------------------------------------------------------
# 训练/测试分界线
# ------------------------------------------------------------

plt.axvline(

    x=len(train)+0.5,

    linestyle="--",

    linewidth=1.5

)


plt.xlabel(
    "Sample"
)


plt.ylabel(
    "Value"
)


plt.title(
    "ARIMA and Holt-Winters Fitting and Forecasting Results"
)


plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.show()