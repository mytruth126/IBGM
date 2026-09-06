# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# 1. 全局字体设置
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

plt.rcParams["font.size"] = 24
plt.rcParams["axes.labelsize"] = 32
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 时间序列
# ============================================================

dates = pd.date_range(
    "2024-01-01",
    periods=20,
    freq="MS"
)


# ============================================================
# 3. Table 4 数据
# ============================================================

true = np.array([
    42.2,
    36.4,
    37.9,
    39.4,
    41.4,
    45.6,
    50.1,
    53.0,
    50.4,
    50.0,
    52.0,
    59.1,
    60.0,
    55.1,
    58.7,
    61.4,
    64.5,
    67.1,
    77.1,
    81.0
])


NDGM = np.array([
    42.20,
    34.80,
    38.02,
    40.92,
    43.53,
    45.87,
    47.97,
    49.86,
    51.56,
    53.09,
    54.46,
    55.69,
    56.80,
    57.80,
    58.69,
    59.49,
    60.22,
    60.87,
    61.45,
    61.97
])


DAGM = np.array([
    42.20,
    36.91,
    38.56,
    40.29,
    42.09,
    43.98,
    45.95,
    48.01,
    50.16,
    52.41,
    54.76,
    57.22,
    59.78,
    62.46,
    65.26,
    68.18,
    71.24,
    74.43,
    77.77,
    81.25
])


# ARIMA 第一个0.00视为无有效预测值，改为 np.nan
ARIMA = np.array([
    np.nan,
    42.20,
    36.40,
    37.90,
    39.40,
    41.40,
    45.60,
    50.10,
    53.00,
    50.40,
    50.00,
    52.00,
    59.10,
    60.00,
    55.10,
    58.70,
    58.70,
    58.70,
    58.70,
    58.70
])


Holt_Winters = np.array([
    36.91,
    38.72,
    40.50,
    42.25,
    43.97,
    45.67,
    47.35,
    48.99,
    50.61,
    52.21,
    53.78,
    55.32,
    56.85,
    58.35,
    59.82,
    61.28,
    62.71,
    64.11,
    65.50,
    66.87
])


KRR = np.array([
    np.nan,
    np.nan,
    np.nan,
    42.77,
    42.48,
    44.24,
    47.59,
    50.99,
    53.56,
    52.35,
    52.98,
    53.90,
    58.57,
    58.68,
    56.65,
    59.94,
    59.33,
    59.46,
    59.88,
    60.02
])


KNN = np.array([
    np.nan,
    np.nan,
    np.nan,
    39.40,
    41.40,
    45.60,
    50.10,
    53.00,
    50.40,
    50.00,
    52.00,
    59.10,
    60.00,
    55.10,
    58.70,
    58.70,
    55.10,
    58.70,
    58.70,
    55.10
])


IBGM = np.array([
    42.20,
    36.42,
    37.72,
    39.82,
    40.20,
    42.24,
    44.83,
    46.83,
    55.26,
    54.45,
    52.66,
    57.81,
    60.78,
    57.71,
    62.50,
    63.62,
    67.84,
    68.30,
    70.59,
    80.49
])


# ============================================================
# 4. Forecast MAPE (%)
# ============================================================

forecast_mape = {
    "NDGM": 12.56,
    "DAGM": 6.72,
    "ARIMA": 15.46,
    "Holt-Winters": 7.98,
    "KRR": 14.00,
    "KNN": 17.47,
    "IBGM": 3.93
}


# ============================================================
# 5. 创建主图
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 10),
    dpi=200
)


# ============================================================
# 6. 绘制各模型
# ============================================================

line_true, = ax.plot(
    dates,
    true,
    marker="s",
    linewidth=3.2,
    markersize=8.5,
    label="True data"
)

line_ndgm, = ax.plot(
    dates,
    NDGM,
    marker="^",
    linewidth=2.3,
    markersize=7,
    label="NDGM"
)

line_dagm, = ax.plot(
    dates,
    DAGM,
    marker="D",
    linewidth=2.3,
    markersize=7,
    label="DAGM"
)

line_arima, = ax.plot(
    dates,
    ARIMA,
    marker="o",
    linewidth=2.3,
    markersize=7,
    label="ARIMA"
)

line_hw, = ax.plot(
    dates,
    Holt_Winters,
    marker="v",
    linewidth=2.3,
    markersize=7,
    label="Holt-Winters"
)

line_krr, = ax.plot(
    dates,
    KRR,
    marker="P",
    linewidth=2.3,
    markersize=7,
    label="KRR"
)

line_knn, = ax.plot(
    dates,
    KNN,
    marker="X",
    linewidth=2.3,
    markersize=7,
    label="KNN"
)

line_ibgm, = ax.plot(
    dates,
    IBGM,
    marker="<",
    linewidth=3.2,
    markersize=8.5,
    label="IBGM"
)


# ============================================================
# 7. 训练集和测试集分界线
# ============================================================

split_date = pd.Timestamp("2025-03-16")

ax.axvline(
    split_date,
    linestyle=(0, (5, 4)),
    linewidth=2.0
)


# ============================================================
# 8. 坐标轴设置
# ============================================================

ax.set_xlabel(
    "Date",
    fontsize=32,
    labelpad=16
)

ax.set_ylabel(
    "Charging Volume of Public Charging Facilities in China",
    fontsize=24,
    labelpad=18
)


# 纵坐标调整为 30–85
ax.set_ylim(
    30,
    85
)

# 每10一个刻度
ax.set_yticks(
    np.arange(30, 86, 10)
)


ax.set_xlim(
    dates[0] - pd.Timedelta(days=12),
    dates[-1] + pd.Timedelta(days=12)
)


# ============================================================
# 9. X轴时间格式
# ============================================================

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=24,
    width=1.8,
    length=7
)


# ============================================================
# 10. 图例
# ============================================================

legend = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.0010, 0.9985),

    ncol=4,
    frameon=True,
    fancybox=False,
    edgecolor="black",

    fontsize=24,

    columnspacing=1.2,
    handlelength=2.0,
    handletextpad=0.5,
    borderpad=0.6,
    labelspacing=0.6
)

legend.get_frame().set_linewidth(1.5)


# ============================================================
# 11. Training data / Testing data
# ============================================================

ax.text(
    pd.Timestamp("2025-01-10"),
    69,
    "Training data",
    fontsize=26,
    ha="center",
    va="center"
)

ax.text(
    pd.Timestamp("2025-05-10"),
    69,
    "Testing data",
    fontsize=26,
    ha="center",
    va="center"
)


# ============================================================
# 12. 双向箭头
# ============================================================

ax.annotate(
    "",
    xy=(
        pd.Timestamp("2025-02-05"),
        66
    ),
    xytext=(
        pd.Timestamp("2025-04-20"),
        66
    ),
    arrowprops=dict(
        arrowstyle="<->",
        linewidth=2.0
    )
)


# ============================================================
# 13. 右下角 Forecast MAPE 柱状图
# ============================================================

axins = inset_axes(
    ax,

    width="29%",
    height="23%",

    loc="lower right",

    # 第二个数字控制上下位置
    # 越大越往上
    bbox_to_anchor=(
        0,
        0.085,
        0.965,
        0.965
    ),

    bbox_transform=ax.transAxes,
    borderpad=0.5
)


# ============================================================
# 14. 柱状图数据
# ============================================================

bar_names = [
    "NDGM",
    "DAGM",
    "ARIMA",
    "H-W",
    "KRR",
    "KNN",
    "IBGM"
]

source_names = [
    "NDGM",
    "DAGM",
    "ARIMA",
    "Holt-Winters",
    "KRR",
    "KNN",
    "IBGM"
]

bar_values = [
    forecast_mape[name]
    for name in source_names
]


# ============================================================
# 15. 柱状图颜色与曲线一致
# ============================================================

line_dict = {
    "NDGM": line_ndgm,
    "DAGM": line_dagm,
    "ARIMA": line_arima,
    "Holt-Winters": line_hw,
    "KRR": line_krr,
    "KNN": line_knn,
    "IBGM": line_ibgm
}

bar_colors = [
    line_dict[name].get_color()
    for name in source_names
]


# ============================================================
# 16. 绘制柱状图
# ============================================================

x_bar = np.arange(
    len(bar_names)
)

bars = axins.bar(
    x_bar,
    bar_values,
    width=0.62,
    color=bar_colors
)


# ============================================================
# 17. 小图标题
# ============================================================

axins.set_title(
    "Forecast MAPE (%)",
    fontsize=22,
    pad=5
)


# ============================================================
# 18. 小图 X 轴
# ============================================================

axins.set_xticks(
    x_bar
)

axins.set_xticklabels(
    bar_names,
    rotation=38,
    ha="right",
    fontsize=18
)


# ============================================================
# 19. 小图 Y 轴
# ============================================================

axins.set_ylim(
    0,
    21
)

axins.tick_params(
    axis="y",
    labelsize=16,
    width=1.2,
    length=4
)


# ============================================================
# 20. 柱顶显示 MAPE
# ============================================================

for bar, value in zip(
    bars,
    bar_values
):

    axins.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.35,

        f"{value:.2f}",

        ha="center",
        va="bottom",
        fontsize=18
    )


# ============================================================
# 21. 边框设置
# ============================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.8)

for spine in axins.spines.values():
    spine.set_linewidth(1.3)


# ============================================================
# 22. 整体边距
# ============================================================

fig.subplots_adjust(
    left=0.14,
    right=0.985,
    bottom=0.14,
    top=0.975
)


# ============================================================
# 23. 保存为 300 dpi TIFF
# ============================================================

tiff_file = "Public_Charging_Volume_Forecasting.tiff"

fig.savefig(
    tiff_file,
    format="tiff",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ============================================================
# 24. 同时保存 PNG
# ============================================================

fig.savefig(
    "Public_Charging_Volume_Forecasting.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ============================================================
# 25. 显示图片
# ============================================================

plt.show()

print("TIFF图片已保存：", tiff_file)