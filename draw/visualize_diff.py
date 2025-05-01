import math

import mscale
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

import draw.scale


def smooth_difficulty_data(difficulties):
    data = np.array(difficulties)
    n = len(data)

    if n < 5:
        return data

    min_window = 20
    max_window = 81
    target_ratio = 0.15

    window_size = int(n * target_ratio) // 2 * 2 + 1
    window_size = np.clip(window_size, min_window, max_window)

    window_size = min(window_size, n)
    if window_size % 2 == 0:
        window_size = max(3, window_size - 1)

    poly_order = min(3, window_size // 2)

    try:
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, int(n * 1.5))
        spline = CubicSpline(x_old, data)
        interpolated = spline(x_new)

        smoothed = savgol_filter(interpolated, window_size, poly_order)

        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(smoothed)), smoothed)
    except Exception as e:
        print(f"Smoothing failed: {str(e)}")
        return data


def generate_major_ticks(start_diff, max_diff):
    major_ticks = [-1, 0, 5, 10]

    rounded_max_diff = round(float(max_diff), 2)
    start = start_diff

    int_part = math.floor(rounded_max_diff)

    if start <= int_part:
        dynamic_ticks = np.arange(start, int_part + 1, 1).round(2)
    else:
        dynamic_ticks = np.array([])

    if dynamic_ticks.size > 0 and dynamic_ticks[-1] < rounded_max_diff:
        dynamic_ticks = np.append(dynamic_ticks, rounded_max_diff)

    major_ticks += dynamic_ticks.tolist()

    return major_ticks


def visualize_difficulty(times, difficulties, song_name, diff_name, save_path="difficulty_curve"):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',  # 优先使用雅黑（支持更全的Unicode）
        'Noto Sans CJK SC',  # 谷歌思源字体
        'SimHei',  # 备选黑体
        'Arial Unicode MS'  # Mac系统字体
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 添加字体回退路径（需要安装相应字体）
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        fm.addfont('C:/Windows/Fonts/msgothic.ttc')  # 添加日语字体支持
    except:
        pass

    # 在代码中显式指定绘图使用的字体
    plt.rcParams.update({
        'figure.titlesize': 14,
        'font.weight': 'medium'
    })
    # 创建数据框架
    df = pd.DataFrame({
        "timestamp": times,
        "difficulty": difficulties
    }).sort_values("timestamp").reset_index(drop=True)

    max_diff = max(df['difficulty'].max(), 13.0)
    max_length = max_diff + 0.1
    start_diff = math.ceil(max_diff * 0.7)

    df['smooth'] = smooth_difficulty_data(df['difficulty'])

    # 可视化设置
    plt.figure(figsize=(14, 7), dpi=120, facecolor='#f5f5f5')
    ax = plt.gca()

    # 应用自定义坐标轴
    ax.set_yscale('difficulty', threshold=10, linear_scale=0.7, exp_scale=3,
                  major_ticks=generate_major_ticks(start_diff, max_diff))

    # 绘制曲线
    line_raw = ax.plot(df['timestamp'], df['difficulty'],
                       alpha=0.2, color='#4d4d4d', label='原始数据')
    line_smooth = ax.plot(df['timestamp'], df['smooth'],
                          linewidth=2.5, color='#e63946', label='平滑曲线')

    # 坐标轴配置
    ax.set_ylim(bottom=min(-2, df['difficulty'].min()),
                top=max_length)
    ax.set_xlim(left=0)

    # 网格线配置
    ax.grid(True, which='major', linestyle='--',
            linewidth=0.8, alpha=0.6, color='#666666')
    ax.grid(True, which='minor', linestyle=':',
            linewidth=0.5, alpha=0.4, color='#999999')

    # 高亮关键区间
    ax.axhspan(start_diff, max_length, facecolor='#ffd700', alpha=0.1)
    ax.annotate('', xy=(0.05, 0.85), xycoords='axes fraction',
                fontsize=12, color='#e76f51')

    # 标签和标题
    ax.set_title(f"{song_name}({diff_name}) - 动态难度曲线",
                 fontsize=16, pad=20,
                 fontweight='bold', color='#2a2a2a')
    ax.set_xlabel("时间（秒）", fontsize=12)
    ax.set_ylabel("难度值", fontsize=12)

    # 时间轴格式化
    def time_formatter(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes}:{seconds:02d}"

    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))

    # 图例和样式优化
    ax.legend(frameon=True, framealpha=0.9,
              loc='upper left', facecolor='white')

    plt.show()

    return df
