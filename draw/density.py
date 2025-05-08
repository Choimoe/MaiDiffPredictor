import json
import csv
import re
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import concurrent.futures
import os
import matplotlib.font_manager as fm
from collections import defaultdict


with open("info/songs.json", "r", encoding="utf-8") as f:
    songs = json.load(f)


def get_song_info(song_id):
    for song in songs:
        if str(song["id"]) == str(song_id):
            return song["name"]
    return "Unknown Song"


diff_name_mapping = {
    "0": "Basic",
    "1": "Advanced",
    "2": "Expert",
    "3": "Master",
    "4": "Re:Master"
}

# --- Font Configuration for Matplotlib ---
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans',
    'Noto Sans CJK JP', 'Noto Sans CJK KR', 'Noto Sans CJK SC', 'Noto Sans CJK TC'
]
plt.rcParams['axes.unicode_minus'] = False

# --- Configuration Constants ---
TOUCH_FILTER_THRESHOLD = 0.1
MAX_TOUCH_PER_GROUP = 2
INST_DENSITY_WINDOW_SIZE = 1.0  # seconds
DENSITY_CALC_TIME_STEP = 0.1  # seconds
AVERAGE_DENSITY_WINDOW_SIZES = [15.0]  # seconds
SKIP_CHART_FRONT_SIZE = 2.0  # seconds
SKIP_CHART_END_SIZE = 1.0  # seconds

NUM_TOP_A_INSTANTANEOUS_PER_LEVEL = 5  # Top A per difficulty level
NUM_TOP_B_AVERAGE_PER_LEVEL = 5  # Top B per difficulty level

OUTPUT_PLOT_DIR = Path("output_plots")
MAX_WORKERS = os.cpu_count()
PROCESS_FIRST_N_CHARTS = None  # Set to e.g., 20 for testing, None for all

# --- NEW: Smoothing configuration for instantaneous density ---
INST_DENSITY_SMOOTHING_POINTS = 5  # Number of data points for the moving average.

# --- Difficulty Level Definitions (based on 'ds' value) ---
DIFFICULTY_LEVELS_DS = {
    "Level_12_DS_12.0-12.6": (12.0, 12.5999),
    "Level_12p_DS_12.6-13.0": (12.6, 12.9999),
    "Level_13_DS_13.0-12.6": (13.0, 13.5999),
    "Level_13p_DS_13.6-14.0": (13.6, 13.9999),
    "Level_14_DS_14.0-14.6": (14.0, 14.5999),
    "Level_14p_DS_14.6-15.0": (14.6, 14.9999),
}


# --- Helper Functions ---

def get_difficulty_level_group(ds_value: float) -> str:
    """Categorizes charts into broader DS-based difficulty groups."""
    if pd.isna(ds_value):
        return "Level_Unknown_DS"
    min_req = 1e4
    max_req = -1
    for level_name, (min_ds, max_ds) in DIFFICULTY_LEVELS_DS.items():
        min_req = min(min_req, min_ds)
        max_req = max(max_req, max_ds)
        if min_ds <= ds_value <= max_ds:
            return level_name
    if ds_value < min_req:
        return "Level_Easy_DS"
    if ds_value > max_req:
        return "Level_Hard_DS"
    return "Level_Other_DS"


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or len(data) < window_size:
        return data
    gaussian_weights = np.exp(-np.linspace(-3, 3, window_size)**2)
    gaussian_weights /= gaussian_weights.sum()
    return np.convolve(data, gaussian_weights, mode='valid')


def load_chart_json(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def preprocess_notes(chart_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    processed_notes = []
    max_time = 0.0
    for time_event in chart_data:
        current_time = time_event.get("Time", 0.0)
        max_time = max(max_time, current_time)
        for note in time_event.get("Notes", []):
            note_type = note.get("noteType")
            sort_time = current_time
            if note_type == "Tap":
                processed_notes.append({"type": "Tap", "time": current_time, "sort_time": sort_time})
            elif note_type == "Touch":
                processed_notes.append({"type": "Touch", "time": current_time, "sort_time": sort_time})
            elif note_type == "Hold":
                hold_time = note.get("holdTime", 0.0)
                end_time = current_time + hold_time
                processed_notes.append(
                    {"type": "Hold", "start_time": current_time, "end_time": end_time, "sort_time": sort_time})
                max_time = max(max_time, end_time)
            elif note_type == "Slide":
                slide_start_time = note.get("slideStartTime", current_time)
                slide_duration = note.get("slideTime", 0.0)
                slide_end_time = slide_start_time + slide_duration
                processed_notes.append({"type": "Slide_Tap", "time": current_time, "sort_time": current_time})
                processed_notes.append(
                    {"type": "Slide_Hold", "start_time": slide_start_time, "end_time": slide_end_time,
                     "sort_time": slide_start_time})
                max_time = max(max_time, slide_end_time, current_time)
    return processed_notes, max_time


def count_notes_in_window(notes: List[Dict[str, Any]], window_start: float, window_end: float) -> int:
    count = 0
    touch_activation_times_in_window = []
    for note in notes:
        note_type = note["type"]
        if note_type in ["Tap", "Slide_Tap"]:
            if window_start <= note["time"] < window_end: count += 1
        elif note_type in ["Hold", "Slide_Hold"]:
            if max(note["start_time"], window_start) < min(note["end_time"], window_end): count += 1
        elif note_type == "Touch":
            if window_start <= note["time"] < window_end: touch_activation_times_in_window.append(note["time"])
    if touch_activation_times_in_window:
        touch_activation_times_in_window.sort()
        idx = 0
        while idx < len(touch_activation_times_in_window):
            current_group_start_time = touch_activation_times_in_window[idx]
            touches_in_group = 0
            group_idx = idx
            while group_idx < len(touch_activation_times_in_window) and \
                    touch_activation_times_in_window[group_idx] < current_group_start_time + TOUCH_FILTER_THRESHOLD:
                touches_in_group += 1
                group_idx += 1
            count += min(touches_in_group, MAX_TOUCH_PER_GROUP)
            idx = group_idx
    return count


def calculate_chart_peaks_worker(chart_file_path_str: str, song_id: str, level_idx_str: str, ds_value_str: str) -> \
Optional[
    Dict[str, Any]]:
    chart_file_path = Path(chart_file_path_str)
    try:
        ds_value = float(ds_value_str) if ds_value_str and ds_value_str.lower() != 'nan' else np.nan
    except ValueError:
        ds_value = np.nan

    difficulty_group = get_difficulty_level_group(ds_value)  # Using DS-based grouping

    chart_data_json = load_chart_json(chart_file_path_str)
    if not chart_data_json: return None

    processed_notes, song_duration = preprocess_notes(chart_data_json)
    if not processed_notes or song_duration <= max(1.0, INST_DENSITY_WINDOW_SIZE,
                                                   max(AVERAGE_DENSITY_WINDOW_SIZES if AVERAGE_DENSITY_WINDOW_SIZES else [
                                                       1.0])):
        return None  # Skip very short or empty charts

    # --- Instantaneous Density Calculation ---
    peak_inst_density_value = -1.0
    peak_inst_density_time_center = -1.0
    peak_inst_time_start = -1.0
    peak_inst_time_end = -1.0

    raw_instantaneous_densities = []
    density_window_center_times = []

    # Effective window for instantaneous density
    inst_effective_window = min(INST_DENSITY_WINDOW_SIZE, song_duration - SKIP_CHART_FRONT_SIZE - SKIP_CHART_END_SIZE)
    if inst_effective_window < 0.5:  # Ensure a minimum sensible window
        inst_effective_window = 0.5
        if song_duration < inst_effective_window + SKIP_CHART_FRONT_SIZE + SKIP_CHART_END_SIZE:  # not enough duration
            # Fallback: use a small portion of song if too short, or skip
            if song_duration > 0.5:
                inst_effective_window = song_duration * 0.8
            else:
                return None  # Cannot process extremely short charts for density

    # Define time centers for instantaneous density calculation
    # Ensure t_centers allow full window within song bounds (considering skips)
    min_center_time = SKIP_CHART_FRONT_SIZE + inst_effective_window / 2
    max_center_time = song_duration - SKIP_CHART_END_SIZE - inst_effective_window / 2

    if min_center_time >= max_center_time:  # Not enough range for centers
        # Attempt a single calculation if possible, or return None
        if song_duration >= inst_effective_window:
            t_centers = np.array([song_duration / 2])
            inst_effective_window = song_duration  # Use full song if forced to single calc
        else:
            return None
    else:
        t_centers = np.arange(
            min_center_time,
            max_center_time + 1e-9,  # include endpoint
            DENSITY_CALC_TIME_STEP
        )
    if not t_centers.any(): return None

    for t_center in t_centers:
        t_start = t_center - inst_effective_window / 2
        t_end = t_center + inst_effective_window / 2
        count = count_notes_in_window(processed_notes, t_start, t_end)
        density = count / inst_effective_window if inst_effective_window > 0 else 0
        raw_instantaneous_densities.append(density)
        density_window_center_times.append(t_center)

    if raw_instantaneous_densities:
        raw_densities_np = np.array(raw_instantaneous_densities)
        smoothed_densities = raw_densities_np  # Default if no smoothing
        times_for_smoothed = np.array(density_window_center_times)

        if INST_DENSITY_SMOOTHING_POINTS > 1 and len(raw_densities_np) >= INST_DENSITY_SMOOTHING_POINTS:
            smoothed_densities = moving_average(raw_densities_np, INST_DENSITY_SMOOTHING_POINTS)
            # Adjust times for smoothed data (center of smoothing window)
            half_smooth_win = (INST_DENSITY_SMOOTHING_POINTS - 1) // 2
            times_for_smoothed = times_for_smoothed[half_smooth_win: len(times_for_smoothed) - half_smooth_win]
            if INST_DENSITY_SMOOTHING_POINTS % 2 == 0 and len(times_for_smoothed) > len(
                    smoothed_densities):  # Adjust for even window
                times_for_smoothed = times_for_smoothed[:len(smoothed_densities)]

        if smoothed_densities.size > 0:
            peak_idx = np.argmax(smoothed_densities)
            peak_inst_density_value = smoothed_densities[peak_idx]
            peak_inst_density_time_center = times_for_smoothed[peak_idx]
            peak_inst_time_start = peak_inst_density_time_center - inst_effective_window / 2
            peak_inst_time_end = peak_inst_density_time_center + inst_effective_window / 2

    # --- Average Density Calculation ---
    peak_avg_density_value = -1.0
    peak_avg_window_used = -1.0
    peak_avg_time_start = -1.0
    peak_avg_time_end = -1.0
    MIN_AVG_WINDOW_DURATION = 3.0  # Minimum duration for an average window to be meaningful

    for avg_win_size in AVERAGE_DENSITY_WINDOW_SIZES:
        if avg_win_size > (
                song_duration - SKIP_CHART_FRONT_SIZE - SKIP_CHART_END_SIZE) or avg_win_size < MIN_AVG_WINDOW_DURATION:
            continue

        current_max_density_for_this_ws = -1.0
        current_peak_t_start_for_this_ws = -1.0
        current_peak_t_end_for_this_ws = -1.0

        # Define time centers for this average window size
        min_avg_center = SKIP_CHART_FRONT_SIZE + avg_win_size / 2
        max_avg_center = song_duration - SKIP_CHART_END_SIZE - avg_win_size / 2

        if min_avg_center >= max_avg_center: continue  # Not enough range

        avg_window_centers = np.arange(min_avg_center, max_avg_center + 1e-9, DENSITY_CALC_TIME_STEP)
        if not avg_window_centers.any(): continue

        for t_center in avg_window_centers:
            t_start = t_center - avg_win_size / 2
            t_end = t_center + avg_win_size / 2
            count = count_notes_in_window(processed_notes, t_start, t_end)
            density = count / avg_win_size if avg_win_size > 0 else 0

            if density > current_max_density_for_this_ws:
                current_max_density_for_this_ws = density
                current_peak_t_start_for_this_ws = t_start
                current_peak_t_end_for_this_ws = t_end

        if current_max_density_for_this_ws > peak_avg_density_value:
            peak_avg_density_value = current_max_density_for_this_ws
            peak_avg_window_used = avg_win_size
            peak_avg_time_start = current_peak_t_start_for_this_ws
            peak_avg_time_end = current_peak_t_end_for_this_ws

    return {
        "full_path": chart_file_path_str,
        "stem": chart_file_path.stem,
        "song_id": song_id,
        "level_idx": level_idx_str,  # Keep original level_idx from CSV
        "ds_value": ds_value,
        "difficulty_group": difficulty_group,  # DS-based group for folder structure

        "peak_inst_val": peak_inst_density_value,
        "peak_inst_time_start": peak_inst_time_start,
        "peak_inst_time_end": peak_inst_time_end,
        "inst_effective_window_used": inst_effective_window,  # Actual window size used

        "peak_avg_val": peak_avg_density_value,
        "peak_avg_time_start": peak_avg_time_start,
        "peak_avg_time_end": peak_avg_time_end,
        "avg_win_for_peak": peak_avg_window_used,  # The window size (e.g., 15s) that gave this peak

        "song_duration": song_duration,
        "raw_inst_densities_for_plot": raw_instantaneous_densities,  # For plotting the curve
        "density_curve_times_for_plot": density_window_center_times  # Times for the raw curve
    }


def worker_wrapper(args_tuple: Tuple[str, str, str, str]) -> Optional[Dict[str, Any]]:
    return calculate_chart_peaks_worker(*args_tuple)


def sanitize_filename_component(text: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    text = str(text)  # Ensure it's a string
    text = re.sub(r'[\\/*?:"<>|]', "", text)  # Remove forbidden characters
    text = text.replace(" ", "_")  # Replace spaces with underscores
    return text[:50]  # Limit length to avoid overly long names


def generate_density_plot(chart_metrics: Dict[str, Any],
                          plot_context: Dict[str, Any],
                          output_dir_for_level: Path):
    # --- Extract data from chart_metrics ---
    chart_stem = chart_metrics["stem"]
    song_id = chart_metrics.get("song_id", "N/A")
    diff_name = chart_metrics.get("diff_name", "UnknownDiff")
    ds_val = chart_metrics.get("ds_value", np.nan)
    ds_str = f"{ds_val:.2f}" if not pd.isna(ds_val) else "N/A"
    song_name = chart_metrics.get("song_name", f"SongID_{song_id}")

    song_duration = chart_metrics["song_duration"]
    raw_inst_densities_from_worker = chart_metrics.get("raw_inst_densities_for_plot", [])
    density_curve_times_from_worker = np.array(chart_metrics.get("density_curve_times_for_plot", []))
    inst_calc_window_size = chart_metrics.get("inst_effective_window_used", INST_DENSITY_WINDOW_SIZE)

    if not raw_inst_densities_from_worker or density_curve_times_from_worker.size == 0 or song_duration <= 0:
        print(f"Skipping plot for {chart_stem} due to missing density data.")
        return

    # --- Prepare data for plotting the density curve ---
    raw_densities_np_for_plot = np.array(raw_inst_densities_from_worker)
    plot_times = density_curve_times_from_worker
    plot_densities = raw_densities_np_for_plot

    if INST_DENSITY_SMOOTHING_POINTS > 1 and len(raw_densities_np_for_plot) >= INST_DENSITY_SMOOTHING_POINTS:
        smoothed_plot_densities = moving_average(raw_densities_np_for_plot, INST_DENSITY_SMOOTHING_POINTS)
        half_smooth_win = (INST_DENSITY_SMOOTHING_POINTS - 1) // 2
        plot_times_smoothed = plot_times[half_smooth_win: len(plot_times) - half_smooth_win]
        if INST_DENSITY_SMOOTHING_POINTS % 2 == 0 and len(plot_times_smoothed) > len(smoothed_plot_densities):
            plot_times_smoothed = plot_times_smoothed[:len(smoothed_plot_densities)]

        plot_densities = smoothed_plot_densities
        plot_times = plot_times_smoothed

    if plot_densities.size == 0:
        print(f"Skipping plot for {chart_stem} after smoothing resulted in no data.")
        return

    plt.figure(figsize=(16, 7))

    # --- Highlight Top Average Density Period ---
    # This uses the time_start and time_end from plot_context if the reason is TopAvg
    # These times in plot_context were set from chart_metrics['peak_avg_time_start'] and chart_metrics['peak_avg_time_end']
    if plot_context.get("reason") == "TopAvg":
        avg_peak_start_time = plot_context.get("time_start", -1)
        avg_peak_end_time = plot_context.get("time_end", -1)
        avg_window_size_for_highlight = plot_context.get("avg_window_size", "N/A")  # Get the window size for label

        if avg_peak_start_time != -1 and avg_peak_end_time != -1:
            plt.axvspan(avg_peak_start_time, avg_peak_end_time,
                        color='yellow', alpha=0.3, zorder=0,
                        label=f"最高平均密度时段 ({avg_window_size_for_highlight}s 窗口)")

    # Plot the main density curve (ensure it's plotted after axvspan so it's on top if zorder not specified for line)
    plt.plot(plot_times, plot_densities, antialiased=True, zorder=5,  # zorder higher than axvspan
             label=f"瞬时密度 (计算窗口: {inst_calc_window_size:.1f}s, 绘图平滑: {INST_DENSITY_SMOOTHING_POINTS}点)")

    # --- Construct Title ---
    title_main = f"音符密度: {song_name} (ID: {song_id})\n谱面: {chart_stem} - {diff_name} (定数 DS: {ds_str})"
    selection_reason_str = plot_context.get('reason_text', "N/A")
    title_selection_info = f"\n选为: {selection_reason_str}"

    overall_peak_inst_val = chart_metrics['peak_inst_val']
    overall_peak_inst_time_start = chart_metrics['peak_inst_time_start']
    overall_peak_inst_window = chart_metrics['inst_effective_window_used']

    if overall_peak_inst_val >= 0 and overall_peak_inst_time_start != -1:
        overall_peak_marker_time_center = overall_peak_inst_time_start + overall_peak_inst_window / 2
        plt.plot(overall_peak_marker_time_center, overall_peak_inst_val, 'ro', markersize=8, zorder=10,  # zorder higher
                 label=f"谱面最高瞬时密度 (平滑后): {overall_peak_inst_val:.2f} n/s")
        plt.text(overall_peak_marker_time_center, overall_peak_inst_val, f'{overall_peak_inst_val:.2f}',
                 ha='left', va='bottom', color='red', fontsize=9, zorder=11)  # zorder higher
        title_main += f"\n谱面最高瞬时密度: {overall_peak_inst_val:.2f} n/s @ {overall_peak_inst_time_start:.2f}s-{overall_peak_inst_time_start + overall_peak_inst_window:.2f}s (窗口 {overall_peak_inst_window:.1f}s)"

    plt.title(title_main + title_selection_info, fontsize=10)
    plt.xlabel(f"时间 (s) (瞬时密度计算窗口中心, 窗口大小 {inst_calc_window_size:.1f}s)")
    plt.ylabel("密度 (音符数/秒)")
    plt.grid(True, linestyle='--', alpha=0.7, zorder=1)  # Ensure grid is behind density curve and markers

    # --- Y-axis scaling ---
    if plot_densities.size > 0:
        max_plot_density = np.max(plot_densities) if plot_densities.size > 0 else 0
        y_upper_bound = max(max_plot_density, overall_peak_inst_val if overall_peak_inst_val >= 0 else 0)
        y_max = y_upper_bound * 1.2 if y_upper_bound > 0 else 6.0
        y_max = max(y_max, 6.0)
        plt.ylim(bottom=0, top=y_max)
    else:
        plt.ylim(bottom=0, top=6.0)

    plt.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Filename Construction ---
    reason_tag = plot_context.get("reason", "Unknown")
    rank_tag = plot_context.get("rank", 0)
    time_start_tag = plot_context.get("time_start", -1)
    time_end_tag = plot_context.get("time_end", -1)

    time_period_tag = "NoTime"
    if time_start_tag != -1 and time_end_tag != -1:
        time_period_tag = f"{time_start_tag:.1f}s-{time_end_tag:.1f}s"

    sanitized_stem = sanitize_filename_component(chart_stem)
    sanitized_song_name = sanitize_filename_component(song_name)
    sanitized_diff_name = sanitize_filename_component(diff_name)

    output_filename_base = f"{sanitized_song_name}_{sanitized_stem}_{sanitized_diff_name}_DS{ds_str}_{reason_tag}{rank_tag}_{time_period_tag}_density"
    output_filename = output_dir_for_level / f"{output_filename_base}.png"

    output_dir_for_level.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename)
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    charts_csv_file = Path("pre_info/chart.csv")
    if not charts_csv_file.exists():
        print(f"错误: 主要谱面CSV文件 {charts_csv_file} 未找到。程序将退出。")
        exit()

    try:
        charts_df_full = pd.read_csv(charts_csv_file)
        required_cols = ['file_name', 'song_id', 'level_index', 'ds']
        if not all(col in charts_df_full.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in charts_df_full.columns]
            print(f"错误: CSV文件 {charts_csv_file} 中缺少列: {', '.join(missing_cols)}。程序将退出。")
            exit()
    except Exception as e:
        print(f"错误: 读取CSV文件 {charts_csv_file} 失败: {e}")
        exit()

    if PROCESS_FIRST_N_CHARTS is not None and isinstance(PROCESS_FIRST_N_CHARTS, int) and PROCESS_FIRST_N_CHARTS > 0:
        charts_df = charts_df_full.head(PROCESS_FIRST_N_CHARTS).copy()  # Use .copy() to avoid SettingWithCopyWarning
        print(f"注意：仅处理前 {len(charts_df)}/{len(charts_df_full)} 个谱面。")
    else:
        charts_df = charts_df_full.copy()
        if PROCESS_FIRST_N_CHARTS is not None:
            print(f"警告：PROCESS_FIRST_N_CHARTS ({PROCESS_FIRST_N_CHARTS}) 设置无效，将处理所有谱面。")

    # Convert 'level_index' to string for consistent dictionary key usage
    charts_df['level_index'] = charts_df['level_index'].astype(str)

    print(f"正在从 {charts_csv_file} 处理 {len(charts_df)} 个谱面...")
    tasks_to_process = []
    for index, row in charts_df.iterrows():
        # Ensure ds is passed as string, handle potential NaNs from CSV
        ds_val_str = str(row['ds']) if pd.notna(row['ds']) else ""
        tasks_to_process.append((row['file_name'], str(row['song_id']), str(row['level_index']), ds_val_str))

    all_chart_metrics_data_raw = []
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(worker_wrapper, tasks_to_process)
        all_chart_metrics_data_raw = list(tqdm(results_iterator, total=len(tasks_to_process), desc="分析谱面"))

    all_chart_metrics_data_intermediate = [m for m in all_chart_metrics_data_raw if m is not None]
    if not all_chart_metrics_data_intermediate:
        print("未能处理任何谱面数据。程序将退出。")
        exit()

    # --- NEW: Add song_name and diff_name to metrics ---
    all_chart_metrics_data = []
    for metrics in all_chart_metrics_data_intermediate:
        metrics['song_name'] = get_song_info(metrics['song_id'])
        metrics['diff_name'] = diff_name_mapping.get(metrics['level_idx'], f"UnknownLvlIdx_{metrics['level_idx']}")
        all_chart_metrics_data.append(metrics)

    # Group charts by their DS-based difficulty group
    charts_by_difficulty_group = defaultdict(list)
    for metrics in all_chart_metrics_data:
        charts_by_difficulty_group[metrics['difficulty_group']].append(metrics)

    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- 各难度组统计数据与Top谱面 (基于定数DS分组) ---")

    for group_name, group_charts_metrics in charts_by_difficulty_group.items():
        if not group_charts_metrics:
            print(f"\n-- {group_name} --\n  此组无谱面数据。")
            continue

        print(f"\n-- {group_name} (共 {len(group_charts_metrics)} 个谱面) --")

        # Filter for valid metrics before sorting
        valid_inst_metrics = [m for m in group_charts_metrics if
                              isinstance(m.get('peak_inst_val'), (int, float)) and m['peak_inst_val'] >= 0 and m.get(
                                  'peak_inst_time_start', -1) != -1]
        valid_avg_metrics = [m for m in group_charts_metrics if
                             isinstance(m.get('peak_avg_val'), (int, float)) and m['peak_avg_val'] >= 0 and m.get(
                                 'peak_avg_time_start', -1) != -1]

        if not valid_inst_metrics and not valid_avg_metrics:
            print("  此组所有谱面未能计算有效峰值。")
            continue

        # --- Top Instantaneous Density Charts ---
        if valid_inst_metrics:
            top_inst_charts = sorted(valid_inst_metrics, key=lambda x: x['peak_inst_val'], reverse=True)[
                              :NUM_TOP_A_INSTANTANEOUS_PER_LEVEL]
            print(f"  Top {len(top_inst_charts)} 瞬时密度最高 (平滑后):")
            for i, chart_data in enumerate(top_inst_charts):
                ds_display = f"{chart_data['ds_value']:.2f}" if pd.notna(chart_data['ds_value']) else "N/A"
                time_period_str = f"{chart_data['peak_inst_time_start']:.2f}s-{chart_data['peak_inst_time_end']:.2f}s"
                print(
                    f"    {i + 1}. {chart_data['song_name']} ({chart_data['song_id']}) - {chart_data['stem']} ({chart_data['diff_name']}, DS: {ds_display})")
                print(
                    f"       瞬时密度: {chart_data['peak_inst_val']:.2f} n/s @ {time_period_str} (窗口 {chart_data['inst_effective_window_used']:.1f}s)")

                plot_context_inst = {
                    'reason': "TopInst",
                    'rank': i + 1,
                    'time_start': chart_data['peak_inst_time_start'],
                    'time_end': chart_data['peak_inst_time_end'],
                    'reason_text': (f"瞬时密度第 {i + 1} ({chart_data['peak_inst_val']:.2f} n/s "
                                    f"@ {time_period_str}, 计算窗口 {chart_data['inst_effective_window_used']:.1f}s)")
                }
                level_plot_output_dir = OUTPUT_PLOT_DIR / sanitize_filename_component(group_name)
                generate_density_plot(chart_data, plot_context_inst, level_plot_output_dir)
        else:
            print("  无有效数据计算Top瞬时密度。")

        # --- Top Average Density Charts ---
        if valid_avg_metrics:
            top_avg_charts = sorted(valid_avg_metrics, key=lambda x: x['peak_avg_val'], reverse=True)[
                             :NUM_TOP_B_AVERAGE_PER_LEVEL]
            print(f"  Top {len(top_avg_charts)} 平均密度最高:")
            for i, chart_data in enumerate(top_avg_charts):
                ds_display = f"{chart_data['ds_value']:.2f}" if pd.notna(chart_data['ds_value']) else "N/A"
                time_period_str = f"{chart_data['peak_avg_time_start']:.2f}s-{chart_data['peak_avg_time_end']:.2f}s"
                avg_win_display = f"{chart_data['avg_win_for_peak']:.1f}s" if chart_data[
                                                                                  'avg_win_for_peak'] != -1 else "N/A"
                print(
                    f"    {i + 1}. {chart_data['song_name']} ({chart_data['song_id']}) - {chart_data['stem']} ({chart_data['diff_name']}, DS: {ds_display})")
                print(
                    f"       平均密度: {chart_data['peak_avg_val']:.2f} n/s @ {time_period_str} ({avg_win_display} 窗口)")

                plot_context_avg = {
                    'reason': "TopAvg",
                    'rank': i + 1,
                    'time_start': chart_data['peak_avg_time_start'],
                    'time_end': chart_data['peak_avg_time_end'],
                    'avg_window_size': chart_data['avg_win_for_peak'],  # For title info
                    'reason_text': (f"平均密度第 {i + 1} ({chart_data['peak_avg_val']:.2f} n/s "
                                    f"@ {time_period_str}, 计算窗口 {avg_win_display})")
                }
                level_plot_output_dir = OUTPUT_PLOT_DIR / sanitize_filename_component(group_name)
                generate_density_plot(chart_data, plot_context_avg, level_plot_output_dir)
        else:
            print("  无有效数据计算Top平均密度。")

    print("\n分析与绘图完成。")
    print(f"输出图表已保存至: {OUTPUT_PLOT_DIR.resolve()}")
