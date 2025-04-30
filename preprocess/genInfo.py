from fileinput import filename
import json
import csv
import os
from pathlib import Path
import re

def process_level_string(level_str):
    if not level_str:
        return None
    
    # 统一转换为字符串处理
    str_value = str(level_str).strip()
    
    # 处理特殊符号
    has_plus = '+' in str_value
    has_question = '?' in str_value
    
    # 包含问号视为无效
    if has_question:
        return None
    
    # 提取基础数值
    clean_str = re.sub(r'[^0-9.]', '', str_value)
    if not clean_str:
        return None
    
    try:
        base = float(clean_str)
        # 应用修正规则
        if has_plus:
            return base + 0.75
        elif re.match(r'^\d+$', str_value):  # 纯整数
            return base + 0.25
        return base  # 已有小数则直接返回
    except ValueError:
        return None

def generate_chart_csv():
    # 初始化路径
    songs_json = Path('info/songs.json')
    stat_diff_json = Path('pre_info/chart_stat_diff.json')
    output_csv = Path('pre_info/chart.csv')
    chart_prefix = 'serialized_data/'
    
    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 读取歌曲元数据
    with open(songs_json, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    
    # 构建歌曲难度映射 {song_id: {level_index: ds}}
    song_ds_map = {}
    for song in songs:
        song_id = str(song['id'])
        valid_charts = []
        
        # 过滤无效chart条目
        for chart in song.get('charts', []):
            # 跳过空字典和缺少level字段的条目
            if not chart or 'level' not in chart:
                continue
            
            # 使用增强处理器
            processed_level = float(chart['level'])
            if processed_level is None:
                print(f"警告：歌曲 {song_id} 的难度等级解析失败：{chart['level']}")
                continue
            
            valid_charts.append({
                'level': processed_level,
                'original': chart['level']  # 保留原始值用于调试
            })
        
        # 仅当有有效谱面时才记录
        if valid_charts:
            song_ds_map[song_id] = {
                idx: chart['level']
                for idx, chart in enumerate(valid_charts)
            }

    # 读取拟合难度数据
    with open(stat_diff_json, 'r', encoding='utf-8') as f:
        stat_diff = json.load(f)

    chart_info_path = Path('pre_info/chart_info.csv')
    chart_info_map = {}
    if chart_info_path.exists():
        with open(chart_info_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 从文件路径提取song_id和level_index
                match = re.search(r'(\d+)_(\d+).json', row['FilePath'])
                if match:
                    song_id = match.group(1)
                    level_index = int(match.group(2)) - 1
                    chart_info_map[(song_id, level_index)] = row['Level']

    
    # 生成CSV数据
    csv_data = []
    for song_id, diff_list in stat_diff['charts'].items():
        # 新增过滤条件：剔除6位数的歌曲ID
        if len(song_id) == 6 and song_id.isdigit():
            continue
        # 原有的跳过无效歌曲ID逻辑
        if not song_id.isdigit():
            continue

        ds_map = song_ds_map.get(song_id, {})
        
        for level_index, diff_entry in enumerate(diff_list):
            # 过滤空条目和缺失关键字段的情况
            if not diff_entry or 'diff' not in diff_entry or 'fit_diff' not in diff_entry:
                continue

            try:
                # 优先使用歌曲元数据
                target_ds = song_ds_map.get(song_id, {}).get(level_index)
                
                # 次优先使用chart_info数据
                if target_ds is None:
                    level_from_info = chart_info_map.get((song_id, str(level_index)), '')
                    target_ds = process_level_string(level_from_info)
                
                # 最后使用stat_diff数据
                if target_ds is None:
                    raw_ds = diff_entry.get('diff', '')
                    target_ds = process_level_string(raw_ds) if raw_ds else None

                # 生成记录
                if target_ds is not None:
                    # 处理fit_diff调整
                    original_fit = diff_entry.get('fit_diff', 0)
                    try:
                        fit_diff_value = float(original_fit)
                    except (ValueError, TypeError):
                        fit_diff_value = 0.0  # 处理无效值

                    decimal_part = target_ds - int(target_ds)  # 获取小数部分
                    if decimal_part >= 0.7:
                        adjustment = 0.75
                    else:
                        adjustment = 0.25
                    
                    # 应用调整公式
                    adjusted_fit = fit_diff_value - adjustment + decimal_part
                    adjusted_fit_rounded = round(adjusted_fit, 4)

                    # 计算综合难度
                    combined = (target_ds + adjusted_fit_rounded) / 2
                    combined_rounded = round(combined, 4) if adjusted_fit_rounded else ''

                    csv_data.append({
                        'file_name': f"{chart_prefix}{song_id}_{level_index+1}.json",
                        'song_id': song_id,
                        'level_index': level_index,
                        'fit_diff': adjusted_fit_rounded,  # 使用调整后的值
                        'ds': round(target_ds, 3),
                        'combined_diff': combined_rounded  # 使用调整后的fit_diff计算
                    })
            except Exception as e:
                print(f"数据处理异常 song_id={song_id}: {str(e)}")
                continue

    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file_name', 'song_id', 'level_index', 
            'fit_diff', 'ds', 'combined_diff'
        ])
        writer.writeheader()
        writer.writerows(csv_data)

if __name__ == '__main__':
    generate_chart_csv()