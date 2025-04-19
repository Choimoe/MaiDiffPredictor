import numpy as np
import pandas as pd
import os

from dataset import ChartDataset


class ChartAugmenter:
    """生成保留难度特性的增强谱面"""

    def __init__(self, difficulty_dims=6):
        self.dims = difficulty_dims

    def time_warp(self, sequence, factor=0.9):
        """时间轴缩放增强"""
        new_len = int(len(sequence) * factor)
        indices = np.linspace(0, len(sequence) - 1, new_len).astype(int)
        return sequence[indices]

    def track_mirror(self, sequence):
        """轨道镜像增强"""
        mirrored = sequence.copy()
        mirrored[..., :8] = mirrored[..., [4, 3, 2, 1, 0, 7, 6, 5]]  # 轨道1-8镜像映射
        return mirrored

    def difficulty_preserved_augment(self, orig_data, max_seq_len=1000):
        """难度保持增强"""
        # 随机选择增强方式
        aug_type = np.random.choice(['time_warp', 'track_mirror', 'noise'])

        orig_len = orig_data.shape[0]

        aug_data = ''

        if aug_type == 'time_warp':
            factor = np.random.uniform(0.8, 1.2)
            aug_data = self.time_warp(orig_data, factor)
        elif aug_type == 'track_mirror':
            aug_data = self.track_mirror(orig_data)
        else:
            noise = np.random.normal(0, 0.05, orig_data.shape)
            aug_data = np.clip(orig_data + noise, 0, 1)

        if aug_data.shape[0] < max_seq_len:
            # 填充
            pad_len = max_seq_len - aug_data.shape[0]
            aug_data = np.pad(aug_data, ((0, pad_len), (0, 0)), mode='constant')
        else:
            # 截断
            aug_data = aug_data[:max_seq_len]

        return aug_data

class EnhancedChartDataset(ChartDataset):
    def __init__(self, json_files, labels, chart_stat_path, chart_info_path, max_seq_len=1000):
        super().__init__(json_files, labels, max_seq_len)

        # 加载统计信息和谱面信息
        with open(chart_stat_path) as f:
            import json
            self.stat_data = json.load(f)["charts"]

        # 构建难度等级到统计特征的映射
        self.diff_level_stats = self._build_diff_level_stats(chart_info_path)

        # 提取统计特征
        self.stat_features = []
        for file, label in zip(json_files, labels):
            # 从文件路径解析ID和Difficulty
            parts = os.path.basename(file).split('_')
            song_id = parts[0]
            diff_level = int(parts[1].split('.')[0])

            # 获取chart_info中的原始难度等级
            chart_info = pd.read_csv(chart_info_path)
            raw_level = chart_info[
                (chart_info['ID'] == int(song_id)) &
                (chart_info['Difficulty'] == diff_level)
                ]['Level'].values[0]
            base_diff = float(raw_level.replace('+', '.5').replace('?', ''))

            # 尝试获取统计信息
            stats = {}
            if str(song_id) in self.stat_data:
                chart_list = self.stat_data[str(song_id)]
                if diff_level <= len(chart_list):
                    stats = chart_list[diff_level - 1]

            # 智能填充特征（优先使用同难度统计）
            self.stat_features.append([
                stats.get('fit_diff', base_diff),  # 第一优先级使用chart_info的Level
                stats.get('std_dev', self._get_diff_mean('std_dev', base_diff)),
                np.log(stats.get('cnt', self._get_diff_mean('cnt', base_diff, log=True))),
                self._safe_fc_ratio(stats.get('fc_dist', None), base_diff)
            ])

        # 标准化统计特征
        from sklearn.preprocessing import StandardScaler
        self.stat_scaler = StandardScaler()
        self.stat_features = self.stat_scaler.fit_transform(self.stat_features)

    def _build_diff_level_stats(self, chart_info_path):
        """构建按难度等级分组的统计特征库"""
        # 读取所有谱面信息
        df = pd.read_csv(chart_info_path)
        df['Level'] = df['Level'].str.replace('+', '.5').str.replace('?', '').astype(float)

        # 收集所有统计记录
        from collections import defaultdict
        stats_pool = defaultdict(list)
        for song_id, charts in self.stat_data.items():
            for diff_level, stat in enumerate(charts, 1):
                # 获取对应的官方难度等级
                official_level = df[
                    (df['ID'] == int(song_id)) &
                    (df['Difficulty'] == diff_level)
                    ]['Level'].values
                if len(official_level) == 0:
                    continue

                record = {
                    'std_dev': stat.get('std_dev', 0),
                    'cnt': stat.get('cnt', 0),
                    'fc_ratio': self._calc_fc_ratio(stat.get('fc_dist', [])),
                    'official_level': official_level[0]
                }
                stats_pool[official_level[0]].append(record)

        # 计算各难度等级的平均统计量
        diff_stats = {}
        for level, records in stats_pool.items():
            if len(records) == 0:
                continue
            diff_stats[level] = {
                'std_dev': np.mean([r['std_dev'] for r in records]),
                'cnt': np.exp(np.mean([np.log(r['cnt'] + 1) for r in records])),  # 几何平均
                'fc_ratio': np.mean([r['fc_ratio'] for r in records])
            }
        return diff_stats

    def _get_diff_mean(self, field, base_diff, log=False):
        """获取同难度等级的平均值"""
        closest_level = self._find_closest_level(base_diff)
        default_val = {
            'std_dev': 1.0,
            'cnt': 1000,
            'fc_ratio': 0.5
        }.get(field, 0)

        # 获取平均值
        value = self.diff_level_stats.get(closest_level, {}).get(field, default_val)

        # 对cnt字段的特殊处理
        if field == 'cnt' and log:
            return np.log(value + 1)
        return value

    def _find_closest_level(self, target_level):
        """找到最接近的官方难度等级"""
        available_levels = list(self.diff_level_stats.keys())
        if not available_levels:
            return None
        return min(available_levels, key=lambda x: abs(x - target_level))

    def _safe_fc_ratio(self, fc_dist, base_diff):
        """带难度感知的FC比例计算"""
        if fc_dist is not None and sum(fc_dist) > 0:
            return sum(fc_dist[-2:]) / sum(fc_dist)

        # 使用同难度等级的平均FC比例
        closest_level = self._find_closest_level(base_diff)
        return self.diff_level_stats.get(closest_level, {}).get('fc_ratio', 0.5)

    def _calc_fc_ratio(self, fc_dist):
        """通用FC比例计算"""
        if not fc_dist or sum(fc_dist) == 0:
            return 0
        return sum(fc_dist[-2:]) / sum(fc_dist)