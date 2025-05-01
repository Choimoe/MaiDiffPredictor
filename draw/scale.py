import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.ticker import FuncFormatter


class DifficultyScale(mscale.ScaleBase):
    name = 'difficulty'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.axis = axis
        self.threshold = kwargs.pop('threshold', 12)
        self.linear_scale = kwargs.pop('linear_scale', 0.8)
        self.exp_scale = kwargs.pop('exp_scale', 1.6)
        self.major_ticks = kwargs.pop('major_ticks', [-5, 0, 5, 10, 12, 13, 14, 15, 15.7])

        self.exp_offset = self.linear_scale * self.threshold

        # 将参数挂载到 axis 上
        axis.threshold = self.threshold
        axis.linear_scale = self.linear_scale
        axis.exp_scale = self.exp_scale
        axis.exp_offset = self.exp_offset
        axis.major_ticks = self.major_ticks

    def get_transform(self):
        # 传递当前 axis 到 DifficultyTransform
        return self.DifficultyTransform(self.axis)

    def set_default_locators_and_formatters(self, axis):
        # 设置主要刻度
        axis.set_major_locator(plt.FixedLocator(axis.major_ticks))

        # 设置次要刻度
        minor_ticks = []
        for i in range(len(axis.major_ticks) - 1):
            minor_ticks.extend(np.linspace(axis.major_ticks[i], axis.major_ticks[i + 1], 5))
        axis.set_minor_locator(plt.FixedLocator(minor_ticks))

        # 设置格式化
        axis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))

    class DifficultyTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, axis):
            super().__init__()
            self.axis = axis  # 显式保存 axis 引用

        def transform_non_affine(self, values):
            if len(values) == 0:
                return np.array([])

            def _transform(y):
                if y <= 0:
                    return y
                elif y <= self.axis.threshold:  # 现在可以正确访问 axis.threshold
                    return y * self.axis.linear_scale
                else:
                    return self.axis.exp_offset + (y - self.axis.threshold) ** self.axis.exp_scale

            return np.vectorize(_transform, otypes=[np.float64])(values)

        def inverted(self):
            return DifficultyScale.InvertedDifficultyTransform(self.axis)

    class InvertedDifficultyTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, axis):
            super().__init__()
            self.axis = axis  # 同样需要保存 axis 引用

        def transform_non_affine(self, values):
            if len(values) == 0:
                return np.array([])

            def _inverse(y):
                if y <= 0:
                    return y
                elif y <= self.axis.linear_scale * self.axis.threshold:
                    return y / self.axis.linear_scale
                else:
                    return self.axis.threshold + (y - self.axis.exp_offset) ** (1 / self.axis.exp_scale)

            return np.vectorize(_inverse, otypes=[np.float64])(values)


mscale.register_scale(DifficultyScale)
