import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# cuhk 数据集
cuhk_data = {
    'Method': ['HPT', 'DPO', 'EIDR'],
    'R1': [78.88, 79.04, 81.17],
    'R2': [93.89, 94.11, 95.62],
    'R3': [96.23, 96.70, 96.84],
    'mAP': [74.31, 77.24, 78.39]
}

# icfg 数据集
icfg_data = {
    'Method': ['HPT', 'DPO', 'EIDR'],
    'R1': [73.00, 74.21, 74.70],
    'R2': [89.10, 90.01, 90.20],
    'R3': [92.78, 93.56, 94.18],
    'mAP': [53.30, 53.81, 54.19]
}

# RSTPreid 数据集
rstp_data = {
    'Method': ['HPT', 'DPO', 'EIDR'],
    'R1': [70.00, 74.11, 75.50],
    'R2': [89.51, 93.27, 94.50],
    'R3': [93.21, 95.38, 96.44],
    'mAP': [57.82, 62.92, 63.59]
}

datasets = [
    ('CUHK-PEDES', pd.DataFrame(cuhk_data)),
    ('ICFG-PEDES', pd.DataFrame(icfg_data)),
    ('RSTPReID', pd.DataFrame(rstp_data))
]

# 设置绘图风格
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义用户指定的配色 (尽管只有3个方法，但这里传入完整的列表，matplotlib会自动按顺序使用)
custom_colors = ['#516b91', '#59c4e6', '#edafda', '#93b7e3']

# 2. 循环绘制 3 个柱状图
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

for idx, (name, df) in enumerate(datasets):
    # 转置以便指标作为 X 轴
    df_plot = df.set_index('Method').T

    # 绘图时应用自定义颜色
    ax = df_plot.plot(kind='bar', ax=axes[idx], rot=0, width=0.8, color=custom_colors)

    axes[idx].set_title(f'{name} Performance Comparison at Each Stage', fontsize=15, fontweight='bold', pad=12)
    axes[idx].set_ylabel('Score', fontsize=12)
    axes[idx].set_ylim(0, 115) # 给顶部标注留出空间
    axes[idx].legend(title='Experimental Phase', loc='upper right', fontsize=10)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.6)

    # 标注数值
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=9, weight='semibold')

plt.tight_layout()
plt.savefig('/home/wangrui/code/LLaMA-Factory/evaluation_reid/vis_function/5_zhang/pic_save/48k_DPO_init_stage2_vis.png', dpi=300)
plt.show()

