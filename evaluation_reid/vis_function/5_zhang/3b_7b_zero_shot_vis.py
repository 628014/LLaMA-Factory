import pandas as pd
import matplotlib.pyplot as plt

# 准备数据
models = ['qwen2.5vl-7b', 'qwen2.5vl-3b']

# Rank-1 数据
data_rank1 = {
    'Dataset': ['CUHK-PEDES', 'ICFG-PEDES'],
    'qwen2.5vl-7b': [0.7040, 0.6980],
    'qwen2.5vl-3b': [0.6720, 0.6730]
}
df_rank1 = pd.DataFrame(data_rank1).set_index('Dataset')

# mAP 数据
data_map = {
    'Dataset': ['CUHK-PEDES', 'ICFG-PEDES'],
    'qwen2.5vl-7b': [0.7767, 0.5368],
    'qwen2.5vl-3b': [0.7682, 0.5297]
}
df_map = pd.DataFrame(data_map).set_index('Dataset')

# 设置字体以支持中文
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义用户指定的配色
custom_colors = ['#516b91', '#59c4e6']

# 创建左右两个子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 定义统一的绘图函数
def plot_subgraph(ax, df, title, ylabel):
    # 使用自定义颜色
    df.plot(kind='bar', ax=ax, rot=0, width=0.7, color=custom_colors)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('T2I Dataset', fontsize=12)
    # 优化网格线
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_axisbelow(True) # 让网格线在柱子后面

    # 调整图例到右下角
    ax.legend(title='Model', frameon=True, shadow=True, loc='lower right')

    # 保持顶部空间
    ax.set_ylim(0, df.max().max() * 1.15)

    # 添加数值标注
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.4f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points',
                    fontsize=10, weight='semibold', color='#333333')

# 绘制 Rank-1
plot_subgraph(axes[0], df_rank1, 'Rank-1 ', 'Score')

# 绘制 mAP
plot_subgraph(axes[1], df_map, 'mAP ', 'Score')

plt.tight_layout()
plt.savefig('/home/wangrui/code/LLaMA-Factory/evaluation_reid/vis_function/5_zhang/pic_save/3b_7B-zero_shot.png', dpi=300)
plt.show()

# # 显示数据表格
# print("--- Rank-1 Data ---")
# display(df_rank1)
# print("\n--- mAP Data ---")
# display(df_map)