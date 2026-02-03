import matplotlib.pyplot as plt

# 实验数据
m_values = [1, 2, 3]
avg_vals = [0.8638 * 100, 0.8640 * 100, 0.8641 * 100]
max_vals = [0.9965 * 100, 0.9969 * 100, 0.9973 * 100]
min_vals = [0.6500 * 100, 0.6544 * 100, 0.6563 * 100]
bleu4_vals = [69.585, 69.621, 69.678]

# 定义指标、对应数据以及不同的专业颜色
# 颜色依次为：经典蓝、森林绿、活力橙、学术红
metrics = [
    ('Avg. Similarity', avg_vals, '#1f77b4'), 
    ('Max. Similarity', max_vals, '#2ca02c'), 
    ('Min. Similarity', min_vals, '#ff7f0e'), 
    ('BLEU-4', bleu4_vals, '#d62728')
]

# 创建 1x4 的画布，增加宽度以保证四个子图不拥挤
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, (title, vals, color) in enumerate(metrics):
    # 绘制折线，并应用指定的颜色
    axes[i].plot(m_values, vals, marker='o', linestyle='-', linewidth=2.5, color=color, markersize=8)
    
    # 设置标题和标签
    axes[i].set_title(title, fontsize=14, fontweight='bold', pad=15)
    axes[i].set_xlabel('Numerical Precision ($M$)', fontsize=12)
    axes[i].set_ylabel('Metric Value (%)', fontsize=12)
    
    # 设置刻度和网格
    axes[i].set_xticks([1, 2, 3])
    axes[i].grid(True, linestyle='--', alpha=0.6)
    
    # 动态调整 y 轴范围，使趋势显示更加直观
    # 如果数据完全一致，则给一个默认边距
    if max(vals) != min(vals):
        margin = (max(vals) - min(vals)) * 0.3
        axes[i].set_ylim(min(vals) - margin, max(vals) + margin)
    else:
        axes[i].set_ylim(vals[0] - 0.005, vals[0] + 0.005)

# 调整布局防止标签重叠
plt.tight_layout()
# 保存高质量图片
plt.savefig('m_ablation_line_charts_multicolor_new.png', dpi=300)
plt.show()