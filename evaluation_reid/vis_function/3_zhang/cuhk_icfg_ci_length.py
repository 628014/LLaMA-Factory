import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# plt.rcParams['axes.unicode_minus'] = False

# files = [
#     'CUHK-PEDES',
#     'RSTPReid', 
#     'ICFG-PEDES'
# ]

files = [
    'reid_raw_match_score_40201.json',
    'data_caption_all_qwen.json', 
    'ICFG-PEDES_match_score_54522.json'
]

# Store extracted lengths
processed_data = []

print("Processing files (Filtering length > 2000)...")

for f_path in files:
    if not os.path.exists(f_path):
        print(f"Warning: File {f_path} not found.")
        continue

    with open(f_path, 'r') as f:
        data = json.load(f)

    lens_col1 = [] # First caption
    lens_col2 = [] # Comparison caption (2nd or 3rd)

    lengths = [len(x.get('captions', [])) for x in data if 'captions' in x]

    if not lengths:
        processed_data.append({}) # Placeholder
        continue

    # Determine strategy
    from collections import Counter
    mode_len = Counter(lengths).most_common(1)[0][0]

    target_idx_2 = 1 
    strategy_name = ""

    if mode_len >= 3:
        target_idx_2 = 2 # Use 3rd caption (index 2)
        strategy_name = "1st vs 3rd"
    else:
        target_idx_2 = 1 # Use 2nd caption (index 1)
        strategy_name = "1st vs 2nd"

    print(f"File: {f_path}, Mode Length: {mode_len}, Strategy: {strategy_name}")

    for item in data:
        caps = item.get('captions', [])
        if len(caps) > 0:
            v1 = len(caps[0])
            # Apply filter > 750
            if v1 <= 2000:
                lens_col1.append(v1)

            if len(caps) > target_idx_2:
                v2 = len(caps[target_idx_2])
                # Apply filter > 750
                if v2 <= 2000:
                    lens_col2.append(v2)

    processed_data.append({
        'name': f_path,
        'col1': lens_col1,
        'col2': lens_col2,
        'idx2': target_idx_2
    })

# Plot with independent axes
fig, axes = plt.subplots(3, 2, figsize=(15, 12)) 
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, file_info in enumerate(processed_data):
    if not file_info: continue 

    row_axes = axes[i]
    fname = file_info['name']

    # Col 1 Plot
    ax1 = row_axes[0]
    data1 = file_info['col1']
    if data1:
        ax1.hist(data1, bins=10, color='#516b91', edgecolor='black', alpha=0.7)
        ax1.set_title(f"{fname}\nCaption Original \nN={len(data1)}")
        ax1.set_ylabel("Frequency")

    # Col 2 Plot
    ax2 = row_axes[1]
    data2 = file_info['col2']
    target_idx = file_info['idx2']
    idx_label = f"Caption {target_idx + 1} (Index {target_idx})"

    if data2:
        ax2.hist(data2, bins=10, color='#59c4e6', edgecolor='black', alpha=0.7)
        ax2.set_title(f"{fname}\nCaption Smart CoT \nN={len(data2)}")

plt.tight_layout()
plt.savefig('/home/wangrui/code/LLaMA-Factory/evaluation_reid/vis_function/3_zhang/pic_save/cuhk_icfg_ci_length.png', dpi=300)
plt.show()
