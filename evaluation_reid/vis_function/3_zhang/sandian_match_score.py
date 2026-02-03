import json
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_score(text):
    """
    使用正则表达式从文本中提取分值。
    修正：去除了数字末尾可能存在的句号（点）。
    """
    # 改进的正则：匹配数字开头，中间可能有小数点，但排除了末尾的句号
    # \d+ 表示匹配数字，(?:\.\d+)? 表示匹配可选的小数部分
    match = re.search(r"image is:\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    
    # 备选方案：如果上面的正则没匹配到，尝试匹配带点的字符串并手动去掉末尾的点
    match_fallback = re.search(r"image is:\s*([\d.]+)", text)
    if match_fallback:
        score_str = match_fallback.group(1).rstrip('.')
        try:
            return float(score_str)
        except ValueError:
            return None
            
    return None

def plot_match_scores(file_path, output_image='/home/wangrui/code/LLaMA-Factory/evaluation_reid/vis_function/3_zhang/pic_save/match_score_scatter.png'):
    labels = []
    predictions = []

    # 1. 读取并解析数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # 提取 predict 和 label 中的分数
                pred_val = extract_score(data.get('predict', ''))
                label_val = extract_score(data.get('label', ''))
                
                if pred_val is not None and label_val is not None:
                    predictions.append(pred_val)
                    labels.append(label_val)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"解析过程中出现错误: {e}")
        return

    if not labels:
        print("未提取到有效的分数数据，请检查文件内容和正则匹配规则。")
        return

    # 2. 绘图
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 绘制散点
    plt.scatter(labels, predictions, alpha=0.6, s=20, c='royalblue', edgecolors='none', label='Data Points')
    
    # 绘制 y=x 理想线
    max_val = max(max(labels), max(predictions))
    min_val = min(min(labels), min(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal ($y=x$)')
    
    # 设置坐标轴标签和标题
    plt.xlabel('True Match-Score (Label)', fontsize=12)
    plt.ylabel('Predicted Match-Score (Predict)', fontsize=12)
    plt.title('Prediction Accuracy: Predicted vs. True Match-Scores', fontsize=14)
    
    # 添加图例和网格
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # 保持坐标轴比例一致（可选，根据分数范围决定）
    # plt.axis('equal')

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"散点图已成功保存至: {output_image}")

if __name__ == "__main__":
    # 您的文件路径
    target_file = '/home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/with_score_ccls/lora64_full_unfreeze/generated_predictions_with_images_sorted.jsonl'
    
    plot_match_scores(target_file)