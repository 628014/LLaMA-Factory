"""
合并vLLm推理的结果与原caption相似度，同时将结果到合并到原始CUHK-PEDES数据文件中
"""

import json
import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ================= 配置路径 =================
# 1. 预测结果文件路径
PREDICT_FILE = '/root/autodl-tmp/saves/qwen2_5vl-7b/lora/pre_retpreid_with_score/lora64_full_cuhk_train_and_test_all/prediction_results.json'

# 2. 原始数据文件路径
RAW_FILE = '/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/reid_raw.json'

# 3. 图片前缀 (用于路径匹配)
IMG_PREFIX = '/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/imgs/'

# 4. 输出文件路径
OUTPUT_FULL = '/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/reid_raw_match_score_40201.json'
OUTPUT_100 = '/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/reid_raw_match_score_100.json'

# 5. 模型路径
MODEL_PATH = '/root/autodl-tmp/model/models/Qwen/Qwen3-Embedding-4B'


def extract_info(prediction_text):
    """
    使用正则提取描述(str1)和分数(number)
    """
    # 正则逻辑：
    # 1. 匹配 "The final description of the image is:" 和 "In summary" 之间的内容
    # 2. 匹配 "relevance to the image is:" 之后的数字
    # re.DOTALL 允许 . 匹配换行符
    pattern = re.compile(
        r"The final description of the image is:\s*(.*?)\.?\s*In summary, the degree of relevance to the image is:\s*([\d\.]+)", 
        re.DOTALL | re.IGNORECASE
    )
    
    match = pattern.search(prediction_text)
    if match:
        desc = match.group(1).strip()
        # 去掉末尾可能多余的标点
        if desc.endswith('.'):
            desc = desc[:-1]
        
        score = match.group(2).strip()
        # 再次清理分数，防止后面还有句号
        score = score.rstrip('.')
        return desc, score
    return None, None

def merge_data():
    """
    功能2: 提取并合并数据
    """
    print(">>> 阶段1: 开始提取并合并数据...")
    
    # 1. 加载原始数据
    print(f"读取原始文件: {RAW_FILE}")
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 建立快速查找索引: full_img_path -> raw_data_item
    # 注意：raw_json 里的 file_path 是相对路径，需要拼上前缀
    raw_map = {}
    for item in raw_data:
        # 构造绝对路径作为 Key，用于和预测结果匹配
        # 注意处理路径分隔符，防止 //
        rel_path = item.get('file_path', '').lstrip('/')
        abs_path = os.path.join(IMG_PREFIX, rel_path)
        raw_map[abs_path] = item

    # 2. 加载预测数据
    print(f"读取预测文件: {PREDICT_FILE}")
    with open(PREDICT_FILE, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    match_count = 0
    
    # 3. 遍历预测结果进行合并
    for pred_item in tqdm(pred_data, desc="合并进度"):
        pred_path = pred_item.get('file_path')
        prediction_text = pred_item.get('prediction', '')
        
        # 提取信息
        str1, score_str = extract_info(prediction_text)
        
        if str1 and score_str and pred_path in raw_map:
            target_item = raw_map[pred_path]
            
            # 将新描述追加到 captions
            target_item['captions'].append(str1)
            
            # 新增 match_score 字段
            target_item['match_score'] = score_str
            
            match_count += 1
    
    print(f"合并完成，共匹配并更新了 {match_count} 条数据。")

    # 4. 保存文件
    print(f"正在保存全量文件: {OUTPUT_FULL}")
    with open(OUTPUT_FULL, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
        
    print(f"正在保存前100条样例: {OUTPUT_100}")
    with open(OUTPUT_100, 'w', encoding='utf-8') as f:
        json.dump(raw_data[:100], f, ensure_ascii=False, indent=2)

    return raw_data

def load_model():
    """加载Qwen3-Embedding-4B模型"""
    print(f"正在加载模型: {MODEL_PATH}")
    # trust_remote_code=True 是必须的
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
    
    # 如果有GPU，移动到GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("模型已加载至 CUDA")
    else:
        print("警告: 未检测到 GPU，推理将非常慢")
    return model

def calculate_similarity_batch(data_list, model, batch_size=32):
    """
    功能1: 批量计算相似度 (比单条并发快得多)
    """
    print(">>> 阶段2: 开始计算 Embedding 相似度...")
    
    valid_pairs = []
    
    # 1. 准备数据对
    # str1 = item['captions'][-1] (新生成的)
    # str2 = " ".join(item['captions'][:-1]) (原始的合并)
    for item in data_list:
        if 'match_score' in item and len(item['captions']) >= 2:
            new_desc = item['captions'][-1]
            # 原始描述合并
            original_desc = " ".join(item['captions'][:-1])
            valid_pairs.append((new_desc, original_desc))
            
    print(f"共有 {len(valid_pairs)} 条有效数据参与相似度计算。")
    if len(valid_pairs) == 0:
        return 0.0

    all_similarities = []
    
    # 2. 批处理计算
    # 将列表拆分为 source_texts (new) 和 target_texts (old)
    source_texts = [p[0] for p in valid_pairs]
    target_texts = [p[1] for p in valid_pairs]
    
    total_batches = (len(valid_pairs) + batch_size - 1) // batch_size
    
    # 使用 tqdm 显示进度
    for i in tqdm(range(0, len(valid_pairs), batch_size), total=total_batches, desc="Embedding 计算中"):
        batch_source = source_texts[i : i + batch_size]
        batch_target = target_texts[i : i + batch_size]
        
        # 编码: 注意 prompt_name="query"
        # encode 返回的是 numpy array
        source_emb = model.encode(batch_source, prompt_name="query", convert_to_tensor=True, show_progress_bar=False)
        target_emb = model.encode(batch_target, prompt_name="query", convert_to_tensor=True, show_progress_bar=False)
        
        # 计算成对余弦相似度
        # torch.nn.functional.cosine_similarity 可以直接按行计算
        # dim=1 表示在特征维度上计算
        sim_scores = torch.nn.functional.cosine_similarity(source_emb, target_emb, dim=1)
        
        # 转回 CPU list
        all_similarities.extend(sim_scores.cpu().tolist())

    # 3. 计算平均值
    avg_similarity = np.mean(all_similarities)
    return avg_similarity

def main():
    # 步骤 1: 合并数据 (CPU操作)
    # merged_data = merge_data()
    
    with open(OUTPUT_FULL, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    

    
    # 步骤 2: 计算相似度 (GPU操作)
    # 为了防止显存溢出，batch_size 根据你的显存调整，4B模型比较大，设为 8 或 16 比较安全
    # 如果是 24G 显存 (4090)，可以尝试 32
    try:
        model = load_model()
        avg_score = calculate_similarity_batch(merged_data, model, batch_size=32)
        
        print("\n" + "="*50)
        print(f"最终结果统计:")
        print(f"处理总数据量: {len(merged_data)}")
        print(f"平均语义相似度 (Cosine Similarity): {avg_score:.6f}")
        print("="*50)
        
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()