import json
import os
import re
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

# ================= 配置路径 =================  
# test的只有10条，
# INPUT_FILE = "/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_2k_new.json" 
INPUT_FILE = '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_2k_new_copy.json'
OUTPUT_FILE = "/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_2k_new_dpo.json"
BATCH_SIZE = 64  # 根据显存大小调整
NUM_WORKERS = 8  # DataLoader的线程数

# ================= ResNet 特征提取器 =================
class ResNet50Extractor:
    def __init__(self, device="cuda"):
        self.device = device
        # 加载预训练模型
        print("Loading ResNet50 model...")
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Identity() # 去掉全连接层
        self.model.to(device)
        self.model.eval()
        
        # 定义预处理
        self.tf = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ================= 图片数据集类 (用于批量提取) =================
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个全黑图防止崩溃，后续需处理
            return torch.zeros(3, 256, 128), path

# ================= 核心处理逻辑 =================

def extract_index_from_response(response_text):
    """从回答文本中解析出 image {number} 的数字"""
    # 匹配 "image 4", "image 1" 等格式
    match = re.search(r"image\s+(\d+)", response_text, re.IGNORECASE)
    if match:

        return int(match.group(1))
    else : 
        print(f"Warning: Could not extract image index from response: {response_text}")
    return None

def process_single_item(item, feature_cache):
    """
    处理单条数据，计算相似度并构建DPO格式
    """
    try:
        # 1. 解析原始数据
        user_msg = next(msg for msg in item['messages'] if msg['role'] == 'user')
        gpt_msg = next(msg for msg in item['messages'] if msg['role'] == 'assistant')
        
        image_list = item['images']
        
        # 2. 获取正确答案的索引 (Base Choose / Good Choose)
        # 注意：这里的 index 是从 1 开始的 (image 1, image 2...)，代码中转为 0-based
        correct_number = extract_index_from_response(gpt_msg['content'])
        if correct_number is None:
            return None # 无法解析，跳过
        
        correct_idx = correct_number - 1 # 转为列表索引
        
        if correct_idx < 0 or correct_idx >= len(image_list):
            return None # 索引越界

        # 3. 获取 Target Image 的特征
        target_img_path = image_list[correct_idx]
        if target_img_path not in feature_cache:
            return None # 特征缺失
        
        target_feat = feature_cache[target_img_path]

        # 4. 计算相似度并寻找 Bad Choose (Hard Negative)
        # 逻辑：在所有非 Target 图片中，找到与 Target 相似度最高的那张
        max_sim = -1.0
        bad_choose_idx = -1
        
        for i, img_path in enumerate(image_list):
            if i == correct_idx:
                continue # 跳过正确答案自己
            
            if img_path in feature_cache:
                feat = feature_cache[img_path]
                # 计算余弦相似度 (特征已经 normalize 过了，直接点积)
                sim = np.dot(target_feat, feat)
                
                if sim > max_sim:
                    max_sim = sim
                    bad_choose_idx = i
        
        if bad_choose_idx == -1:
            return None # 没有找到负样本

        # Bad Choose 的 number (用于填入 rejected)
        bad_choose_number = bad_choose_idx + 1

        # 5. 构建 DPO 数据格式
        new_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": user_msg['content']
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": gpt_msg['content'] # 原始的正确回答
            },
            "rejected": {
                "from": "gpt",
                "value": f"The best matches the following description is image {bad_choose_number}."
            },
            "images": image_list
        }
        return new_item

    except Exception as e:
        print(f"Error processing item: {e}")
        return None

def main():
    # 1. 加载原始 JSON
    print(f"Loading dataset from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 收集所有唯一的图片路径 (去重)
    print("Collecting unique images...")
    all_image_paths = set()
    for item in data:
        for img_path in item['images']:
            all_image_paths.add(img_path)
    
    unique_paths_list = list(all_image_paths)
    print(f"Total unique images to process: {len(unique_paths_list)}")

    # 3. 初始化模型和特征提取 (GPU 批量处理)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ResNet50Extractor(device=device)
    
    dataset = ImageDataset(unique_paths_list, extractor.tf)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    feature_cache = {}
    
    print("Extracting features (Batch Processing)...")
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            # 提取特征
            feats = extractor.model(imgs) # [B, 2048]
            # 归一化
            feats = torch.nn.functional.normalize(feats, dim=1)
            feats_np = feats.cpu().numpy()
            
            for path, feat in zip(paths, feats_np):
                feature_cache[path] = feat

    del extractor # 释放显存
    torch.cuda.empty_cache()

    # 4. 并发生成 DPO 数据 (CPU/IO Bound)
    print("Generating DPO dataset with concurrency...")
    dpo_data = []
    
    # 使用 ThreadPoolExecutor 并发处理列表中的每个条目
    # 注意：此时特征已在内存中，主要是CPU计算点积和字典操作
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_item, item, feature_cache) for item in data]
        
        # 获取结果
        for future in tqdm(futures, desc="Processing JSON logic"):
            result = future.result()
            if result:
                dpo_data.append(result)

    print(f"Original items: {len(data)}, DPO items generated: {len(dpo_data)}")

    # 5. 保存结果
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()