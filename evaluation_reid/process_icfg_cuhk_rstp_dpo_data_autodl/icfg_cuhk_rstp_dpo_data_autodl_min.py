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

# 设置权重路径环境变量
os.environ['TORCH_HOME'] = "/root/autodl-tmp/weights/torch/" 

"""
计算 Target 图片与其他候选图片的余弦相似度，选择相似度最低的那张作为 rejected。这能让 DPO 训练更稳定，明确告诉模型“选这个是对的，选那个差异巨大的是错的”。
"""

# ================= 配置路径 =================  
# 这里填入你合并好的 3 个 16k 数据集路径

INPUT_FILES = [
    '/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_cuhk_16k_unified_autodl.json',
    '/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_icfg_16k_unified_autodl.json',
    '/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_rstp_16k_unified_autodl.json'
]

OUTPUT_FILE = "/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_t2i_dpo_mixed_least_similar.json"

BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= ResNet 特征提取器 =================
class ResNet50Extractor:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading ResNet50 model on {device}...")
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Identity()
        self.model.to(device)
        self.model.eval()
        
        self.tf = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ================= 图片数据集类 =================
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # 兼容路径，防止部分路径缺少 /root/ 前缀
            if not os.path.exists(path) and not path.startswith('/'):
                 # 这里可以根据你的实际环境做路径修正，比如拼上前缀
                 pass
            
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as e:
            # 返回全黑图，避免中断
            return torch.zeros(3, 256, 128), path

# ================= 核心处理逻辑 =================

def extract_index_from_response(response_text):
    """从回答文本中解析出 image {number} 的数字"""
    # 匹配 "image 4", "image 1", "Image 1" 等格式
    match = re.search(r"image\s+(\d+)", response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def is_target_task(item):
    """
    判断是否是 Text-to-Image One-to-Many 任务
    特征：Prompt 里包含 'select the image' 且有多张图片
    """
    try:
        user_msg = next(msg for msg in item['messages'] if msg['role'] == 'user')
        content = user_msg['content'].lower()
        # 你的 Prompt 模板里包含这些关键词
        if "select the image" in content and "given the following images" in content:
            return True
        # 兼容旧版 Prompt 或者其他类似任务
        if "best matches the following description" in content:
            return True
    except:
        return False
    return False

def process_single_item(item, feature_cache):
    try:
        # 1. 解析原始数据
        user_msg = next(msg for msg in item['messages'] if msg['role'] == 'user')
        gpt_msg = next(msg for msg in item['messages'] if msg['role'] == 'assistant')
        image_list = item['images']
        
        # 2. 获取正确答案索引 (Chosen)
        correct_number = extract_index_from_response(gpt_msg['content'])
        if correct_number is None:
            return None
        
        correct_idx = correct_number - 1
        if correct_idx < 0 or correct_idx >= len(image_list):
            return None

        # 3. 获取 Target 特征
        target_img_path = image_list[correct_idx]
        if target_img_path not in feature_cache:
            return None
        target_feat = feature_cache[target_img_path]

        # 4. 寻找 Negative (Least Similar - 差异最大的)
        # 策略：在所有候选图中，找一个和 target 余弦相似度最低的
        min_sim = 2.0  # Cosine sim range [-1, 1], set init to > 1
        bad_choose_idx = -1
        
        for i, img_path in enumerate(image_list):
            if i == correct_idx:
                continue 
            
            if img_path in feature_cache:
                feat = feature_cache[img_path]
                # 计算相似度
                sim = np.dot(target_feat, feat)
                
                # 【关键修改】：寻找最小值 (Least Similar)
                if sim < min_sim:
                    min_sim = sim
                    bad_choose_idx = i
        
        if bad_choose_idx == -1:
            return None

        bad_choose_number = bad_choose_idx + 1

        # 5. 构建 DPO 条目
        # 注意：Rejected 的回答格式必须和 Chosen 一模一样，只是数字不同
        rejected_content = gpt_msg['content'].replace(str(correct_number), str(bad_choose_number))
        
        # 如果 replace 失败（比如 numbers 没匹配上），手动构建
        if str(bad_choose_number) not in rejected_content:
             rejected_content = f"The best matches the following description is image {bad_choose_number}."

        new_item = {
            "conversations": [
                {
                    "from": "human",
                    "value": user_msg['content']
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": gpt_msg['content']
            },
            "rejected": {
                "from": "gpt",
                "value": rejected_content
            },
            "images": image_list
        }
        return new_item

    except Exception as e:
        return None

def main():
    # 1. 读取并合并所有 JSON 数据
    all_data = []
    print(f"Reading {len(INPUT_FILES)} input files...")
    
    # 用于简单的去重 (防止不同文件里有完全一样的 id)
    seen_ids = set() 
    
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  Loaded {len(data)} entries from {os.path.basename(file_path)}")
            
            for item in data:
                # 检查是否是我们需要处理的 T2I 任务
                if not is_target_task(item):
                    continue
                
                # 简单的 ID 去重 (如果 id 字段存在)
                item_id = item.get('id', str(item.get('images', []))) # 如果没有 ID，用 images 列表做 hash
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                all_data.append(item)

    print(f"Total T2I One-to-Many tasks filtered: {len(all_data)}")
    if len(all_data) == 0:
        print("No valid tasks found. Exiting.")
        return

    # 2. 收集图片路径
    unique_paths = set()
    for item in all_data:
        for p in item['images']:
            unique_paths.add(p)
    unique_paths_list = list(unique_paths)
    print(f"Unique images to process: {len(unique_paths_list)}")

    # 3. 提取特征
    extractor = ResNet50Extractor(device=DEVICE)
    dataset = ImageDataset(unique_paths_list, extractor.tf)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    feature_cache = {}
    print("Extracting features...")
    
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            feats = extractor.model(imgs)
            feats = torch.nn.functional.normalize(feats, dim=1)
            feats_np = feats.cpu().numpy()
            
            for p, f in zip(paths, feats_np):
                feature_cache[p] = f
    
    del extractor
    torch.cuda.empty_cache()

    # 4. 生成 DPO
    print("Generating DPO pairs (Least Similar Negative Sampling)...")
    dpo_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_item, item, feature_cache) for item in all_data]
        for future in tqdm(futures):
            res = future.result()
            if res:
                dpo_data.append(res)

    # 5. 保存
    print(f"Saving {len(dpo_data)} DPO samples to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    print("Finished.")

if __name__ == "__main__":
    main()