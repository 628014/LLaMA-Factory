import os
import json
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# 设置权重路径环境变量
os.environ['TORCH_HOME'] = "/root/autodl-tmp/weights/torch/" 

# ================= 配置区域 =================
BATCH_SIZE = 256        # 推理时的 Batch Size
NUM_WORKERS = 16        # IO 读取线程数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReIDDataset(Dataset):
    """
    自定义数据集，用于 DataLoader 多进程并发读取
    """
    def __init__(self, file_paths, root_dir):
        self.file_paths = file_paths
        self.root_dir = root_dir
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)
        try:
            img = Image.open(full_path).convert("RGB")
            img = self.transform(img)
            return img, rel_path, True # True 表示读取成功
        except Exception as e:
            # 返回全0 tensor 占位
            return torch.zeros((3, 256, 128)), rel_path, False

def get_pid(sample, file_path):
    if "id" in sample:
        return str(sample["id"])
    return os.path.basename(file_path).split("_")[0]

def extract_features_parallel(unique_file_paths, image_root):
    """
    使用 DataLoader + Batch 并行提取特征
    """
    print(f"[Step 1] Extracting features for {len(unique_file_paths)} images (Batch Size: {BATCH_SIZE})...")
    
    # 1. 准备模型
    model = models.resnet50(weights='IMAGENET1K_V2')
    model.fc = torch.nn.Identity()
    model.to(DEVICE).eval()
    
    # 2. 准备数据加载器
    dataset = ReIDDataset(unique_file_paths, image_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    feat_dict = {}
    
    # 3. 批量推理
    with torch.no_grad():
        for imgs, paths, valids in tqdm(dataloader, desc="GPU Batch Extract"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda'):
                feats = model(imgs)
            
            # 归一化
            feats = torch.nn.functional.normalize(feats, dim=1)
            feats = feats.cpu().numpy() 
            
            for i, path in enumerate(paths):
                if valids[i]: 
                    feat_dict[path] = feats[i]
                    
    return feat_dict

def baseline_filter_50(test_samples, image_root, save_path, gallery_size=50):
    # --- 1. 数据准备 ---
    # 获取所有涉及到的图片路径（用于构建 Gallery 库）
    unique_file_paths = list(set([s["file_path"] for s in test_samples]))
    
    # --- 2. 提取特征 ---
    features_map = extract_features_parallel(unique_file_paths, image_root)
    
    # --- 3. 建立索引 & 筛选 Query ---
    print("[Step 2] Indexing IDs and Selecting One Query per ID...")
    
    id_to_samples_map = {} # PID -> [sample1, sample2, ...]
    
    # 3.1 将所有样本按 ID 分组
    for s in test_samples:
        if s["file_path"] in features_map:
            pid = get_pid(s, s["file_path"])
            if pid not in id_to_samples_map:
                id_to_samples_map[pid] = []
            id_to_samples_map[pid].append(s)
            
    # 3.2 每个 ID 选一个作为 Query
    query_samples = []
    
    for pid, samples in id_to_samples_map.items():
        # 策略：随机选一个，或者固定选第一个
        # 这里使用 random.choice 增加随机性，或者 samples[0] 保证确定性
        # 为了实验可复现，建议固定 seed 后随机选，或者直接选第一个
        # q_sample = random.choice(samples) 
        q_sample = samples[0] 
        query_samples.append(q_sample)

    print(f"Total Unique IDs: {len(id_to_samples_map)}")
    print(f"Selected Queries: {len(query_samples)} (One per ID)")

    # --- 4. 矩阵相似度计算 & 增量写入 ---
    print(f"[Step 3] Computing Similarity & Building Gallery (Size={gallery_size})...")
    
    # 准备全部特征矩阵 (Gallery Source)
    all_paths_list = list(features_map.keys())
    all_feats_tensor = torch.tensor(np.array([features_map[p] for p in all_paths_list])).to(DEVICE)
    
    # 准备增量写入
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    QUERY_BATCH_SIZE = 1000 
    total_queries_processed = 0

    print(f"Saving results incrementally to {save_path}...")
    
    with open(save_path, "w") as f:
        f.write("{\n")  # JSON 开始
        
        is_first_item = True
        
        # 分批处理 Query
        for i in tqdm(range(0, len(query_samples), QUERY_BATCH_SIZE), desc="Processing Queries"):
            batch_samples = query_samples[i : i + QUERY_BATCH_SIZE]
            
            # 获取当前批次 Query 的特征
            q_feats_numpy = [] 
            for s in batch_samples:
                q_feats_numpy.append(features_map[s["file_path"]])
            
            q_feats_tensor = torch.tensor(np.array(q_feats_numpy)).to(DEVICE)
            
            # 矩阵乘法: [Batch, 2048] @ [2048, N] -> [Batch, N]
            sim_matrix = torch.matmul(q_feats_tensor, all_feats_tensor.T)
            
            # 获取 Top-K 候选
            # 我们多取一些 (如 200)，以确保在排除正样本后还有足够的负样本填充
            k_search = min(200 + gallery_size, len(all_paths_list))
            topk_vals, topk_indices = torch.topk(sim_matrix, k=k_search, dim=1)
            
            # 转回 CPU
            topk_indices = topk_indices.cpu().numpy()
            topk_vals = topk_vals.cpu().numpy()
            
            # 处理批次中的每一个 Query
            for j, s in enumerate(batch_samples):
                q_path = s["file_path"]
                q_pid = get_pid(s, q_path)
                q_feat_single = q_feats_numpy[j] 
                
                # 获取该 ID 下所有的图片 (作为正样本候选)
                all_positives_samples = id_to_samples_map.get(q_pid, [])
                all_positives_paths = [sample["file_path"] for sample in all_positives_samples]
                
                # 排除 Query 自身
                if q_path in all_positives_paths:
                    all_positives_paths.remove(q_path)
                
                final_gallery = [] 
                final_scores = []
                pos_set = set(all_positives_paths) # 用于快速查找
                
                # === 核心逻辑 A: 强制包含正样本 (number1) ===
                # 这里我们把能找到的所有正样本都放进去
                # 剩下的位置留给负样本
                for pos_path in all_positives_paths:
                    if len(final_gallery) >= gallery_size:
                        break # 如果正样本太多，填满了 gallery (虽然很少见)，就停止
                    
                    final_gallery.append(pos_path)
                    # 手动计算正样本相似度
                    if pos_path in features_map:
                        pos_feat = features_map[pos_path]
                        score = float(np.dot(q_feat_single, pos_feat))
                        final_scores.append(score)
                    else:
                        final_scores.append(0.0)

                # === 核心逻辑 B: 用 Top-K 负样本填充剩余位置 (n) ===
                current_indices = topk_indices[j]
                current_vals = topk_vals[j]
                
                for k, idx in enumerate(current_indices):
                    # 如果 Gallery 满了，停止
                    if len(final_gallery) >= gallery_size:
                        break
                    
                    candidate_path = all_paths_list[idx]
                    candidate_score = float(current_vals[k])
                    
                    # 跳过自身
                    if candidate_path == q_path: continue
                        
                    # 关键：只添加负样本 (正样本已经在上面加过了)
                    if candidate_path not in pos_set:
                        final_gallery.append(candidate_path)
                        final_scores.append(candidate_score)
                
                # === 核心逻辑 C: 同步打乱 ===
                if len(final_gallery) > 0:
                    combined = list(zip(final_gallery, final_scores))
                    random.shuffle(combined)
                    final_gallery, final_scores = zip(*combined)
                    final_gallery = list(final_gallery)
                    final_scores = list(final_scores)

                # === D: 增量写入 ===
                # key 使用 id 或者 id_path 组合，因为这里每个id只有一个query，用id做key也可以
                # 但为了兼容性，还是保留原来的 key 风格
                key = str(s.get("id", os.path.basename(q_path)))
                
                result_item = {
                    "query": q_path,
                    "gallery": final_gallery,
                    "similarity_scores": final_scores,
                    "ground_truth_pid": q_pid,
                    "num_positives": len(all_positives_paths), # 记录该Query有多少个正样本被包含
                    "num_negatives": len(final_gallery) - len(all_positives_paths)
                }
                
                if not is_first_item:
                    f.write(",\n")
                else:
                    is_first_item = False
                
                json_string = json.dumps(result_item, ensure_ascii=False)
                f.write(f'  "{key}": {json_string}')
                
                total_queries_processed += 1
            
            # 刷新缓冲区到磁盘
            f.flush()

        f.write("\n}")  # JSON 结束

    print(f"\n[Done] Saved gallery to {save_path}")
    print(f"[Info] Total Unique IDs Processed: {total_queries_processed}")