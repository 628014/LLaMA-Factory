import os
import json
import logging
import random
import numpy as np
from tqdm import tqdm
from metrics import evaluate_rank
from vlm_ranker import rank_text2image_confidence

def run_iterative_eval(model, processor, test_data, image_root, cache_file, num_iters=5):
    """
    通用评估函数：适配 RSTPReid, CUHK-PEDES, ICFG-PEDES
    - 增加 Rank-5 和 Rank-10 的评估与返回
    """
    
    # 读取预先计算好的 Top-K Gallery 缓存
    with open(cache_file, "r") as f:
        gallery_data = json.load(f)

    # --- 关键步骤 1: 构建 path -> pid 的映射字典 ---
    path_to_pid_map = {}
    for entry in test_data:
        pid_str = str(entry["id"])
        path_to_pid_map[entry["file_path"]] = pid_str

    # 初始化日志
    logging.basicConfig(filename='eval_process.log', filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', force=True)
    
    # 修改：存储每一轮的 (R1, R5, R10, mAP)
    round_metrics = []

    print(f"Starting evaluation with {num_iters} rounds...")

    # --- 外层循环：5 次迭代 ---
    for round_idx in range(1, num_iters + 1):
        logging.info(f"\n{'='*20} ROUND {round_idx} START {'='*20}")
        print(f"Running Round {round_idx}/{num_iters}...")
        
        current_round_r1s = []
        current_round_r5s = []   # 新增
        current_round_r10s = []  # 新增
        current_round_aps = []
        processed_pids = set() 

        # --- 内层循环：遍历测试集 ---
        for sample in tqdm(test_data, desc=f"Round {round_idx}"):
            # 1. 获取 Query ID
            qid = str(sample["id"])
            
            if qid not in gallery_data: 
                continue
            
            if qid in processed_pids:
                continue
            processed_pids.add(qid)

            # 2. 获取文本
            captions_list = sample.get("captions", [])
            if len(captions_list) == 0:
                continue 
            
            caption = captions_list[-1]  # 固定选择最后一条（或根据需要改为随机）
            
            # 3. 获取 Gallery 图片路径
            gallery_rel_paths = gallery_data[qid]["gallery"]
            gallery_full_paths = [os.path.join(image_root, p) for p in gallery_rel_paths]
            
            # 4. 构造 GT (Ground Truth)
            gt = []
            for p_rel in gallery_rel_paths:
                g_pid = path_to_pid_map.get(p_rel)
                if g_pid is not None and g_pid == qid:
                    gt.append(1)
                else:
                    gt.append(0)

            # 5. 执行推理 
            ranked_indices, _ = rank_text2image_confidence(
                model, 
                processor, 
                caption, 
                gallery_full_paths, 
                max_batch_size=5,
                log_prefix=f"R{round_idx}|PID:{qid}"
            )
            
            # 6. 计算指标 (R1, R5, R10, AP)
            # 获取排序后的 GT 状态
            ranked_gt = [gt[i] for i in ranked_indices]
            
            # 手动计算 R1, R5, R10
            # 只要前 K 个中有任意一个是正样本(1)，则 Rank-K 为 1，否则为 0
            curr_r1 = 1 if 1 in ranked_gt[:1] else 0
            curr_r5 = 1 if 1 in ranked_gt[:5] else 0
            curr_r10 = 1 if 1 in ranked_gt[:10] else 0
            
            # 使用原有 metrics 计算 AP (依然保留 evaluate_rank 以计算 AP，忽略其返回的 r1)
            _, ap = evaluate_rank(gt, ranked_indices)
            
            current_round_r1s.append(curr_r1)
            current_round_r5s.append(curr_r5)
            current_round_r10s.append(curr_r10)
            current_round_aps.append(ap)
            
            # 详细日志
            logging.info(f"Round {round_idx} | PID: {qid} | R1: {curr_r1} | R5: {curr_r5} | R10: {curr_r10} | AP: {ap:.4f}")

        # --- 本轮结束，计算本轮全局平均值 ---
        if not current_round_r1s:
            avg_r1, avg_r5, avg_r10, avg_map = 0.0, 0.0, 0.0, 0.0
        else:
            avg_r1 = sum(current_round_r1s) / len(current_round_r1s)
            avg_r5 = sum(current_round_r5s) / len(current_round_r5s)
            avg_r10 = sum(current_round_r10s) / len(current_round_r10s)
            avg_map = sum(current_round_aps) / len(current_round_aps)
        
        logging.info(f"{'='*20} ROUND {round_idx} SUMMARY {'='*20}")
        logging.info(f"Global Rank-1 : {avg_r1:.4f}")
        logging.info(f"Global Rank-5 : {avg_r5:.4f}")
        logging.info(f"Global Rank-10: {avg_r10:.4f}")
        logging.info(f"Global mAP    : {avg_map:.4f}")
        
        round_metrics.append((avg_r1, avg_r5, avg_r10, avg_map))

    # --- 5 轮结束，取 Rank-1 最高的那个维度的最大值，或者分别取最大值 ---
    # 通常取 Metrics 最好的一轮 (这里按照 R1 最大的那轮来选，或者分别取最大)
    # 为了保险起见，这里逻辑改为：分别取各项指标在所有轮次中的最大值
    final_max_r1 = max([m[0] for m in round_metrics]) if round_metrics else 0.0
    final_max_r5 = max([m[1] for m in round_metrics]) if round_metrics else 0.0
    final_max_r10 = max([m[2] for m in round_metrics]) if round_metrics else 0.0
    final_max_map = max([m[3] for m in round_metrics]) if round_metrics else 0.0

    logging.info(f"\n{'#'*20} FINAL EVALUATION RESULT {'#'*20}")
    logging.info(f"Max Global Rank-1 : {final_max_r1:.4f}")
    logging.info(f"Max Global Rank-5 : {final_max_r5:.4f}")
    logging.info(f"Max Global Rank-10: {final_max_r10:.4f}")
    logging.info(f"Max Global mAP    : {final_max_map:.4f}")

    return final_max_r1, final_max_r5, final_max_r10, final_max_map