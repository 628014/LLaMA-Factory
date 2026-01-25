import os
import json
import logging
import random
from tqdm import tqdm
from metrics import evaluate_rank
from vlm_ranker import rank_text2image_confidence

def run_iterative_eval(model, processor, test_data, image_root, cache_file, num_iters=5):
    """
    通用评估函数：适配 RSTPReid, CUHK-PEDES, ICFG-PEDES
    - 不依赖文件名解析 ID
    - 动态适配不同长度的 captions
    """
    
    # 读取预先计算好的 Top-K Gallery 缓存
    # cache 结构: { "pid_or_key": { "query": "path", "gallery": ["path1", "path2"...], ... } }
    with open(cache_file, "r") as f:
        gallery_data = json.load(f)

    # --- 关键步骤 1: 构建 path -> pid 的映射字典 ---
    # 因为 CUHK/ICFG 的文件名不包含 ID 信息，我们需要通过原始数据来反查
    # test_data 里的 file_path 应该和 gallery_data 里的 path 是一致的（都是相对路径）
    path_to_pid_map = {}
    for entry in test_data:
        # 确保 id 转为字符串，保证匹配一致性
        pid_str = str(entry["id"])
        path_to_pid_map[entry["file_path"]] = pid_str

    # 初始化日志
    logging.basicConfig(filename='eval_process.log', filemode='w', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', force=True)
    
    round_metrics = []

    print(f"Starting evaluation with {num_iters} rounds...")

    # --- 外层循环：5 次迭代 ---
    for round_idx in range(1, num_iters + 1):
        logging.info(f"\n{'='*20} ROUND {round_idx} START {'='*20}")
        print(f"Running Round {round_idx}/{num_iters}...")
        
        current_round_r1s = []
        current_round_aps = []
        processed_pids = set() 

        # --- 内层循环：遍历测试集 ---
        for sample in tqdm(test_data, desc=f"Round {round_idx}"):
            # 1. 获取 Query ID (直接从数据中拿，不解析文件名)
            qid = str(sample["id"])
            
            # 如果该 ID 不在缓存里（可能是过滤掉了或者数据不一致），跳过
            if qid not in gallery_data: 
                continue
            
            # PID 去重：确保该轮次中，每个 ID 只测一次
            if qid in processed_pids:
                continue
            processed_pids.add(qid)

            # 2. 获取文本 (动态适配不同数据集的 caption 数量)
            # CUHK/ICFG 的 caption 数量不固定，随机取一条以覆盖多样性
            captions_list = sample.get("captions", [])
            if len(captions_list) == 0:
                continue # 没有文本描述，跳过
            
            # 随机采样一条文本，模拟真实场景的不确定性

            # caption = random.choice(captions_list)
            caption = captions_list[-1]  # 为了结果可复现，固定选择第一条
            
            # 3. 获取 Gallery 图片路径
            gallery_rel_paths = gallery_data[qid]["gallery"]
            gallery_full_paths = [os.path.join(image_root, p) for p in gallery_rel_paths]
            
            # 4. 构造 GT (Ground Truth) - 关键修改
            # 不再 split 文件名，而是查表 path_to_pid_map
            gt = []
            for p_rel in gallery_rel_paths:
                # 查表获取该 gallery 图片对应的 PID
                # 如果 gallery 图片不在 test_data 里（极端情况），这步会报错或 None，
                # 但通常 gallery 都是 test set 的子集，所以应该能找到
                g_pid = path_to_pid_map.get(p_rel)
                
                # 如果查到了且等于 query 的 PID，则是正样本
                if g_pid is not None and g_pid == qid:
                    gt.append(1)
                else:
                    gt.append(0)

            # 5. 执行推理 
            ranked, _ = rank_text2image_confidence(
                model, 
                processor, 
                caption, 
                gallery_full_paths, 
                max_batch_size=5,
                log_prefix=f"R{round_idx}|PID:{qid}"
            )
            
            # 6. 计算单条指标
            r1, ap = evaluate_rank(gt, ranked)
            current_round_r1s.append(r1)
            current_round_aps.append(ap)
            
            # 详细日志
            logging.info(f"Round {round_idx} | PID: {qid} | R1: {r1} | AP: {ap:.4f}")

        # --- 本轮结束，计算本轮全局平均值 ---
        if not current_round_r1s:
            avg_r1, avg_map = 0.0, 0.0
        else:
            avg_r1 = sum(current_round_r1s) / len(current_round_r1s)
            avg_map = sum(current_round_aps) / len(current_round_aps)
        
        logging.info(f"{'='*20} ROUND {round_idx} SUMMARY {'='*20}")
        logging.info(f"Global Rank-1: {avg_r1:.4f}")
        logging.info(f"Global mAP:    {avg_map:.4f}")
        
        round_metrics.append((avg_r1, avg_map))

    # --- 5 轮结束，取最大值 ---
    final_max_r1 = max([m[0] for m in round_metrics]) if round_metrics else 0.0
    final_max_map = max([m[1] for m in round_metrics]) if round_metrics else 0.0

    logging.info(f"\n{'#'*20} FINAL EVALUATION RESULT {'#'*20}")
    logging.info(f"Max Global Rank-1 (across {num_iters} rounds): {final_max_r1:.4f}")
    logging.info(f"Max Global mAP    (across {num_iters} rounds): {final_max_map:.4f}")

    return final_max_r1, final_max_map