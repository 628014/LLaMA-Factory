import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# 工具函数
# ==============================================================================

def number_to_ordinal(num: int) -> str:
    """将数字转换为英文序数词"""
    ordinal_exceptions = {
        1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
        6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
        11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth',
        15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth',
        19: 'nineteenth', 20: 'twentieth', 30: 'thirtieth', 40: 'fortieth',
        50: 'fiftieth', 60: 'sixtieth', 70: 'seventieth', 80: 'eightieth',
        90: 'ninetieth'
    }
    if num in ordinal_exceptions:
        return ordinal_exceptions[num]
    else:
        tens = num // 10
        units = num % 10
        tens_words = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        return f"{tens_words[tens]}-{ordinal_exceptions[units]}"

def load_reid_data(json_path, split="train"):
    """
    加载并标准化数据格式。
    ICFG/CUHK 使用 'file_path', RSTP 使用 'img_path'。
    统一转换为包含 'img_path', 'id'(str), 'captions' 的字典列表。
    """
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 过滤 split
    raw_data = [d for d in raw_data if d.get("split") == split]
    
    normalized_data = []
    for item in raw_data:
        # 统一图片路径字段: 优先取 img_path (RSTP), 若无则取 file_path (ICFG/CUHK)
        img_path = item.get("img_path", item.get("file_path"))
        if not img_path:
            continue
            
        # 统一 ID 为字符串，防止 int 和 str 比较出错
        pid = str(item["id"])
        
        normalized_data.append({
            "id": pid,
            "img_path": img_path,
            "captions": item["captions"]
        })
    
    print(f"Loaded {len(normalized_data)} items for split '{split}'.")
    return normalized_data

# ==============================================================================
# 构建任务函数 (直接生成 LLaMA-Factory Qwen2-VL 格式)
# ==============================================================================

def build_img2img_o2o(data, image_root, num_samples=5000, positive_prob=0.5, start_id=0):
    """任务1: Image-to-Image One-to-One"""
    id2imgs = defaultdict(list)
    for d in data:
        id2imgs[d["id"]].append(d["img_path"])
    
    valid_ids = [pid for pid, imgs in id2imgs.items() if len(imgs) >= 2]
    all_ids = list(id2imgs.keys())
    
    samples = []
    for i in range(num_samples):
        is_positive = random.random() < positive_prob
        
        if is_positive:
            pid = random.choice(valid_ids)
            img1, img2 = random.sample(id2imgs[pid], 2)
            answer = "Yes."
        else:
            pid1, pid2 = random.sample(all_ids, 2)
            # 确保不同人
            while pid1 == pid2:
                pid2 = random.choice(all_ids)
            img1 = random.choice(id2imgs[pid1])
            img2 = random.choice(id2imgs[pid2])
            answer = "No."

        samples.append({
            "images": [
                os.path.join(image_root, img1),
                os.path.join(image_root, img2)
            ],
            "messages": [
                {
                    "role": "user", 
                    "content": "<image><image>\nAre these two images showing the same person?"
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        })
    return samples

def build_img2img_o2m(data, image_root, num_samples=5000, N_range=(3, 6), n_range=(1, 3), start_id=0):
    """
    任务2: Image-to-Image One-to-Many
    Prompt: "...the second image is <image>..."
    Answer: "...second image, third image..."
    """
    id2imgs = defaultdict(list)
    for d in data:
        id2imgs[d["id"]].append(d["img_path"])
    
    valid_ids = [pid for pid, imgs in id2imgs.items() if len(imgs) >= 2]
    all_ids = list(id2imgs.keys())
    
    samples = []
    for i in range(num_samples):
        N = random.randint(*N_range)
        n = random.randint(*n_range)
        n = min(n, N - 1)
        
        pid = random.choice(valid_ids)
        query_img = random.choice(id2imgs[pid])
        
        pos_candidates = [img for img in id2imgs[pid] if img != query_img]
        k = min(n, len(pos_candidates))
        if k == 0: continue
        pos_imgs = random.sample(pos_candidates, k)
        
        neg_imgs = []
        while len(neg_imgs) < (N - len(pos_imgs)):
            neg_pid = random.choice(all_ids)
            if neg_pid == pid: continue
            neg_img = random.choice(id2imgs[neg_pid])
            neg_imgs.append(neg_img)
            
        gallery_imgs = pos_imgs + neg_imgs
        random.shuffle(gallery_imgs)
        
        images = [query_img] + gallery_imgs
        full_image_paths = [os.path.join(image_root, p) for p in images]
        
        gallery_desc_parts = []
        answer_tokens = []
        
        for idx, img in enumerate(gallery_imgs):
            # idx=0 (Gallery第1张) -> Global Index=2 (Query是1)
            global_idx = idx + 2
            ordinal_word = number_to_ordinal(global_idx)
            
            gallery_desc_parts.append(f"the {ordinal_word} image is <image>")
            
            if img in pos_imgs:
                answer_tokens.append(f"{ordinal_word} image")
        
        gallery_tokens_str = ", ".join(gallery_desc_parts)
        
        human_prompt = (
            f"Given the query image <image> of a pedestrian and a gallery set containing: "
            f"{gallery_tokens_str}, please select the images from the gallery set "
            f"that match the identity of the pedestrian in the query image."
        )
        
        answer_str = "The images that match the pedestrian identity of the query image are: " + ", ".join(answer_tokens) + "."
        
        samples.append({
            "images": full_image_paths,
            "messages": [
                {"role": "user", "content": human_prompt},
                {"role": "assistant", "content": answer_str}
            ]
        })
    return samples

def build_txt2img_o2o(data, image_root, num_samples=2000, positive_prob=0.5, start_id=0):
    """任务3: Text-to-Image One-to-One"""
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)
    all_ids = list(id2items.keys())
    
    samples = []
    for i in range(num_samples):
        is_positive = random.random() < positive_prob
        
        if is_positive:
            item = random.choice(data)
            img_path = item["img_path"]
            caption = random.choice(item["captions"])
            answer = "Yes."
        else:
            img_item = random.choice(data)
            img_path = img_item["img_path"]
            img_pid = img_item["id"]
            
            neg_pid = random.choice(all_ids)
            while neg_pid == img_pid:
                neg_pid = random.choice(all_ids)
            
            neg_item = random.choice(id2items[neg_pid])
            caption = random.choice(neg_item["captions"])
            answer = "No."
            
        human_prompt = (
            f"Does the image <image> of the pedestrian match the following caption:\n"
            f"\"{caption}\""
        )
        
        samples.append({
            "images": [os.path.join(image_root, img_path)],
            "messages": [
                {"role": "user", "content": human_prompt},
                {"role": "assistant", "content": answer}
            ]
        })
    return samples

def build_txt2img_m2o(data, image_root, num_samples=2000, start_id=0):
    """任务4: Text-to-Image Many-to-One (3 captions choose 1)"""
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)
    all_ids = list(id2items.keys())
    
    samples = []
    for i in range(num_samples):
        item = random.choice(data)
        img_path = item["img_path"]
        gt_caption = random.choice(item["captions"])
        
        neg_captions = []
        while len(neg_captions) < 2:
            neg_pid = random.choice(all_ids)
            if neg_pid == item["id"]: continue
            neg_item = random.choice(id2items[neg_pid])
            neg_captions.append(random.choice(neg_item["captions"]))
            
        captions = [gt_caption] + neg_captions
        random.shuffle(captions)
        gt_index = captions.index(gt_caption)
        
        caption_lines = [
            f"The first caption: \"{captions[0]}\"",
            f"The second caption: \"{captions[1]}\"",
            f"The third caption: \"{captions[2]}\""
        ]
        
        human_prompt = (
            "Given the following image <image>, which of the captions accurately describes it?\n"
            + "\n".join(caption_lines)
        )
        
        ans_map = ["The first caption.", "The second caption.", "The third caption."]
        
        samples.append({
            "images": [os.path.join(image_root, img_path)],
            "messages": [
                {"role": "user", "content": human_prompt},
                {"role": "assistant", "content": ans_map[gt_index]}
            ]
        })
    return samples

def build_txt2img_o2m(data, image_root, num_samples=2000, gallery_size_range=(3, 6), start_id=0):
    """
    任务5: Text-to-Image One-to-Many Retrieval
    Prompt: "image 1 is <image>, image 2 is <image>..."
    Answer: "image X"
    """
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)
    all_ids = list(id2items.keys())
    
    samples = []
    for i in range(num_samples):
        pos_item = random.choice(data)
        caption = random.choice(pos_item["captions"])
        pos_img = pos_item["img_path"]
        
        N = random.randint(*gallery_size_range)
        images = [pos_img]
        
        while len(images) < N:
            neg_pid = random.choice(all_ids)
            if neg_pid == pos_item["id"]: continue
            neg_item = random.choice(id2items[neg_pid])
            images.append(neg_item["img_path"])
            
        random.shuffle(images)
        gt_index = images.index(pos_img) + 1
        
        image_tokens_parts = []
        for idx in range(len(images)):
            image_tokens_parts.append(f"image {idx + 1} is <image>")
        
        all_image_tokens = ", ".join(image_tokens_parts)
        
        human_prompt = (
            f"Given the following images: {all_image_tokens}. "
            "Please select the image that best matches the following description of the pedestrian's appearance:\n"
            f"\"{caption}\""
        )
        
        assistant_answer = f"The best matches the following description is image {gt_index}."
        
        samples.append({
            "images": [os.path.join(image_root, p) for p in images],
            "messages": [
                {"role": "user", "content": human_prompt},
                {"role": "assistant", "content": assistant_answer}
            ]
        })
    return samples

# ==============================================================================
# 主逻辑
# ==============================================================================

def build_dataset_for_source(source_name, caption_path, image_root, output_json, seed=42):
    print(f"\n======== Building 16k dataset for: {source_name} ========")
    random.seed(seed)
    
    # 1. 加载数据
    data = load_reid_data(caption_path, split="train")
    
    all_samples = []
    
    # 2. 生成各个任务的数据
    print("Building Task 1: Img2Img One2One (5k)...")
    task1 = build_img2img_o2o(data, image_root, num_samples=5000, positive_prob=0.5)
    all_samples.extend(task1)
    
    print("Building Task 2: Img2Img One2Many (5k)...")
    task2 = build_img2img_o2m(data, image_root, num_samples=5000, N_range=(3, 6), n_range=(1, 3))
    all_samples.extend(task2)
    
    print("Building Task 3: Txt2Img One2One (2k)...")
    task3 = build_txt2img_o2o(data, image_root, num_samples=2000, positive_prob=0.5)
    all_samples.extend(task3)
    
    print("Building Task 4: Txt2Img Many2One (2k)...")
    task4 = build_txt2img_m2o(data, image_root, num_samples=2000)
    all_samples.extend(task4)
    
    print("Building Task 5: Txt2Img One2Many (2k)...")
    task5 = build_txt2img_o2m(data, image_root, num_samples=8000, gallery_size_range=(3, 6))
    all_samples.extend(task5)
    
    # 3. 乱序并保存
    random.shuffle(all_samples)
    print(f"Total samples generated: {len(all_samples)}")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_json}")


if __name__ == "__main__":
    # 1. RSTPReid 配置
    CAPTION_PATH_RSTP = "/root/autodl-tmp/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
    IMAGE_PATH_RSTP = "/root/autodl-tmp/MLLM4Text-ReID-main/data/RSTPReid/imgs"
    OUTPUT_RSTP = "/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_rstp_22k_unified_autodl.json"

    # 2. ICFG-PEDES 配置
    CAPTION_PATH_ICFG = "/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/ICFG-PEDES_match_score_54522.json"
    IMAGE_PATH_ICFG = "/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/imgs"
    OUTPUT_ICFG = "/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_icfg_22k_unified_autodl.json"

    # 3. CUHK-PEDES 配置
    CAPTION_PATH_CUHK = "/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/reid_raw_match_score_40201.json"
    IMAGE_PATH_CUHK = "/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/imgs/"
    OUTPUT_CUHK = "/root/home/wangrui/code/LLaMA-Factory/data/mllm_reid_cuhk_22k_unified_autodl.json"

    # === 执行生成 ===
    
    # 生成 RSTPReid 数据集 (16k)
    build_dataset_for_source(
        source_name="RSTPReid",
        caption_path=CAPTION_PATH_RSTP,
        image_root=IMAGE_PATH_RSTP,
        output_json=OUTPUT_RSTP,
        seed=42
    )

    # 生成 ICFG 数据集 (16k)
    build_dataset_for_source(
        source_name="ICFG-PEDES",
        caption_path=CAPTION_PATH_ICFG,
        image_root=IMAGE_PATH_ICFG,
        output_json=OUTPUT_ICFG,
        seed=123 # 使用不同的随机种子
    )

    # 生成 CUHK 数据集 (16k)
    build_dataset_for_source(
        source_name="CUHK-PEDES",
        caption_path=CAPTION_PATH_CUHK,
        image_root=IMAGE_PATH_CUHK,
        output_json=OUTPUT_CUHK,
        seed=999 # 使用不同的随机种子
    )
    
    print("\nAll datasets (RSTP, ICFG, CUHK) generated successfully.")