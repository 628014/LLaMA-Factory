import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
from change_to_qwen_input_format import convert_five_to_qwen

def number_to_ordinal(num: int) -> str:
    """
    将数字转换为英文序数词
    :param num: 输入数字（需≥1）
    :return: 英文序数词，如 1→first, 2→second, 3→third...
    """
    # 特殊序数词规则（1-20 + 整十数）
    ordinal_exceptions = {
        1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
        6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
        11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth',
        15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth',
        19: 'nineteenth', 20: 'twentieth', 30: 'thirtieth', 40: 'fortieth',
        50: 'fiftieth', 60: 'sixtieth', 70: 'seventieth', 80: 'eightieth',
        90: 'ninetieth'
    }
    # 优先匹配特殊规则
    if num in ordinal_exceptions:
        return ordinal_exceptions[num]
    # 普通数字规则：基数词 + th（如 21→twenty-first）
    else:
        # 拆分十位和个位
        tens = num // 10
        units = num % 10
        # 基数词（十位部分）
        tens_words = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        # 拼接：十位基数词 + 连字符 + 个位序数词
        return f"{tens_words[tens]}-{ordinal_exceptions[units]}"
# 图像的一对一数据集构建，基于0.5的概率选择同一人或不同人
def build_image2image_dataset_prob(
    caption_json,
    image_root,
    save_path,
    split="train",
    num_samples=5000,
    positive_prob=0.5,
    seed=42
):
    """
    按论文设置：
    - 每个样本以 positive_prob 概率为同一人（Yes）
    - 否则为不同人（No）

    num_samples:
        None  -> 使用所有可用 id 构造
        int   -> 随机采样 num_samples 个 pair
    """

    random.seed(seed)

    data = json.load(open(caption_json))
    data = [d for d in data if d["split"] == split]

    # id -> images
    id2imgs = defaultdict(list)
    for d in data:
        id2imgs[d["id"]].append(d["img_path"])

    # 只保留至少有 2 张图的 id（正样本才可能）
    valid_ids = [pid for pid, imgs in id2imgs.items() if len(imgs) >= 2]
    all_ids = list(id2imgs.keys())

    assert len(valid_ids) > 0, "No valid identities with >=2 images."

    samples = []
    sample_id = 0

    # 决定生成多少条
    if num_samples is None:
        num_samples = len(data)

    for _ in tqdm(range(num_samples)):
        is_positive = random.random() < positive_prob

        # ---------- Yes：同一人 ----------
        if is_positive:
            pid = random.choice(valid_ids)
            img1, img2 = random.sample(id2imgs[pid], 2)

            answer = "Yes."
            images = [img1, img2]

        # ---------- No：不同人 ----------
        else:
            pid1, pid2 = random.sample(all_ids, 2)
            img1 = random.choice(id2imgs[pid1])
            img2 = random.choice(id2imgs[pid2])

            answer = "No."
            images = [img1, img2]

        samples.append({
            "id": f"img2img_{sample_id}",
            "images": [
                os.path.join(image_root, images[0]),
                os.path.join(image_root, images[1])
            ],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image><image>\nAre these two images showing the same person?"
                },
                {
                    "from": "assistant",
                    "value": answer
                }
            ]
        })

        sample_id += 1

    json.dump(samples, open(save_path, "w"), indent=2)
    print(f"Saved {len(samples)} samples to {save_path}")

# 构造 image→image one-to-many 数据集
def build_image2image_one2many(
    caption_json,
    image_root,
    save_path,
    split="train",
    num_samples=50000,
    N_range=(2, 4),     # gallery size N
    n_range=(1, 3),     # positive matches n
    seed=42
):
    """
    构造 image→image one-to-many 数据集（ChatReID 论文风格）
    """

    random.seed(seed)

    data = json.load(open(caption_json))
    data = [d for d in data if d["split"] == split]

    # id -> images
    id2imgs = defaultdict(list)
    for d in data:
        id2imgs[d["id"]].append(d["img_path"])

    # 可作为 query 的 id（至少 2 张图）
    valid_ids = [pid for pid, imgs in id2imgs.items() if len(imgs) >= 2]
    all_ids = list(id2imgs.keys())

    samples = []
    sample_id = 0

    for _ in tqdm(range(num_samples)):
        # ---------- 随机参数 ----------
        N = random.randint(*N_range)
        n = random.randint(*n_range)
        n = min(n, N - 1)  # 至少留一个负样本

        # ---------- query ----------
        pid = random.choice(valid_ids)
        query_img = random.choice(id2imgs[pid])

        # ---------- 正样本 ----------
        pos_candidates = [img for img in id2imgs[pid] if img != query_img]
        pos_imgs = random.sample(
            pos_candidates,
            min(n, len(pos_candidates))
        )

        # ---------- 负样本 ----------
        neg_imgs = []
        neg_ids = [i for i in all_ids if i != pid]

        while len(neg_imgs) < (N - len(pos_imgs)):
            neg_pid = random.choice(neg_ids)
            neg_img = random.choice(id2imgs[neg_pid])
            neg_imgs.append(neg_img)

        # ---------- gallery ----------
        gallery_imgs = pos_imgs + neg_imgs
        random.shuffle(gallery_imgs)

        # ---------- images 顺序 ----------
        images = [query_img] + gallery_imgs

        # ---------- prompt ----------
        gallery_tokens = "".join(
            [f"<image>" for i in range(len(gallery_imgs))]
        )

        human_prompt = (
            f"Given the query image <image> of a pedestrian and a gallery set "
            f"{gallery_tokens}, please select the images from the gallery set "
            f"that match the identity of the pedestrian in the query image."
        )

        # ---------- answer ----------
        answer_tokens = []
        for idx, img in enumerate(gallery_imgs):
            if img in pos_imgs:
                ordinal_word = number_to_ordinal(idx + 2)
                answer_tokens.append(f"{ordinal_word} image")
                # answer_tokens.append(f"image {idx+2}")

        answer_str = ", ".join(answer_tokens)
        pre_str = 'The images that match the pedestrian identity of the query image are: '
        answer_str = pre_str + answer_str.strip() + "."

        samples.append({
            "id": f"img2img_o2m_{sample_id}",
            "images": [os.path.join(image_root, p) for p in images],
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "assistant", "value": answer_str}
            ]
        })

        sample_id += 1

    json.dump(samples, open(save_path, "w"), indent=2)
    print(f"Saved {len(samples)} one-to-many samples to {save_path}")

# text→image 的 one-to-one matching，而且使用 0.75 正样本概率

def build_text2image_one2one(
    caption_json,
    image_root,
    save_path,
    split="train",
    num_samples=30000,
    positive_prob=0.75,
    seed=42
):
    """
    Text-to-Image One-to-One Matching
    正样本概率 = positive_prob (默认 0.75)
    """

    random.seed(seed)

    data = json.load(open(caption_json))
    data = [d for d in data if d["split"] == split]

    # id -> list of samples
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)

    all_ids = list(id2items.keys())
    samples = []
    sample_id = 0

    for _ in tqdm(range(num_samples)):
        is_positive = random.random() < positive_prob

        # ---------- 正样本 ----------
        if is_positive:
            item = random.choice(data)
            img_path = item["img_path"]
            caption = random.choice(item["captions"])
            answer = "Yes."

        # ---------- 负样本 ----------
        else:
            img_item = random.choice(data)
            img_path = img_item["img_path"]
            img_pid = img_item["id"]

            # 选一个不同 id 的 caption
            neg_pid = random.choice([pid for pid in all_ids if pid != img_pid])
            neg_item = random.choice(id2items[neg_pid])
            caption = random.choice(neg_item["captions"])

            answer = "No."

        human_prompt = (
            f"Does the image <image> of the pedestrian match the following caption:\n"
            f"\"{caption}\""
        )

        samples.append({
            "id": f"txt2img_o2o_{sample_id}",
            "images": [os.path.join(image_root, img_path)],
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "assistant",
                    "value": answer
                }
            ]
        })

        sample_id += 1

    json.dump(samples, open(save_path, "w"), indent=2)
    print(f"Saved {len(samples)} text-to-image one-to-one samples to {save_path}")

# text→image 的 many-to-one matching，多 caption 选择 一个，哪个最匹配这张图？
def build_text2image_many2one(
    caption_json,
    image_root,
    save_path,
    split="train",
    num_samples=30000,
    seed=42
):
    """
    Text-to-Image Many-to-One Matching (Fixed 3 captions)
    """

    random.seed(seed)

    data = json.load(open(caption_json))
    data = [d for d in data if d["split"] == split]

    # id -> items
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)

    all_items = data
    all_ids = list(id2items.keys())

    samples = []
    sample_id = 0

    for _ in tqdm(range(num_samples)):
        # ---------- query image ----------
        item = random.choice(all_items)
        img_path = item["img_path"]
        img_pid = item["id"]

        # ---------- 正确 caption（3 选 1） ----------
        gt_caption = random.choice(item["captions"])

        # ---------- 负 caption（来自其他 ID） ----------
        neg_captions = []
        while len(neg_captions) < 2:
            neg_pid = random.choice(all_ids)
            if neg_pid == img_pid:
                continue
            neg_item = random.choice(id2items[neg_pid])
            neg_caption = random.choice(neg_item["captions"])
            neg_captions.append(neg_caption)

        captions = [gt_caption] + neg_captions

        # ---------- 打乱顺序 ----------
        random.shuffle(captions)
        gt_index = captions.index(gt_caption)

        # ---------- 构造 prompt ----------
        caption_lines = [
            f"The first caption: \"{captions[0]}\"",
            f"The second caption: \"{captions[1]}\"",
            f"The third caption: \"{captions[2]}\""
        ]

        human_prompt = (
            "Given the following image <image>, which of the captions accurately describes it?\n"
            + "\n".join(caption_lines)
        )

        assistant_answer = [
            "The first caption.",
            "The second caption.",
            "The third caption."
        ][gt_index]

        samples.append({
            "id": f"txt2img_m2o_{sample_id}",
            "images": [os.path.join(image_root, img_path)],
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "assistant",
                    "value": assistant_answer
                }
            ]
        })

        sample_id += 1

    json.dump(samples, open(save_path, "w"), indent=2)
    print(f"Saved {len(samples)} samples to {save_path}")

# text→image 的 one-to-many retrieval 多张图 + 一个caption，选哪个图最合适这个caption
def build_text2image_one2many(
    caption_json,
    image_root,
    save_path,
    split="train",
    num_samples=20000,
    gallery_size_range=(3, 6),
    seed=42
):
    """
    Text-to-Image One-to-Many Retrieval
    """

    random.seed(seed)

    data = json.load(open(caption_json))
    data = [d for d in data if d["split"] == split]

    # id -> items
    id2items = defaultdict(list)
    for d in data:
        id2items[d["id"]].append(d)

    all_items = data
    all_ids = list(id2items.keys())

    samples = []
    sample_id = 0

    for _ in tqdm(range(num_samples)):
        # ---------- 正样本 ----------
        pos_item = random.choice(all_items)
        pos_pid = pos_item["id"]
        caption = random.choice(pos_item["captions"])
        pos_img = pos_item["img_path"]

        # ---------- gallery size ----------
        N = random.randint(*gallery_size_range)

        images = [pos_img]

        # ---------- 负样本图像 ----------
        while len(images) < N:
            neg_pid = random.choice(all_ids)
            if neg_pid == pos_pid:
                continue
            neg_item = random.choice(id2items[neg_pid])
            images.append(neg_item["img_path"])

        # ---------- shuffle ----------
        random.shuffle(images)
        gt_index = images.index(pos_img) + 1  # 1-based

        all_image_tokens = "".join(
            [f"<image>" for i in range(len(images))]
        )

        # ---------- prompt ----------
        human_prompt = (
            f"Given the following images {all_image_tokens}, "
            "select the image that best matches the following description of the pedestrian's appearance:\n"
            f"\"{caption}\""
        )

        assistant_answer = (
            f'The best matches the following description is image {gt_index}.'
            # f"image_{gt_index} matches the following description."
        )

        samples.append({
            "id": f"txt2img_o2m_{sample_id}",
            "images": [os.path.join(image_root, p) for p in images],
            "conversations": [
                {
                    "from": "human",
                    "value": human_prompt
                },
                {
                    "from": "assistant",
                    "value": assistant_answer
                }
            ]
        })

        sample_id += 1

    json.dump(samples, open(save_path, "w"), indent=2)
    print(f"Saved {len(samples)} samples to {save_path}")





caption_json = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
image_root = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs"
# save_path = "/home/wangrui/code/LLaMA-Factory/data/"



output_path = [
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2one_5k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2many_5k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2one_2k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_many2one_2k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_2k_new.json', 
]

input_path = [
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2one_5k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2many_5k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2one_2k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_many2one_2k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_2k.json', 
]

# 1对1 image→image，0.5概率同一人

build_image2image_dataset_prob(
    caption_json= caption_json,
    image_root= image_root,
    save_path=input_path[0],
    split="train",
    num_samples=5000,
    positive_prob=0.5
)

# 1对多 image→image 

build_image2image_one2many(
    caption_json=caption_json,
    image_root=image_root,
    save_path=input_path[1],
    split="train",
    num_samples=5000,
    N_range=(4, 8),
    n_range=(1, 3)
)

# 一对一 text→image 这张图的和caption匹配吗？ 0.75概率同一人
build_text2image_one2one(
    caption_json=caption_json,
    image_root=image_root,
    save_path=input_path[2],
    split="train",
    num_samples=2000,
    positive_prob=0.75
)

# 多对一 text→image 这张图对应哪个caption？（3选1）

build_text2image_many2one(
    caption_json=caption_json,
    image_root=image_root,
    save_path=input_path[3],
    split="train",
    num_samples=2000
)

# 一对多 text→image 这段caption对应哪个图？（多图选一图）
build_text2image_one2many(
    caption_json=caption_json,
    image_root=image_root,
    save_path=input_path[4],
    split="train",
    num_samples=2000,
    gallery_size_range=(3, 6)
)

convert_five_to_qwen(input_path, output_path)







