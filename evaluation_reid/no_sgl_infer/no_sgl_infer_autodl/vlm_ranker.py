
import torch
import logging
from PIL import Image
import gc # 引入垃圾回收


# @torch.no_grad()
# def rank_text2image_confidence(model, processor, caption, gallery_full_paths, max_batch_size=50, log_prefix=""):
#     """
#     计算 Query 与 Gallery 图片的匹配度 Logits。
#     """
#     device = model.device
#     all_scores = []
#     # 引导前缀，与训练时的输出格式一致
#     prefix_str = "The best matches the following description is image "

#     # 1. 分批推理 (防止 OOM)
#     for i in range(0, len(gallery_full_paths), max_batch_size):
#         batch_paths = gallery_full_paths[i : i + max_batch_size]
#         n_current = len(batch_paths)
#         images = [Image.open(p).convert("RGB") for p in batch_paths]
#         # print('n_current :', n_current)
#         # 构造 Prompt
#         messages = [{
#             "role": "user",
#             "content": [{"type": "image"} for _ in range(n_current)] + 
#                        [{"type": "text", "text": f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\""}]
#         }]

#         # import pdb; pdb.set_trace()
#         # 应用 Chat 模板并手动拼接前缀
#         prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         full_prompt = prompt + prefix_str
        
#         inputs = processor(text=[full_prompt], images=[images], return_tensors="pt").to(device)
#         outputs = model(**inputs)
        
#         # 获取最后一个 Token 的 Logits
#         logits = outputs.logits[:, -1, :] 

#         # 提取 '1' 到 'n' 的 Logits
#         for j in range(1, n_current + 1):
#             token_id = processor.tokenizer.encode(str(j), add_special_tokens=False)[-1]
#             score = logits[0, token_id].item()
#             all_scores.append(score)
     
#     # 这里分批次处理了 为什么还是会outofmerry呢？

#     # 2. 全局排序
#     # ranked_indices 存储的是 gallery_full_paths 的下标
#     ranked_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    
#     # 3. 日志记录：模型到底选了哪张图？
#     best_global_idx = ranked_indices[0]
#     # 计算该图在它所属 batch 中的相对序号 (例如 image 3)
#     # 注意：这里的 batch_idx 只是为了日志展示，模拟模型输出 "image X"
#     best_batch_idx = (best_global_idx % max_batch_size) + 1
    
#     logging.info(f"[{log_prefix}] Model Answer: \"{prefix_str}{best_batch_idx}\" (Global Index: {best_global_idx})")
    
#     return ranked_indices, all_scores

"""
爆显存后重新构造如下：

"""
import torch
import logging
from PIL import Image
import gc
import traceback

def smart_resize(img, max_size=448):
    """
    智能调整图片大小，限制最长边，同时保持长宽比。
    Qwen2.5-VL 是动态分辨率，太大的图会产生太多 Token 导致 OOM。
    对于 ReID 任务，一般 384-512 足够辨认行人特征。
    """
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return img

def _run_inference_batch(model, processor, caption, image_paths, prefix_str, device):
    """
    内部推理函数：处理具体的模型调用、Prompt构造和显存清理
    """
    # 1. 加载并 Resize 图片 (关键修复点：防止 Token 爆炸)
    images = [smart_resize(Image.open(p).convert("RGB"), max_size=448) for p in image_paths]
    n_current = len(images)
    
    # 2. 构造 Prompt
    messages = [{
        "role": "user",
        "content": [{"type": "image"} for _ in range(n_current)] + 
                   [{"type": "text", "text": f"Select the image that best matches the following description of the pedestrian's appearance:\n\"{caption}\""}]
    }]
    
    # 3. 处理输入
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_prompt = text_prompt + prefix_str
    
    # 注意：Qwen2.5-VL 的 processor 会自动处理 padding
    inputs = processor(text=[full_prompt], images=images, return_tensors="pt", padding=True).to(device)
    
    # 4. 前向传播
    # use_cache=False 节省显存，因为我们不需要生成新的 token
    outputs = model(**inputs, use_cache=False)
    
    # 5. 提取 Logits (最后一个 Token)
    logits = outputs.logits[:, -1, :]
    
    scores = []
    for j in range(1, n_current + 1):
        # 获取对应选项 '1', '2', ... 的 token id
        token_id = processor.tokenizer.encode(str(j), add_special_tokens=False)[-1]
        scores.append(logits[0, token_id].item())
        
    # 6. 显式清理显存引用
    del inputs, outputs, logits, images
    return scores

@torch.no_grad()
def rank_text2image_confidence(model, processor, caption, gallery_full_paths, max_batch_size=5, log_prefix=""):
    device = model.device
    all_scores = []
    prefix_str = "The best matches the following description is image "

    # 遍历 Gallery
    for i in range(0, len(gallery_full_paths), max_batch_size):
        batch_paths = gallery_full_paths[i : i + max_batch_size]
        
        # --- 核心修改：尝试执行 Batch 推理，失败则降级 ---
        try:
            # 尝试以 max_batch_size 进行推理
            current_scores = _run_inference_batch(model, processor, caption, batch_paths, prefix_str, device)
            all_scores.extend(current_scores)
            
        except RuntimeError as e:
            # 捕获 CUDA OOM 或其他运行时错误
            err_msg = str(e)
            if "out of memory" in err_msg or "CUDA" in err_msg:
                print(f"[{log_prefix}] Batch inference failed (OOM). Switching to Batch=1 retry...")
            else:
                print(f"[{log_prefix}] Batch inference failed with error: {err_msg[:100]}... Switching to Batch=1.")
            
            # 紧急清理显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # --- 降级策略：逐张重试 (Batch Size = 1) ---
            for single_path in batch_paths:
                try:
                    single_score = _run_inference_batch(model, processor, caption, [single_path], prefix_str, device)
                    all_scores.extend(single_score)
                except Exception as e_single:
                    print(f"[{log_prefix}] Single image failed: {single_path}. Skipping. Error: {e_single}")
                    # 如果单张还挂，填一个极小值兜底，保证列表长度对齐
                    all_scores.append(-9999.0)
            
            # 再次清理，准备下一轮 Batch
            torch.cuda.empty_cache()
            gc.collect()

    # 排序逻辑
    if not all_scores:
        logging.error(f"[{log_prefix}] No scores computed!")
        return [], []

    ranked_indices = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    
    # 记录日志 (只记录第一名的选择)
    best_global_idx = ranked_indices[0]
    best_batch_idx = (best_global_idx % max_batch_size) + 1
    
    logging.info(f"[{log_prefix}] Model Answer: \"{prefix_str}{best_batch_idx}\" (Global Index: {best_global_idx})")
    
    return ranked_indices, all_scores