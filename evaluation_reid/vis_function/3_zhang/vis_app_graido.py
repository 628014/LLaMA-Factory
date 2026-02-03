import gradio as gr
import json
import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- 配置路径 ---
MODEL_PATH = "/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_with_score_full"
JSONL_PATH = "/home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/with_score_ccls/lora64_full_unfreeze/prediction_sim_score.jsonl"

# --- 全局加载缓存 ---
cache_map = {}
if os.path.exists(JSONL_PATH):
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 获取绝对路径作为 key
            try:
                img_path = data['images_path'][0]['path']
                cache_map[img_path] = data
            except: 
                print("Error processing line:", data)
            # import pdb; pdb.set_trace()
            
            

# --- 加载模型与处理器 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, # 强制使用 bf16 减少显存占用 [cite: 187]
    device_map="auto",
    offload_folder="offload")   # 显存不足时允许部分权重卸载到 CPU [cite: 164]
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def predict(image, img_path_input):
    """
    逻辑：
    1. 如果用户提供了图片路径且在 cache_map 中，直接返回缓存
    2. 否则，执行在线推理
    """
    # 优先检查路径匹配（针对 1k 测试集）
    if img_path_input and img_path_input in cache_map:
        item = cache_map[img_path_input]
        return item['predict'], item['label'], f"{item['sim_score']:.6f}"

    # 在线推理 (a-b 模式切换)
    prompt_text = (
        "###Task###\nYou are an expert in Re-Identification (ReID). "
        "Please refer to the following requirements... (此处填入您完整的Prompt模板)"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text, "N/A (在线推理模式无标准标注)", "N/A"

# --- 构建 Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 图 3-6: 行人细粒度属性描述生成与置信度评估系统")
    
    with gr.Row():
        # 左侧模块：图片上传与操作
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="上传图像 (Input Image)")
            img_path_box = gr.Textbox(label="输入图片绝对路径 (用于匹配测试集缓存)", placeholder="/home/path/to/img.jpg")
            run_btn = gr.Button("开始推理/检索", variant="primary")
            clear_btn = gr.Button("清除内容")
            
        # 右侧模块：预测与标准标注
        with gr.Column(scale=2):
            predict_out = gr.Textbox(label="预测描述与推理路径 (Predicted Result)", lines=12)
            label_out = gr.Textbox(label="标准标注 (Ground Truth Label)", lines=6)
            
    # 下方模块：相似度展示
    with gr.Row():
        sim_score_out = gr.Label(label="语义相似度评估说明 (Similarity Score)")

    run_btn.click(
        fn=predict, 
        inputs=[input_img, img_path_box], 
        outputs=[predict_out, label_out, sim_score_out]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", "", "", None), 
        outputs=[input_img, img_path_box, predict_out, label_out, sim_score_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)