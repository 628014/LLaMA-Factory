import gradio as gr
import json
import os
import io
import base64
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
                img_path = data['images_path'][0]
                cache_map[img_path] = data
            except Exception as e:
                print(f"Error processing line: {e}, data: {data}")
            
            

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
        # 加载缓存中的图片
        try:
            cached_image = Image.open(img_path_input)
        except Exception as e:
            cached_image = None
            print(f"Error loading image: {e}")
        # 有图片时，显示右侧列
        return cached_image, gr.update(visible=True), item['predict'], item['label'], f"{item['sim_score']:.6f}"

    # 检查是否有上传图片
    if image is None:
        return None, gr.update(visible=False), "错误：请上传图片或输入有效的缓存路径", "N/A", "N/A"

    # 在线推理 (a-b 模式切换)
    prompt_text = "<image>\n###Task###\nYou are an expert in the field of Re-Identification (ReID). Please refer to the following requirements to understand the general and specific features of the image, and then combine all the features to return a descriptive language, and then judge the matching score with the image based on the given language.\n###Requirement###\nChoose one color from black, white, red, purple, yellow, blue, green, pink, gray, and brown.\n1、Gender: male、female\n2、Age: teenager、young、adult、old\n3、Body Build: fat、slightly fat、thin\n4、Length Hair: long hair、medium-length hair、short hair、bald\n5、Wearing hat: yes、no; if yes, the color is: XXX\n6、Carrying backpack: yes、no; if yes, the color is:  XXX\n7、Carrying handbag or bag: yes、no; if yes, the color is:  XXX\n8、Upper Body\n8.1、Sleeve Length: long sleeve、short sleeve\n8.2、Inner Lining: yes、no; if yes, the color is:\n8.3、Color of upper-body: XXX\n9、Lower Body\n9.1、Length of lower-body: long lower-body clothing、short\n9.2、Type of lower-body: dress、pants\n9.3、Color of lower-body: XXX\n10、Shoe Color: XXX\n11、Emotion: Happy、Surprised、Sad、Angry、Disgusted、Fearful、Neutral、Other\n12、Gait and Posture: XXX\n###Output###\nThe final description of the image is XXX. In summary, the degree of relevance to the image is XXX.\n"
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
    
    # 用户上传图片时，右侧不展示图片，隐藏右侧列
    return None, gr.update(visible=False), output_text, "N/A (在线推理模式无标准标注)", "N/A"

# --- 构建 Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container { height: 100vh; } .column { height: 80vh; }") as demo:
    gr.Markdown("# 图 3-6: 行人细粒度属性描述生成与置信度评估系统")
    
    with gr.Row():
        # 左侧模块：图片上传与操作
        with gr.Column(scale=1, min_width=300):
            input_img = gr.Image(type="pil", label="上传图像 (Input Image)", height="45vh")
            img_path_box = gr.Textbox(label="输入图片绝对路径 (用于匹配测试集缓存)", placeholder="/home/path/to/img.jpg")
            run_btn = gr.Button("开始推理/检索", variant="primary")
            clear_btn = gr.Button("清除内容")
            
        # 中间模块：预测与标准标注
        with gr.Column(scale=2, min_width=400):
            predict_out = gr.Textbox(label="预测描述与推理路径 (Predicted Result)", lines=20)
            label_out = gr.Textbox(label="标准标注 (Ground Truth Label)", lines=10)
            
        # 右侧模块：图片展示和相似度（可动态显示/隐藏）
        right_column = gr.Column(scale=1, min_width=300, visible=False)
        with right_column:
            display_img = gr.Image(type="pil", label="当前图片", height="65vh")
            sim_score_out = gr.Label(label="语义相似度评估说明 (Similarity Score)")

    run_btn.click(
        fn=predict, 
        inputs=[input_img, img_path_box], 
        outputs=[display_img, right_column, predict_out, label_out, sim_score_out]
    )
    
    clear_btn.click(
        fn=lambda: (None, gr.update(visible=False), None, "", "", None), 
        outputs=[input_img, right_column, display_img, img_path_box, predict_out, label_out, sim_score_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)