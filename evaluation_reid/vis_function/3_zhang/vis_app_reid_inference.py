import gradio as gr
import torch
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import traceback

# ================= Configuration =================
MODELS = {
    "Fine-tuned (LoRA 128)": "/home/wangrui/code/LLaMA-Factory/output/lora128_with_progressive_48k_full_autodl",
    "Base (Qwen2.5-VL-7B)": "/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
}

# Prompt Templates from build_dataset_prob.py
PROMPT_TEMPLATES = {
    "Select a Task...": "",
    "Task 1: Img2Img One2One (Verify Identity)": 
        "<image><image>\nAre these two images showing the same person?",
        
    "Task 2: Img2Img One2Many (Retrieval)": 
        "Given the query image <image> of a pedestrian and a gallery set containing: the first image is <image>, the second image is <image>, please select the images from the gallery set that match the identity of the pedestrian in the query image.",
        
    "Task 3: Txt2Img One2One (Verify Caption)": 
        "Does the image <image> of the pedestrian match the following caption:\n\"[INSERT CAPTION HERE]\"",
        
    "Task 4: Txt2Img Many2One (Select Caption)": 
        "Given the following image <image>, which of the captions accurately describes it?\nThe first caption: \"[CAPTION 1]\"\nThe second caption: \"[CAPTION 2]\"\nThe third caption: \"[CAPTION 3]\"",
        
    "Task 5: Txt2Img One2Many (Select Image)": 
        "Given the following images: image 1 is <image>, image 2 is <image>, image 3 is <image>. Please select the image that best matches the following description of the pedestrian's appearance:\n\"[INSERT CAPTION HERE]\""
}

# Global State
model = None
processor = None
current_model_path = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_fn(model_name):
    global model, processor, current_model_path, device
    
    selected_path = MODELS.get(model_name)
    if not selected_path:
        return f"Error: Invalid model name {model_name}"
        
    if model is not None and current_model_path == selected_path:
        return f"Model '{model_name}' is already loaded."
        
    try:
        print(f"Loading model from {selected_path}...")
        
        # Load Processor
        processor = AutoProcessor.from_pretrained(selected_path)
        
        # Load Model
        # Using bfloat16 to save memory and match training usually
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            selected_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )
        
        current_model_path = selected_path
        return f"Successfully loaded '{model_name}'"
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return f"Failed to load model: {e}"

def update_prompt(task_name):
    return PROMPT_TEMPLATES.get(task_name, "")

def predict(model_name, image_files, prompt_text):
    global model, processor, current_model_path, device
    
    # 1. Ensure Model is Loaded
    if current_model_path != MODELS.get(model_name):
        load_status = load_model_fn(model_name)
        if "Failed" in load_status or "Error" in load_status:
            return load_status
            
    if model is None:
        return "Error: Model not loaded."
        
    if not image_files:
        return "Error: Please upload at least one image."
        
    if not prompt_text:
        return "Error: Please enter a prompt."
        
    # 2. Prepare Images
    loaded_images = []
    try:
        # Handle gr.File output (can be None, single path string, or list of paths)
        if image_files is None:
             return "Error: No images uploaded."
             
        # Normalize to list
        if not isinstance(image_files, list):
            file_paths = [image_files] # It might be a single object if file_count=1, but here file_count="multiple"
        else:
            file_paths = image_files

        for f in file_paths:
            # gr.File returns file paths (strings) or file-like objects depending on version.
            # In most recent Gradio, it returns paths (str) or NamedString.
            # We can try to open it.
            path = f.name if hasattr(f, 'name') else f
            img = Image.open(path).convert("RGB")
            loaded_images.append(img)
            
    except Exception as e:
        return f"Error loading images: {e}"
        
    if not loaded_images:
        return "Error: Failed to process loaded images."

    # 3. Construct Messages
    # Structure: [ {type: image, image: ...}, ..., {type: text, text: ...} ]
    content = []
    for img in loaded_images:
        content.append({"type": "image", "image": img})
        
    content.append({"type": "text", "text": prompt_text})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # 4. Inference
    try:
        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        # Trim input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except Exception as e:
        traceback.print_exc()
        return f"Inference Error: {str(e)}"

# ================= UI Layout =================
with gr.Blocks(title="ReID VL Inference", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ReID Vision-Language Inference App")
    gr.Markdown("Supports fine-tuned and base models with multiple task templates.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Model Selection
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Fine-tuned (LoRA 128)",
                label="Select Model"
            )
            load_btn = gr.Button("Load/Reload Model")
            status_output = gr.Textbox(label="System Status", value="Ready (Model not loaded yet)", interactive=False)
            
            gr.Markdown("### Image Upload")
            # Image Upload
            image_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload Images (Order Matters!)"
            )
            gallery_preview = gr.Gallery(label="Image Preview", columns=3, height=300)
            
            # Sync File upload to Gallery
            def show_in_gallery(files):
                if files:
                    paths = []
                    if not isinstance(files, list):
                        files = [files]
                    for f in files:
                        paths.append(f.name if hasattr(f, 'name') else f)
                    return paths
                return None
            
            image_input.change(show_in_gallery, inputs=image_input, outputs=gallery_preview)
            
        with gr.Column(scale=2):
            # Task Selection
            task_dropdown = gr.Dropdown(
                choices=list(PROMPT_TEMPLATES.keys()),
                value="Select a Task...",
                label="Select Task Template"
            )
            
            # Prompt Input
            prompt_input = gr.Textbox(
                label="Input Prompt",
                info="Edit placeholders like <image> or [CAPTION]. Ensure <image> count matches uploaded images.",
                lines=8,
                placeholder="Select a task to auto-fill or type your own prompt..."
            )
            
            # Auto-fill prompt when task changes
            task_dropdown.change(update_prompt, inputs=task_dropdown, outputs=prompt_input)
            
            # Run Button
            run_btn = gr.Button("Run Inference", variant="primary")
            
            # Output
            output_box = gr.Textbox(label="Model Output", lines=10)
            
    # Event Handlers
    load_btn.click(
        load_model_fn,
        inputs=[model_dropdown],
        outputs=[status_output]
    )
    
    run_btn.click(
        predict,
        inputs=[model_dropdown, image_input, prompt_input],
        outputs=[output_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
