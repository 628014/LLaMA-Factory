import os
import json
import time
import gc
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 模型路径
MODEL_PATH = "/root/autodl-tmp/output/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_with_score_full"

# 2. 数据路径
DATA_PATH = "/root/home/wangrui/code/LLaMA-Factory/data/ICFG-PEDES_instruct_formatted_train_and_test_all.json"

# 3. 输出路径
OUTPUT_PATH = "/root/autodl-tmp/saves/qwen2_5vl-7b//lora/pre_retpreid_with_score/lora64_full_icfg_train_and_test_all/prediction_results.json"

# 4. 批处理大小
CHUNK_SIZE = 200 

# ===========================================

def load_data(filepath):
    print(f"正在读取元数据 (仅文本): {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"共加载 {len(data)} 条数据索引")
    return data

def main():
    # ----------------------------------------------------------------------
    # 1. 初始化 vLLM
    # ----------------------------------------------------------------------
    print("正在初始化 vLLM 引擎...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90, 
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    # ----------------------------------------------------------------------
    # 2. 准备循环
    # ----------------------------------------------------------------------
    full_dataset = load_data(DATA_PATH)
    total_items = len(full_dataset)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    print(f"开始推理并流式写入文件: {OUTPUT_PATH}")
    print(f"总数据: {total_items}, 分批: {CHUNK_SIZE}")
    
    start_time = time.time()
    pbar = tqdm(total=total_items, desc="进度", unit="img")

    # 【关键修改】：在循环开始前打开文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # 1. 手动写入 JSON 数组的开头
        f.write('[\n')
        
        is_first_item = True  # 用于控制逗号的标志位

        for i in range(0, total_items, CHUNK_SIZE):
            batch_slice = full_dataset[i : i + CHUNK_SIZE]
            
            # --- 构建输入 (加载图片) ---
            batch_inputs = []
            batch_mapping = [] 
            
            for item in batch_slice:
                try:
                    image_path = item["images"][0]
                    raw_prompt = item["messages"][0]["content"].replace("<image>", "").strip()
                    prompt_text = f"<|vision_start|><|image_pad|><|vision_end|>{raw_prompt}"
                    
                    image_obj = Image.open(image_path).convert("RGB")
                    
                    batch_inputs.append({
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image_obj}
                    })
                    batch_mapping.append(item)
                except Exception as e:
                    pbar.update(1)
                    continue

            if not batch_inputs:
                continue

            # --- 推理 ---
            outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)

            # --- 【关键修改】流式写入结果 ---
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                
                result_item = {
                    "file_path": batch_mapping[j]["images"][0],
                    "prediction": generated_text,
                    "ground_truth": batch_mapping[j]["messages"][0]["content"]
                }
                
                # 如果不是第一个元素，需要在前面加逗号和换行
                if not is_first_item:
                    f.write(',\n')
                else:
                    is_first_item = False
                
                # 写入当前的 json 对象 (缩进2格以保持美观)
                # 这里的 indent=2 会让每一项内部有缩进，稍微增加文件体积但易读
                # 如果想追求极致写入速度和最小体积，可以去掉 indent=2
                json_str = json.dumps(result_item, ensure_ascii=False, indent=2)
                f.write(json_str)

            # --- 立即刷新缓冲区 ---
            # 这样即使程序中途崩溃，文件里也保存了前面的数据
            f.flush() 

            # 更新进度条 & 清理内存
            pbar.update(len(batch_inputs))
            del batch_inputs
            del outputs
            del batch_mapping
            gc.collect() 

        # 3. 循环结束后，手动写入数组封口
        f.write('\n]')

    pbar.close()
    print(f"\n全部完成！总耗时: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    main()