import os
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import json


# ================= 配置区域 =================
INPUT_FILE = '/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/ICFG-PEDES.json'
OUTPUT_FILE = '/root/home/wangrui/code/LLaMA-Factory/data/ICFG-PEDES_instruct_formatted_train_and_test_all.json'
IMG_PREFIX = '/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/imgs/'

# ================= 模板定义 (放在全局以便进程调用) =================
PROMPT_TEMPLATE = """<image>
###Task###
You are an expert in the field of Re-Identification (ReID). Please refer to the following requirements to understand the general and specific features of the image, and then combine all the features to return a descriptive language, and then judge the matching score with the image based on the given language.The description of this image is already provided as:{caption}
###Requirement###
Choose one color from black, white, red, purple, yellow, blue, green, pink, gray, and brown.
1、Gender: male、female
2、Age: teenager、young、adult、old
3、Body Build: fat、slightly fat、thin
4、Length Hair: long hair、medium-length hair、short hair、bald
5、Wearing hat: yes、no; if yes, the color is: XXX
6、Carrying backpack: yes、no; if yes, the color is:  XXX
7、Carrying handbag or bag: yes、no; if yes, the color is:  XXX
8、Upper Body
8.1、Sleeve Length: long sleeve、short sleeve
8.2、Inner Lining: yes、no; if yes, the color is:
8.3、Color of upper-body: XXX
9、Lower Body
9.1、Length of lower-body: long lower-body clothing、short
9.2、Type of lower-body: dress、pants
9.3、Color of lower-body: XXX
10、Shoe Color: XXX
11、Emotion: Happy、Surprised、Sad、Angry、Disgusted、Fearful、Neutral、Other
12、Gait and Posture: XXX
###Output###
The final description of the image is XXX. In summary, the degree of relevance to the image is XXX.
"""

# ================= 处理函数 (Worker) =================
def process_batch(data_chunk):
    """
    处理一小批数据，返回处理后的结果列表
    """
    results = []
    for item in data_chunk:
        # 路径处理
        rel_path = item.get('file_path', '')
        # 移除开头的 / 避免路径拼接错误
        clean_rel_path = rel_path.lstrip('/')
        # 拼接绝对路径
        full_image_path = os.path.join(IMG_PREFIX, clean_rel_path)
        
        # 遍历captions
        captions = item.get('captions', [])
        for caption in captions:
            # 快速字符串替换
            user_content = PROMPT_TEMPLATE.replace('{caption}', caption)
            
            new_entry = {
                "messages": [
                    {
                        "content": user_content,
                        "role": "user"
                    },
                    {
                        "content": "",
                        "role": "assistant"
                    }
                ],
                "images": [
                    full_image_path
                ]
            }
            results.append(new_entry)
    return results

# ================= 主程序 =================
def main():
    start_time = time.time()
    
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"Error: 文件不存在 {INPUT_FILE}")
        return

    print(f"正在读取文件 (进程 ID: {os.getpid()})...")
    with open(INPUT_FILE, 'rb') as f: # binary mode for orjson compatibility
        data = json.loads(f.read())
    
    total_items = len(data)
    print(f"数据读取完毕，共 {total_items} 条原始数据。开始并行处理...")

    # 2. 准备并行
    # 获取CPU核心数，预留一个核给系统
    num_processes = max(1, cpu_count() - 1) 
    
    # 计算每个块的大小
    chunk_size = math.ceil(total_items / num_processes)
    # 将数据切片
    chunks = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
    
    print(f"启动 {num_processes} 个进程进行处理...")

    # 3. 并行执行
    processed_data = []
    with Pool(processes=num_processes) as pool:
        # map会自动按顺序收集结果
        result_chunks = pool.map(process_batch, chunks)
        
        # 4. 合并结果
        for chunk in result_chunks:
            processed_data.extend(chunk)

    end_processing_time = time.time()
    print(f"处理完成，耗时: {end_processing_time - start_time:.2f}秒")
    print(f"生成总条目数: {len(processed_data)}")

    # 5. 写入文件
    print(f"正在写入输出文件: {OUTPUT_FILE} ...")
    
    # 使用 binary write 模式兼容 orjson 和 utf-8
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"全部完成！总耗时: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    main()