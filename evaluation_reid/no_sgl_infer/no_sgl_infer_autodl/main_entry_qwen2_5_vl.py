
# --- 配置区域 ---
# CAPTION_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_caption_all_qwen.json"
# IMAGE_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs"
# MODEL_PATH = "/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
# only 微调了2k rstpreid 的 mllm_reid_txt2img_one2many_2k_new 数据集，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_2k_new'
# only 微调了2k rstpreid 的 mllm_reid_txt2img_one2many_2k_new_dpo_min 数据集，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_2k_new_dpo'
# 微调了2k rstpreid 的 mllm_reid_txt2img_one2many_2k_new_qlora_max_samples_1k 数据集中的1k条，没有dpo，但是有qlora，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_2k_new_qlora_max_samples_1k'
# 微调了16k rstpreid 的 mllm_reid_txt2img_one2many_16k_new_max_samples_1k 数据集中的1k条，没有dpo，也没有qlora，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_16k_new_max_samples_1k'
# 微调了16k rstpreid 的 mllm_reid_txt2img_one2many_16k_new_max_samples_1k 数据集中的1k条，没有dpo，也没有qlora，作为使用stage1的初始化模型，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_16k_new_max_samples_1k_stage2_init'
# 微调了16k rstpreid 的 mllm_reid_txt2img_one2many_16k_full 全量数据集，评估查看效果
# MODEL_PATH = '/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_16k_new_full'
# CACHE_FILE = "/home/wangrui/code/LLaMA-Factory/output/reid_resnet50_cache/pstp_test_gallery_50_shuffle.json"
# CACHE_FILE = "/home/wangrui/code/LLaMA-Factory/output/reid_resnet50_cache/pstp_test_gallery_10_shuffle.json"


import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from get_test_datasets import load_pstp_test
from baseline_filter_top50 import baseline_filter_50
from eval_text2image import run_iterative_eval

# --- autodl 配置区域 ---

# 1. ICFG-PEDES 配置
CAPTION_PATH_ICFG = "/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/ICFG-PEDES_match_score_54522.json"
# 注意：确保这里指向包含所有图片的根目录，而不是子文件夹
IMAGE_PATH_ICFG = "/root/autodl-tmp/MLLM4Text-ReID-main/data/ICFG-PEDES/ICFG-PEDES/imgs"

CACHE_FILE_ICFG_50 = "/root/autodl-tmp/output/reid_resnet50_cache/icfg_test_gallery_50_shuffle.json"
CACHE_FILE_ICFG_10 = "/root/autodl-tmp/output/reid_resnet50_cache/icfg_test_gallery_10_shuffle.json"

# 2. CUHK-PEDES 配置
CAPTION_PATH_CUHK = "/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/reid_raw_match_score_40201.json"
IMAGE_PATH_CUHK = "/root/autodl-tmp/MLLM4Text-ReID-main/data/CUHK-PEDES/CUHK-PEDES/imgs/"

CACHE_FILE_CUHK_50 = "/root/autodl-tmp/output/reid_resnet50_cache/cuhk_test_gallery_50_shuffle.json"
CACHE_FILE_CUHK_10 = "/root/autodl-tmp/output/reid_resnet50_cache/cuhk_test_gallery_10_shuffle.json"

# 模型路径
MODEL_PATH = '/root/autodl-tmp/output/home/wangrui/code/LLaMA-Factory/output/qwen2_5vl_lora_sft_retpreid_mllm_reid_txt2img_one2many_16k_new_full'

def generate_cache_for_dataset(dataset_name, caption_path, img_root, cache_path_50, cache_path_10):
    print(f"\n{'='*10} Processing {dataset_name} {'='*10}")
    
    if not os.path.exists(caption_path):
        print(f"Error: Caption file not found: {caption_path}")
        return

    print(f">>> Loading Data...")
    try:
        # 假设 load_pstp_test 返回的是 list of dicts
        test_data = load_pstp_test(caption_path)
        print(f"Loaded {len(test_data)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 生成 Top-50
    if not os.path.exists(cache_path_50):
        print(f">>> [Top-50] Cache missing. Generating...")
        baseline_filter_50(test_data, img_root, cache_path_50, gallery_size=50)
    else:
        print(f">>> [Top-50] Cache exists: {cache_path_50}")

    # 生成 Top-10
    if not os.path.exists(cache_path_10):
        print(f">>> [Top-10] Cache missing. Generating...")
        # 注意：这里会重新提特征。如果想要极致速度，可以修改 baseline_filter_50 让其支持传入 features
        # 但考虑到是一次性工作，重新跑一遍 ResNet 也很快 (因为有并发)
        baseline_filter_50(test_data, img_root, cache_path_10, gallery_size=10)
    else:
        print(f">>> [Top-10] Cache exists: {cache_path_10}")

def main():
    # # 1. 处理 ICFG
    # generate_cache_for_dataset(
    #     "ICFG-PEDES",
    #     CAPTION_PATH_ICFG,
    #     IMAGE_PATH_ICFG,
    #     CACHE_FILE_ICFG_50,
    #     CACHE_FILE_ICFG_10
    # )

    # # 2. 处理 CUHK
    # generate_cache_for_dataset(
    #     "CUHK-PEDES",
    #     CAPTION_PATH_CUHK,
    #     IMAGE_PATH_CUHK,
    #     CACHE_FILE_CUHK_50,
    #     CACHE_FILE_CUHK_10
    # )
    
    print("\n>>> All tasks finished.")
    
    # -----------------------------------------------------------
    # VLM Evaluation Part (Optional / Commented out for now)
    # -----------------------------------------------------------
    # 如果您准备好运行评估，请取消以下注释，并选择您要评估的数据集和缓存文件

    # icfg 
    # pass icfg 中一个测试图一个id 有17 - 19张图 
    # target_data = load_pstp_test(CAPTION_PATH_ICFG)
    # target_img_root = IMAGE_PATH_ICFG
    # target_cache = CACHE_FILE_ICFG_10


    # target_data = load_pstp_test(CAPTION_PATH_ICFG)
    # target_img_root = IMAGE_PATH_ICFG
    # target_cache = CACHE_FILE_ICFG_50

    # cuhk 
    # target_data = load_pstp_test(CAPTION_PATH_CUHK)
    # target_img_root = IMAGE_PATH_CUHK
    # target_cache = CACHE_FILE_CUHK_10

    target_data = load_pstp_test(CAPTION_PATH_CUHK)
    target_img_root = IMAGE_PATH_CUHK
    target_cache = CACHE_FILE_CUHK_50

    print(f"\n>>> Loading Model from {MODEL_PATH}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        print(">>> Starting Iterative Evaluation...")
        final_r1, final_map = run_iterative_eval(
            model, 
            processor, 
            target_data, 
            target_img_root, 
            target_cache, 
            num_iters=5
        )
        print("\n" + "="*50)
        print(f"FINAL RESULT (Max of 5 Rounds):")
        print(f"Rank-1: {final_r1:.4f}")
        print(f"mAP   : {final_map:.4f}")
        print("="*50)
        print("Check 'eval_process.log' for detailed logs.")
    except Exception as e:
        print(f"Model loading or evaluation failed: {e}")

if __name__ == "__main__":
    main()