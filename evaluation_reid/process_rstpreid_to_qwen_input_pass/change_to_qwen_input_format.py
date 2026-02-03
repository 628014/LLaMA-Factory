# ...existing code...
import json
import os

output_path = [
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2many_50k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2one_50k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_many2one_20k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_20k_new.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2one_20k_new.json'
]

input_paths = [
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2many_50k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_img2img_one2one_50k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_many2one_20k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2many_20k.json',
    '/home/wangrui/code/LLaMA-Factory/data/mllm_reid_txt2img_one2one_20k.json'
]


def convert_five_to_qwen(input_paths, output_paths):
    """
    将每个 input_paths[i] 转换为 qwen 格式并保存到 output_paths[i]。
    支持输入文件为单个 dict 或包含多个条目的 list。
    """
    if len(input_paths) != len(output_paths):
        raise ValueError("input_paths and output_paths must have the same length")

    for idx, (inp, outp) in enumerate(zip(input_paths, output_paths)):
        print(f"Processing file {idx}: {inp} -> {outp}")
        with open(inp, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else [data]
        out_entries = []

        for entry in entries:
            images = entry.get('images', [])
            conv = entry.get('conversations', []) or []
            # safe extraction: conversations elements may not be dicts
            user_content = ''
            assistant_content = ''
            if len(conv) > 0 and isinstance(conv[0], dict):
                user_content = conv[0].get('value', '')
            if len(conv) > 1 and isinstance(conv[1], dict):
                assistant_content = conv[1].get('value', '')

            out_entries.append({
                "messages": [
                    {"content": user_content, "role": "user"},
                    {"content": assistant_content, "role": "assistant"}
                ],
                "images": images
            })

        os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(out_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_five_to_qwen(input_paths, output_path)