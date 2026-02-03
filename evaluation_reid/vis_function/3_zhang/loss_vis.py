import json
import pandas as pd

import matplotlib.pyplot as plt

# 读取 Full Fine-tuning 的 Loss
with open("/home/wangrui/code/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft_retpreid_with_score/lora64_full/trainer_state.json", "r") as f:
    full_data = json.load(f)
full_loss = pd.DataFrame(full_data["log_history"])[["step", "loss"]].dropna()

# 读取 LoRA Fine-tuning 的 Loss
with open("/home/wangrui/code/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft_retpreid_with_score_and_ccls/lora64_full/trainer_state.json", "r") as f:
    lora_data = json.load(f)
lora_loss = pd.DataFrame(lora_data["log_history"])[["step", "loss"]].dropna()




# 设置画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制 (a) Full Fine-tuning from Scratch
ax1.plot(full_loss["step"], full_loss["loss"], color="crimson", label="Training Loss")
ax1.set_title("(a) Full Fine-tuning from Scratch")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0, 10) # 和示例图y轴范围对齐

# 绘制 (b) LoRA Fine-tuning
ax2.plot(lora_loss["step"], lora_loss["loss"], color="forestgreen", label="Training Loss")
ax2.set_title("(b) LoRA Fine-tuning")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 10)

# 整体标题
fig.suptitle("Loss Comparison: Full Fine-tuning vs LoRA Fine-tuning", fontsize=14)
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=300)
plt.show()