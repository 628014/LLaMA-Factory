# tony_numeric_sorting_qwen25vl7b.py
import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# =========================
# Config
# =========================
MODEL_PATH = "/home/wangrui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SEED = 42
N_SAMPLES = 200
SEQ_LEN = 10
LOW, HIGH = 1.0, 100.0
DECIMALS = 2

MAX_NEW_TOKENS = 128


# =========================
# Data
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gen_one_sequence(seq_len: int = 10) -> List[float]:
    # Random floats in [LOW, HIGH], rounded to 2 decimals.
    # Ensure uniqueness to avoid ambiguity in "set" based metrics.
    # If you do want duplicates, we can adapt the metric to multiset.
    nums = set()
    while len(nums) < seq_len:
        x = round(random.uniform(LOW, HIGH), DECIMALS)
        nums.add(x)
    return list(nums)


def build_dataset(n: int = 200) -> List[List[float]]:
    data = []
    for _ in range(n):
        seq = gen_one_sequence(SEQ_LEN)
        random.shuffle(seq)  # unsorted input
        data.append(seq)
    return data


# =========================
# Prompt & Parsing
# =========================
def build_prompt(nums: List[float]) -> str:
    # Strong constraints to reduce formatting noise
    nums_str = ", ".join([f"{x:.2f}" for x in nums])
    return (
        "You are given a list of 10 floating-point numbers with two decimals.\n"
        "Task: sort them in strictly increasing order.\n"
        "Output format rules:\n"
        "1) Output ONLY the sorted numbers.\n"
        "2) Keep exactly two decimals for each number.\n"
        "3) Separate numbers with a single comma and a space.\n\n"
        f"Numbers: {nums_str}\n"
        "Sorted:"
    )


_num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

def parse_numbers(text: str) -> List[float]:
    # Extract floats; then round to the configured decimals to align with gt format
    raw = _num_re.findall(text)
    out = []
    for t in raw:
        try:
            out.append(round(float(t), DECIMALS))
        except:
            pass
    return out


# =========================
# Metrics (as in the figure)
# =========================
def numerical_sorting_metrics(gt: List[float], pred: List[float]) -> Tuple[float, float, float]:
    """
    Mirrors Algorithm 1 in the figure:

    gt_set <- set(gt)
    pred_set <- set(pred)
    indices <- [pred.index(x) for x in gt if x in pred]
    if indices is sorted non-decreasing:
        accuracy <- |indices| / |gt|
    else:
        accuracy <- 0

    recall <- |{x in gt | x in pred_set}| / |gt|
    hallucination <- |{x in pred | x not in gt_set}| / |pred|
    """
    if len(gt) == 0:
        return 0.0, 0.0, 0.0

    gt_set = set(gt)
    pred_set = set(pred)

    indices = [pred.index(x) for x in gt if x in pred]
    if indices == sorted(indices):
        accuracy = len(indices) / len(gt)
    else:
        accuracy = 0.0

    recall = len([x for x in gt if x in pred_set]) / len(gt)

    if len(pred) == 0:
        hallucination = 0.0
    else:
        hallucination = len([x for x in pred if x not in gt_set]) / len(pred)

    return accuracy, recall, hallucination


# =========================
# Model
# =========================
def load_model(model_path: str = MODEL_PATH):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()
    return model, processor


@torch.no_grad()
def infer_sorted_numbers(
    model,
    processor,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # deterministic
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    # trim prompt tokens
    out_ids = gen_ids[0, inputs["input_ids"].shape[1]:]
    out = processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out.strip()


# =========================
# Run Experiment
# =========================
@dataclass
class SampleResult:
    idx: int
    input_nums: List[float]
    gt_sorted: List[float]
    raw_output: str
    pred_nums: List[float]
    accuracy: float
    recall: float
    hallucination: float


def run_experiment():
    set_seed(SEED)
    data = build_dataset(N_SAMPLES)

    model, processor = load_model(MODEL_PATH)

    results: List[SampleResult] = []

    for i, seq in enumerate(data):
        gt_sorted = sorted([round(x, DECIMALS) for x in seq])
        prompt = build_prompt(seq)

        raw = infer_sorted_numbers(model, processor, prompt)
        pred = parse_numbers(raw)

        acc, rec, hall = numerical_sorting_metrics(gt_sorted, pred)

        results.append(SampleResult(
            idx=i,
            input_nums=seq,
            gt_sorted=gt_sorted,
            raw_output=raw,
            pred_nums=pred,
            accuracy=acc,
            recall=rec,
            hallucination=hall
        ))

        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{N_SAMPLES}] acc={acc:.3f} rec={rec:.3f} hall={hall:.3f}")

    # Aggregate
    avg_acc = sum(r.accuracy for r in results) / len(results)
    avg_rec = sum(r.recall for r in results) / len(results)
    avg_hall = sum(r.hallucination for r in results) / len(results)

    strict_sorted_ok = sum(1 for r in results if r.accuracy == 1.0) / len(results)

    print("\n==== Overall Results ====")
    print(f"Mean Accuracy:      {avg_acc:.4f}")
    print(f"Mean Recall:        {avg_rec:.4f}")
    print(f"Mean Hallucination: {avg_hall:.4f}")
    print(f"Strict Perfect Sort Rate (Accuracy==1): {strict_sorted_ok:.4f}")

    # Save a few failure cases for analysis
    bad = [r for r in results if r.accuracy < 1.0 or r.hallucination > 0.0]
    print(f"\nFailure/Noisy cases: {len(bad)} / {len(results)}")
    for r in bad[:5]:
        print("\n--- Case", r.idx, "---")
        print("Input: ", ", ".join(f"{x:.2f}" for x in r.input_nums))
        print("GT:    ", ", ".join(f"{x:.2f}" for x in r.gt_sorted))
        print("Out:   ", r.raw_output)
        print("Pred:  ", r.pred_nums)
        print(f"acc={r.accuracy:.3f}, rec={r.recall:.3f}, hall={r.hallucination:.3f}")


if __name__ == "__main__":
    run_experiment()
