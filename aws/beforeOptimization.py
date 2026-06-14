import os
import time
import math
import threading
import torch
import torch.distributed as dist
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

WORKSPACE_DIR = "/home/ubuntu/workspace"
DEEPSPEED_CFG = f"{WORKSPACE_DIR}/configs/config.json"
RESULTS_FILE = f"{WORKSPACE_DIR}/results/results.md"
OUTPUT_DIR = f"{WORKSPACE_DIR}/results"

RUN_MODE = os.environ.get("RUN_MODE", "ZERO3")
MODEL_ID = "Qwen/Qwen2.5-0.5B"


class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None

    def _loop(self):
        nvml_started = False

        try:
            pynvml.nvmlInit()
            nvml_started = True

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_count = pynvml.nvmlDeviceGetCount()

            if local_rank >= device_count:
                local_rank = 0

            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

            while self.running:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                self.samples.append({
                    "gpu_util": float(util.gpu),
                    "mem_used_gb": float(mem.used / (1024 ** 3))
                })

                time.sleep(self.interval)

        except Exception:
            self.samples.append({
                "gpu_util": 0.0,
                "mem_used_gb": 0.0
            })

        finally:
            if nvml_started:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join()

    def summary(self):
        if not self.samples:
            return {
                "avg_gpu_util": 0.0,
                "max_gpu_util": 0.0,
                "avg_mem_gb": 0.0,
                "max_mem_gb": 0.0,
                "samples": 0
            }

        gpu_values = [sample["gpu_util"] for sample in self.samples]
        mem_values = [sample["mem_used_gb"] for sample in self.samples]

        return {
            "avg_gpu_util": round(sum(gpu_values) / len(gpu_values), 2),
            "max_gpu_util": round(max(gpu_values), 2),
            "avg_mem_gb": round(sum(mem_values) / len(mem_values), 2),
            "max_mem_gb": round(max(mem_values), 2),
            "samples": len(self.samples)
        }


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    return get_rank() == 0


def write_results_to_md(metrics):
    if not is_main_process():
        return

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("# Distributed Training Benchmark Results\n\n")
            f.write("| Mode | Model | Time (s) | Throughput (tok/s) | Perplexity | Peak VRAM (GB) | Avg GPU Util (%) | Max GPU Util (%) | Avg GPU Memory (GB) | Max GPU Memory (GB) |\n")
            f.write("|------|-------|----------|--------------------|------------|----------------|------------------|------------------|---------------------|---------------------|\n")

        row = (
            f"| {metrics['mode']} "
            f"| {metrics['model']} "
            f"| {metrics['time']} "
            f"| {metrics['throughput']} "
            f"| {metrics['perplexity']} "
            f"| {metrics['peak_vram']} "
            f"| {metrics['avg_gpu_util']} "
            f"| {metrics['max_gpu_util']} "
            f"| {metrics['avg_gpu_memory']} "
            f"| {metrics['max_gpu_memory']} |\n"
        )

        f.write(row)

    print(f"\n[INFO] Sonuçlar başarıyla {RESULTS_FILE} dosyasına kaydedildi.")


def build_training_args():
    base = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_strategy": "no",
        "eval_strategy": "no",
        "fp16": True,
        "deepspeed": DEEPSPEED_CFG,
        "report_to": "none"
    }

    return TrainingArguments(**base)


def format_gpu_summaries(all_gpu_summaries):
    avg_util_parts = []
    max_util_parts = []
    avg_mem_parts = []
    max_mem_parts = []

    avg_util_values = []
    max_util_values = []
    avg_mem_values = []
    max_mem_values = []

    for rank_id, item in enumerate(all_gpu_summaries):
        if item is None:
            continue

        avg_util_parts.append(f"rank{rank_id}: {item['avg_gpu_util']}%")
        max_util_parts.append(f"rank{rank_id}: {item['max_gpu_util']}%")
        avg_mem_parts.append(f"rank{rank_id}: {item['avg_mem_gb']}GB")
        max_mem_parts.append(f"rank{rank_id}: {item['max_mem_gb']}GB")

        avg_util_values.append(item["avg_gpu_util"])
        max_util_values.append(item["max_gpu_util"])
        avg_mem_values.append(item["avg_mem_gb"])
        max_mem_values.append(item["max_mem_gb"])

    global_avg_util = round(sum(avg_util_values) / len(avg_util_values), 2) if avg_util_values else 0.0
    global_max_util = round(max(max_util_values), 2) if max_util_values else 0.0
    global_avg_mem = round(sum(avg_mem_values) / len(avg_mem_values), 2) if avg_mem_values else 0.0
    global_max_mem = round(max(max_mem_values), 2) if max_mem_values else 0.0

    return {
        "avg_gpu_util": f"{'; '.join(avg_util_parts)} | global avg: {global_avg_util}%",
        "max_gpu_util": f"{'; '.join(max_util_parts)} | global max: {global_max_util}%",
        "avg_gpu_memory": f"{'; '.join(avg_mem_parts)} | global avg: {global_avg_mem}GB",
        "max_gpu_memory": f"{'; '.join(max_mem_parts)} | global max: {global_max_mem}GB"
    }


def get_peak_vram():
    if not torch.cuda.is_available():
        return "0.0GB"

    values = []

    for i in range(torch.cuda.device_count()):
        value = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        values.append(f"{round(value, 2)}GB")

    return ", ".join(values)


def train():
    print(f"[INFO] Eğitim başlatılıyor... Model: {MODEL_ID}, Mod: {RUN_MODE}")

    training_args = build_training_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(500))

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

        labels = []

        for ids, mask in zip(tokens["input_ids"], tokens["attention_mask"]):
            labels.append([token if m == 1 else -100 for token, m in zip(ids, mask)])

        tokens["labels"] = labels

        return tokens

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = GPUMonitor(interval=1.0)
    monitor.start()

    start_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_time

    monitor.stop()
    local_gpu_summary = monitor.summary()

    world_size = get_world_size()
    all_gpu_summaries = [None for _ in range(world_size)]

    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(all_gpu_summaries, local_gpu_summary)
    else:
        all_gpu_summaries = [local_gpu_summary]

    train_loss = train_result.metrics.get("train_loss", 0.0)
    perplexity = math.exp(train_loss) if train_loss > 0 else 0.0

    total_tokens = sum(sum(mask) for mask in tokenized_datasets["attention_mask"])
    throughput = total_tokens / total_train_time if total_train_time > 0 else 0

    gpu_metrics = format_gpu_summaries(all_gpu_summaries)

    metrics = {
        "mode": RUN_MODE,
        "model": MODEL_ID,
        "time": round(total_train_time, 2),
        "throughput": int(throughput),
        "perplexity": round(perplexity, 4),
        "peak_vram": get_peak_vram(),
        "avg_gpu_util": gpu_metrics["avg_gpu_util"],
        "max_gpu_util": gpu_metrics["max_gpu_util"],
        "avg_gpu_memory": gpu_metrics["avg_gpu_memory"],
        "max_gpu_memory": gpu_metrics["max_gpu_memory"]
    }

    write_results_to_md(metrics)


if __name__ == "__main__":
    train()