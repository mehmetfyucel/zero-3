import os
import time
import math
import json
import threading
from functools import wraps

import torch
import torch.distributed as dist
import pynvml
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.integrations import HfDeepSpeedConfig
from datasets import load_dataset


# RunPod: kalici volume genelde /workspace altindadir. Hepsi env ile override edilebilir.
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DEEPSPEED_CFG = os.environ.get("DEEPSPEED_CFG", f"{WORKSPACE_DIR}/configs/config.json")
RESULTS_FILE = os.environ.get("RESULTS_FILE", f"{WORKSPACE_DIR}/results/results.md")
LOG_DIR = os.environ.get("LOG_DIR", f"{WORKSPACE_DIR}/logs")
COMM_LOG_FILE = os.environ.get("COMM_LOG_FILE", f"{LOG_DIR}/log.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{WORKSPACE_DIR}/results")

RUN_MODE = os.environ.get("RUN_MODE", "ZERO3")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B")
DATASET_SAMPLE_SIZE = int(os.environ.get("DATASET_SAMPLE_SIZE", "1000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
USE_ZERO_INIT = os.environ.get("USE_ZERO_INIT", "1") == "1"
ENABLE_COMM_PROFILER = os.environ.get("ENABLE_COMM_PROFILER", "0") == "1"
SYNC_COMM_PROFILER = os.environ.get("SYNC_COMM_PROFILER", "0") == "1"
ENABLE_GRADIENT_CHECKPOINTING = os.environ.get("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"

PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8"))
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))


class CommunicationProfiler:
    def __init__(self):
        self.events = []
        self.original_functions = {}
        self.enabled = False

    def _estimate_size_bytes(self, args, kwargs):
        def tensor_size(obj):
            if torch.is_tensor(obj):
                return obj.numel() * obj.element_size()
            if isinstance(obj, list):
                return sum(tensor_size(item) for item in obj)
            if isinstance(obj, tuple):
                return sum(tensor_size(item) for item in obj)
            if isinstance(obj, dict):
                return sum(tensor_size(item) for item in obj.values())
            return 0

        return tensor_size(args) + tensor_size(kwargs)

    def _patch_function(self, name):
        if not hasattr(dist, name):
            return

        if name in self.original_functions:
            return

        original = getattr(dist, name)
        self.original_functions[name] = original

        @wraps(original)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)

            rank = get_rank()
            world_size = get_world_size()
            start_time = time.time()
            status = "success"
            error = None

            try:
                if SYNC_COMM_PROFILER and torch.cuda.is_available():
                    torch.cuda.synchronize()

                result = original(*args, **kwargs)

                if SYNC_COMM_PROFILER and torch.cuda.is_available():
                    torch.cuda.synchronize()

                return result

            except Exception as exc:
                status = "error"
                error = str(exc)
                raise

            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                estimated_payload_bytes = self._estimate_size_bytes(args, kwargs)

                self.events.append({
                    "rank": rank,
                    "world_size": world_size,
                    "op": name,
                    "start_time": round(start_time, 6),
                    "end_time": round(end_time, 6),
                    "duration_ms": round(duration_ms, 3),
                    "estimated_payload_mb": round(estimated_payload_bytes / (1024 ** 2), 4),
                    "status": status,
                    "error": error
                })

        setattr(dist, name, wrapped)

    def start(self):
        self.enabled = True

        for name in [
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "all_gather_object",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "broadcast",
            "barrier"
        ]:
            self._patch_function(name)

    def stop(self):
        self.enabled = False

        for name, original in self.original_functions.items():
            setattr(dist, name, original)

    def local_summary(self, train_time_seconds):
        op_stats = {}

        for event in self.events:
            op = event["op"]

            if op not in op_stats:
                op_stats[op] = {
                    "count": 0,
                    "total_duration_ms": 0.0,
                    "max_duration_ms": 0.0,
                    "min_duration_ms": None,
                    "total_payload_mb": 0.0
                }

            op_stats[op]["count"] += 1
            op_stats[op]["total_duration_ms"] += event["duration_ms"]
            op_stats[op]["max_duration_ms"] = max(op_stats[op]["max_duration_ms"], event["duration_ms"])

            if op_stats[op]["min_duration_ms"] is None:
                op_stats[op]["min_duration_ms"] = event["duration_ms"]
            else:
                op_stats[op]["min_duration_ms"] = min(op_stats[op]["min_duration_ms"], event["duration_ms"])

            op_stats[op]["total_payload_mb"] += event["estimated_payload_mb"]

        for op, stat in op_stats.items():
            stat["total_duration_ms"] = round(stat["total_duration_ms"], 3)
            stat["max_duration_ms"] = round(stat["max_duration_ms"], 3)
            stat["min_duration_ms"] = round(stat["min_duration_ms"], 3) if stat["min_duration_ms"] is not None else 0.0
            stat["avg_duration_ms"] = round(stat["total_duration_ms"] / stat["count"], 3) if stat["count"] > 0 else 0.0
            stat["total_payload_mb"] = round(stat["total_payload_mb"], 4)
            stat["local_bottleneck_percent_of_train_time"] = round(
                (stat["total_duration_ms"] / 1000) / train_time_seconds * 100,
                2
            ) if train_time_seconds > 0 else 0.0

        slowest_events = sorted(
            self.events,
            key=lambda item: item["duration_ms"],
            reverse=True
        )[:10]

        return {
            "rank": get_rank(),
            "event_count": len(self.events),
            "op_stats": op_stats,
            "slowest_events": slowest_events
        }


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


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def build_training_args():
    base = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 10,
        "fp16": True,
        "save_strategy": "no",
        "eval_strategy": "no",
        "deepspeed": DEEPSPEED_CFG,
        "report_to": "none"
    }

    return TrainingArguments(**base)


def load_deepspeed_config_for_zero_init():
    with open(DEEPSPEED_CFG, "r", encoding="utf-8") as f:
        config = json.load(f)

    world_size = get_world_size()
    train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size

    config["train_micro_batch_size_per_gpu"] = PER_DEVICE_TRAIN_BATCH_SIZE
    config["gradient_accumulation_steps"] = GRADIENT_ACCUMULATION_STEPS
    config["train_batch_size"] = train_batch_size

    if "fp16" in config and isinstance(config["fp16"], dict):
        if config["fp16"].get("enabled") == "auto":
            config["fp16"]["enabled"] = True

    if "bf16" in config and isinstance(config["bf16"], dict):
        if config["bf16"].get("enabled") == "auto":
            config["bf16"]["enabled"] = False

    if "optimizer" in config and isinstance(config["optimizer"], dict):
        params = config["optimizer"].get("params", {})
        if isinstance(params, dict):
            if params.get("lr") == "auto":
                params["lr"] = LEARNING_RATE
            if params.get("betas") == "auto":
                params["betas"] = [0.9, 0.999]
            if params.get("eps") == "auto":
                params["eps"] = 1e-8
            if params.get("weight_decay") == "auto":
                params["weight_decay"] = 0.0

    if "scheduler" in config and isinstance(config["scheduler"], dict):
        params = config["scheduler"].get("params", {})
        if isinstance(params, dict):
            if params.get("warmup_min_lr") == "auto":
                params["warmup_min_lr"] = 0
            if params.get("warmup_max_lr") == "auto":
                params["warmup_max_lr"] = LEARNING_RATE
            if params.get("warmup_num_steps") == "auto":
                params["warmup_num_steps"] = 0

    return config


def inspect_zero3_partitioning(model):
    rank = get_rank()
    total_params = 0
    local_params = 0
    zero3_param_count = 0
    sampled = 0

    for name, param in model.named_parameters():
        ds_numel = getattr(param, "ds_numel", None)
        ds_id = getattr(param, "ds_id", None)
        ds_status = getattr(param, "ds_status", None)

        if ds_numel is not None:
            total_params += int(ds_numel)
            zero3_param_count += 1
        else:
            total_params += int(param.numel())

        local_params += int(param.numel())

        if sampled < 8:
            print(
                f"[ZERO DEBUG] rank={rank} "
                f"name={name} "
                f"local_numel={param.numel()} "
                f"ds_numel={ds_numel} "
                f"ds_id={ds_id} "
                f"ds_status={ds_status}"
            )
            sampled += 1

    print(
        f"[ZERO DEBUG] rank={rank} "
        f"dist_initialized={dist.is_initialized()} "
        f"world_size={get_world_size()} "
        f"use_zero_init={USE_ZERO_INIT} "
        f"zero3_param_count={zero3_param_count} "
        f"total_param_numel_reported={total_params} "
        f"local_param_numel_current={local_params}"
    )


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


def get_local_peak_vram_gb():
    """Bu process'in kullandigi (kendi) GPU'nun peak tahsisi."""
    if not torch.cuda.is_available():
        return 0.0

    device = torch.cuda.current_device()
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2)


def format_peak_vram(all_peak_vram):
    parts = []
    values = []

    for rank_id, val in enumerate(all_peak_vram):
        if val is None:
            continue
        parts.append(f"rank{rank_id}: {val}GB")
        values.append(val)

    global_max = round(max(values), 2) if values else 0.0

    return f"{'; '.join(parts)} | global max: {global_max}GB"


def calculate_avg_tokens_per_sample(tokenized_datasets):
    total_tokens = sum(sum(mask) for mask in tokenized_datasets["attention_mask"])
    total_samples = len(tokenized_datasets)

    if total_samples == 0:
        return 0.0

    return total_tokens / total_samples


def load_training_dataset():
    if is_dist_ready():
        if is_main_process():
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        dist.barrier()

    full_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    full_dataset_size = len(full_dataset)

    if DATASET_SAMPLE_SIZE > 0:
        selected_size = min(DATASET_SAMPLE_SIZE, full_dataset_size)
        dataset = full_dataset.select(range(selected_size))
    else:
        selected_size = full_dataset_size
        dataset = full_dataset

    selected_dataset_percentage = (
        selected_size / full_dataset_size * 100
        if full_dataset_size > 0
        else 0.0
    )

    if is_main_process():
        print(f"[INFO] Full dataset size: {full_dataset_size}")
        print(f"[INFO] Selected dataset size: {selected_size}")
        print(f"[INFO] Selected dataset percentage: {selected_dataset_percentage:.2f}%")

    return dataset, full_dataset_size, selected_size, selected_dataset_percentage


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        labels = []

        for ids, mask in zip(tokens["input_ids"], tokens["attention_mask"]):
            labels.append([token if m == 1 else -100 for token, m in zip(ids, mask)])

        tokens["labels"] = labels

        return tokens

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    return tokenized_datasets


def load_model_with_optional_zero_init():
    print(f"[DEBUG] dist initialized: {dist.is_initialized()}, rank: {get_rank()}, world_size: {get_world_size()}")

    if USE_ZERO_INIT:
        zero_init_config = load_deepspeed_config_for_zero_init()

        with deepspeed.zero.Init(config_dict_or_path=zero_init_config):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    inspect_zero3_partitioning(model)

    if ENABLE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model


def gather_python_objects(local_object):
    world_size = get_world_size()
    gathered = [None for _ in range(world_size)]

    if is_dist_ready():
        dist.all_gather_object(gathered, local_object)
    else:
        gathered = [local_object]

    return gathered


def aggregate_comm_summaries(all_comm_summaries, train_time_seconds):
    op_rank_values = {}

    for rank_summary in all_comm_summaries:
        if not rank_summary:
            continue

        rank = rank_summary.get("rank", None)

        for op, stat in rank_summary.get("op_stats", {}).items():
            if op not in op_rank_values:
                op_rank_values[op] = []

            op_rank_values[op].append({
                "rank": rank,
                "count": stat["count"],
                "total_duration_ms": stat["total_duration_ms"],
                "avg_duration_ms": stat["avg_duration_ms"],
                "max_duration_ms": stat["max_duration_ms"],
                "min_duration_ms": stat["min_duration_ms"],
                "total_payload_mb": stat["total_payload_mb"],
                "local_bottleneck_percent_of_train_time": stat["local_bottleneck_percent_of_train_time"]
            })

    aggregate = {}

    for op, values in op_rank_values.items():
        counts = [item["count"] for item in values]
        total_durations = [item["total_duration_ms"] for item in values]
        avg_durations = [item["avg_duration_ms"] for item in values]
        max_durations = [item["max_duration_ms"] for item in values]
        payloads = [item["total_payload_mb"] for item in values]

        rank_mean_total_duration_ms = sum(total_durations) / len(total_durations) if total_durations else 0.0
        rank_max_total_duration_ms = max(total_durations) if total_durations else 0.0
        rank_min_total_duration_ms = min(total_durations) if total_durations else 0.0

        aggregate[op] = {
            "rank_count": len(values),
            "count_per_rank": values,
            "mean_count_per_rank": round(sum(counts) / len(counts), 3) if counts else 0.0,
            "max_count_per_rank": max(counts) if counts else 0,
            "min_count_per_rank": min(counts) if counts else 0,
            "rank_mean_total_duration_ms": round(rank_mean_total_duration_ms, 3),
            "rank_max_total_duration_ms": round(rank_max_total_duration_ms, 3),
            "rank_min_total_duration_ms": round(rank_min_total_duration_ms, 3),
            "rank_mean_avg_duration_ms": round(sum(avg_durations) / len(avg_durations), 3) if avg_durations else 0.0,
            "rank_max_single_call_duration_ms": round(max(max_durations), 3) if max_durations else 0.0,
            "rank_mean_total_payload_mb": round(sum(payloads) / len(payloads), 4) if payloads else 0.0,
            "rank_max_total_payload_mb": round(max(payloads), 4) if payloads else 0.0,
            "bottleneck_percent_of_train_time_by_slowest_rank": round(
                (rank_max_total_duration_ms / 1000) / train_time_seconds * 100,
                2
            ) if train_time_seconds > 0 else 0.0
        }

    return aggregate


def write_comm_log(all_comm_summaries, train_time_seconds):
    if not is_main_process():
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    aggregate = aggregate_comm_summaries(all_comm_summaries, train_time_seconds)

    payload = {
        "scope": "python_level_torch_distributed_only",
        "limitation": "DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured by this profiler.",
        "profiler_overhead_note": "If SYNC_COMM_PROFILER=1, CUDA synchronization can slow training and make throughput numbers less representative.",
        "sync_comm_profiler": SYNC_COMM_PROFILER,
        "model": MODEL_ID,
        "mode": RUN_MODE,
        "world_size": get_world_size(),
        "dataset_sample_size": DATASET_SAMPLE_SIZE,
        "max_length": MAX_LENGTH,
        "train_time_seconds": round(train_time_seconds, 3),
        "aggregate": aggregate,
        "per_rank": all_comm_summaries
    }

    with open(COMM_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Communication log written to {COMM_LOG_FILE}")


def format_comm_summary_for_results(all_comm_summaries, train_time_seconds):
    aggregate = aggregate_comm_summaries(all_comm_summaries, train_time_seconds)

    if not aggregate:
        return "No Python-level torch.distributed collective calls captured"

    parts = []

    for op, stat in aggregate.items():
        parts.append(
            f"{op}: mean_count_per_rank={stat['mean_count_per_rank']}, "
            f"rank_mean_total_ms={stat['rank_mean_total_duration_ms']}, "
            f"rank_max_total_ms={stat['rank_max_total_duration_ms']}, "
            f"rank_max_single_call_ms={stat['rank_max_single_call_duration_ms']}, "
            f"bottleneck_by_slowest_rank={stat['bottleneck_percent_of_train_time_by_slowest_rank']}%"
        )

    return " | ".join(parts)


def write_results_to_md(metrics):
    if not is_main_process():
        return

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n# Distributed Training Benchmark Result\n\n")
        f.write(f"Mode: {metrics['mode']}\n")
        f.write(f"Model: {metrics['model']}\n")
        f.write(f"Dataset Sample Size: {metrics['dataset_sample_size']}\n")
        f.write(f"Full Dataset Size: {metrics['full_dataset_size']}\n")
        f.write(f"Selected Dataset Size: {metrics['selected_dataset_size']}\n")
        f.write(f"Selected Dataset Percentage: {metrics['selected_dataset_percentage']}%\n")
        f.write(f"Epoch Processed Dataset Percentage: {metrics['epoch_processed_dataset_percentage']}%\n")
        f.write(f"Max Sequence Length: {metrics['max_length']}\n")
        f.write(f"World Size: {metrics['world_size']}\n")
        f.write(f"Use ZeRO Init: {metrics['use_zero_init']}\n")
        f.write(f"Gradient Checkpointing Enabled: {metrics['gradient_checkpointing_enabled']}\n")
        f.write(f"Per Device Train Batch Size: {metrics['per_device_train_batch_size']}\n")
        f.write(f"Gradient Accumulation Steps: {metrics['gradient_accumulation_steps']}\n")
        f.write(f"Communication Profiler Enabled: {metrics['communication_profiler_enabled']}\n")
        f.write(f"Communication Profiler CUDA Sync Enabled: {metrics['communication_profiler_cuda_sync_enabled']}\n")
        f.write(f"Training Time Seconds: {metrics['time']}\n")
        f.write(f"Trainer Reported Samples Per Second: {metrics['trainer_samples_per_second']}\n")
        f.write(f"Average Non Padding Tokens Per Sample: {metrics['avg_tokens_per_sample']}\n")
        f.write(f"Trainer Based Tokens Per Second: {metrics['trainer_based_tokens_per_second']}\n")
        f.write(f"Manual Global Tokens Per Second: {metrics['manual_global_tokens_per_second']}\n")
        f.write(f"Manual Global Total Samples: {metrics['manual_global_total_samples']}\n")
        f.write(f"Manual Global Total Tokens: {metrics['manual_global_total_tokens']}\n")
        f.write(f"Train Loss: {metrics['train_loss']}\n")
        f.write(f"Perplexity: {metrics['perplexity']}\n")
        f.write(f"Peak PyTorch VRAM: {metrics['peak_vram']}\n")
        f.write(f"Average GPU Utilization: {metrics['avg_gpu_util']}\n")
        f.write(f"Maximum GPU Utilization: {metrics['max_gpu_util']}\n")
        f.write(f"Average GPU Memory: {metrics['avg_gpu_memory']}\n")
        f.write(f"Maximum GPU Memory: {metrics['max_gpu_memory']}\n")
        f.write(f"Communication Summary: {metrics['communication_summary']}\n")
        f.write(f"Communication Scope: {metrics['communication_scope']}\n")
        f.write(f"Communication Log File: {metrics['communication_log_file']}\n")
        f.write("\n---\n")

    print(f"\n[INFO] Sonuçlar başarıyla {RESULTS_FILE} dosyasına kaydedildi.")


def train():
    print(f"[INFO] Eğitim başlatılıyor... Model: {MODEL_ID}, Mod: {RUN_MODE}")

    ds_config_ref = HfDeepSpeedConfig(DEEPSPEED_CFG)

    training_args = build_training_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_optional_zero_init()

    dataset, full_dataset_size, selected_dataset_size, selected_dataset_percentage = load_training_dataset()
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)

    avg_tokens_per_sample = calculate_avg_tokens_per_sample(tokenized_datasets)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = GPUMonitor(interval=1.0)
    comm_profiler = CommunicationProfiler()

    monitor.start()

    if ENABLE_COMM_PROFILER:
        comm_profiler.start()

    start_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_time

    if ENABLE_COMM_PROFILER:
        comm_profiler.stop()

    monitor.stop()

    local_gpu_summary = monitor.summary()
    all_gpu_summaries = gather_python_objects(local_gpu_summary)

    # Peak VRAM'i her rank kendi GPU'su icin olcup topla (0.0GB sorununu cozer)
    local_peak_vram = get_local_peak_vram_gb()
    all_peak_vram = gather_python_objects(local_peak_vram)

    if ENABLE_COMM_PROFILER:
        local_comm_summary = comm_profiler.local_summary(total_train_time)
        all_comm_summaries = gather_python_objects(local_comm_summary)
        write_comm_log(all_comm_summaries, total_train_time)
        communication_summary = format_comm_summary_for_results(all_comm_summaries, total_train_time)
        communication_log_file = COMM_LOG_FILE
    else:
        communication_summary = "Disabled. Set ENABLE_COMM_PROFILER=1 to capture Python-level torch.distributed calls."
        communication_log_file = "Not generated"

    train_metrics = train_result.metrics

    train_loss = train_metrics.get("train_loss", 0.0)
    trainer_samples_per_second = train_metrics.get("train_samples_per_second", 0.0)

    num_epochs = float(training_args.num_train_epochs)
    manual_global_total_samples = len(tokenized_datasets) * num_epochs
    manual_global_total_tokens = manual_global_total_samples * avg_tokens_per_sample

    manual_global_tokens_per_second = (
        manual_global_total_tokens / total_train_time
        if total_train_time > 0
        else 0.0
    )

    trainer_based_tokens_per_second = trainer_samples_per_second * avg_tokens_per_sample

    epoch_processed_dataset_percentage = (
        selected_dataset_size * num_epochs / full_dataset_size * 100
        if full_dataset_size > 0
        else 0.0
    )

    perplexity = math.exp(train_loss) if train_loss > 0 else 0.0

    gpu_metrics = format_gpu_summaries(all_gpu_summaries)

    metrics = {
        "mode": RUN_MODE,
        "model": MODEL_ID,
        "dataset_sample_size": DATASET_SAMPLE_SIZE,
        "full_dataset_size": full_dataset_size,
        "selected_dataset_size": selected_dataset_size,
        "selected_dataset_percentage": round(selected_dataset_percentage, 2),
        "epoch_processed_dataset_percentage": round(epoch_processed_dataset_percentage, 2),
        "max_length": MAX_LENGTH,
        "world_size": get_world_size(),
        "use_zero_init": USE_ZERO_INIT,
        "gradient_checkpointing_enabled": ENABLE_GRADIENT_CHECKPOINTING,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "communication_profiler_enabled": ENABLE_COMM_PROFILER,
        "communication_profiler_cuda_sync_enabled": SYNC_COMM_PROFILER,
        "time": round(total_train_time, 2),
        "trainer_samples_per_second": round(trainer_samples_per_second, 2),
        "avg_tokens_per_sample": round(avg_tokens_per_sample, 2),
        "trainer_based_tokens_per_second": int(trainer_based_tokens_per_second),
        "manual_global_tokens_per_second": int(manual_global_tokens_per_second),
        "manual_global_total_samples": int(manual_global_total_samples),
        "manual_global_total_tokens": int(manual_global_total_tokens),
        "train_loss": round(train_loss, 4),
        "perplexity": round(perplexity, 4),
        "peak_vram": format_peak_vram(all_peak_vram),
        "avg_gpu_util": gpu_metrics["avg_gpu_util"],
        "max_gpu_util": gpu_metrics["max_gpu_util"],
        "avg_gpu_memory": gpu_metrics["avg_gpu_memory"],
        "max_gpu_memory": gpu_metrics["max_gpu_memory"],
        "communication_summary": communication_summary,
        "communication_scope": "Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.",
        "communication_log_file": communication_log_file
    }

    write_results_to_md(metrics)

    _ = ds_config_ref


if __name__ == "__main__":
    train()