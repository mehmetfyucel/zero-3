
# Distributed Training Benchmark Result

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 500
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Training Time Seconds: 117.77
Trainer Reported Samples Per Second: 5.57
Average Non Padding Tokens Per Sample: 55.25
Trainer Based Tokens Per Second: 307
Manual Global Tokens Per Second: 234
Manual Global Total Samples: 500
Manual Global Total Tokens: 27625
Train Loss: 3.0542
Perplexity: 21.2045
Peak PyTorch VRAM: 11.47GB
Average GPU Utilization: rank0: 71.94%; rank1: 72.53% | global avg: 72.23%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 11.47GB; rank1: 11.47GB | global avg: 11.47GB
Maximum GPU Memory: rank0: 15.1GB; rank1: 15.1GB | global max: 15.1GB

---

# Distributed Training Benchmark Result

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 500
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Training Time Seconds: 66.34
Trainer Reported Samples Per Second: 7.78
Average Non Padding Tokens Per Sample: 55.25
Trainer Based Tokens Per Second: 429
Manual Global Tokens Per Second: 416
Manual Global Total Samples: 500
Manual Global Total Tokens: 27625
Train Loss: 3.05
Perplexity: 21.115
Peak PyTorch VRAM: 11.03GB
Average GPU Utilization: rank0: 87.73%; rank1: 94.18% | global avg: 90.96%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 12.81GB; rank1: 12.89GB | global avg: 12.85GB
Maximum GPU Memory: rank0: 13.76GB; rank1: 13.76GB | global max: 13.76GB

---

# Distributed Training Benchmark Result

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 1500
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Training Time Seconds: 187.89
Trainer Reported Samples Per Second: 8.04
Average Non Padding Tokens Per Sample: 68.69
Trainer Based Tokens Per Second: 552
Manual Global Tokens Per Second: 548
Manual Global Total Samples: 1500
Manual Global Total Tokens: 103029
Train Loss: 2.7278
Perplexity: 15.2986
Peak PyTorch VRAM: 11.03GB
Average GPU Utilization: rank0: 94.91%; rank1: 94.69% | global avg: 94.8%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 13.46GB; rank1: 13.46GB | global avg: 13.46GB
Maximum GPU Memory: rank0: 13.76GB; rank1: 13.76GB | global max: 13.76GB

---

# Distributed Training Benchmark Result

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 1000
Full Dataset Size: 36718
Selected Dataset Size: 1000
Selected Dataset Percentage: 2.72%
Epoch Processed Dataset Percentage: 2.72%
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Communication Profiler Enabled: False
Communication Profiler CUDA Sync Enabled: False
Training Time Seconds: 127.25
Trainer Reported Samples Per Second: 8.01
Average Non Padding Tokens Per Sample: 64.9
Trainer Based Tokens Per Second: 519
Manual Global Tokens Per Second: 510
Manual Global Total Samples: 1000
Manual Global Total Tokens: 64896
Train Loss: 2.7792
Perplexity: 16.106
Peak PyTorch VRAM: 11.03GB
Average GPU Utilization: rank0: 93.06%; rank1: 93.76% | global avg: 93.41%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 13.2GB; rank1: 13.3GB | global avg: 13.25GB
Maximum GPU Memory: rank0: 13.76GB; rank1: 13.76GB | global max: 13.76GB
Communication Summary: Disabled. Set ENABLE_COMM_PROFILER=1 to capture Python-level torch.distributed calls.
Communication Scope: Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.
Communication Log File: Not generated

---

# Distributed Training Benchmark Result

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 1000
Full Dataset Size: 36718
Selected Dataset Size: 1000
Selected Dataset Percentage: 2.72%
Epoch Processed Dataset Percentage: 2.72%
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Communication Profiler Enabled: True
Communication Profiler CUDA Sync Enabled: False
Training Time Seconds: 127.77
Trainer Reported Samples Per Second: 7.98
Average Non Padding Tokens Per Sample: 64.9
Trainer Based Tokens Per Second: 517
Manual Global Tokens Per Second: 507
Manual Global Total Samples: 1000
Manual Global Total Tokens: 64896
Train Loss: 2.7792
Perplexity: 16.1059
Peak PyTorch VRAM: 11.03GB
Average GPU Utilization: rank0: 93.29%; rank1: 95.69% | global avg: 94.49%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 13.2GB; rank1: 13.31GB | global avg: 13.25GB
Maximum GPU Memory: rank0: 13.76GB; rank1: 13.76GB | global max: 13.76GB
Communication Summary: barrier: mean_count_per_rank=3.0, rank_mean_total_ms=20.785, rank_max_total_ms=23.027, rank_max_single_call_ms=18.292, bottleneck_by_slowest_rank=0.02% | all_gather_into_tensor: mean_count_per_rank=63.0, rank_mean_total_ms=6.3, rank_max_total_ms=6.327, rank_max_single_call_ms=0.23, bottleneck_by_slowest_rank=0.0% | broadcast: mean_count_per_rank=8.0, rank_mean_total_ms=0.702, rank_max_total_ms=0.708, rank_max_single_call_ms=0.17, bottleneck_by_slowest_rank=0.0% | all_reduce: mean_count_per_rank=122.0, rank_mean_total_ms=16.453, rank_max_total_ms=16.545, rank_max_single_call_ms=0.203, bottleneck_by_slowest_rank=0.01% | all_gather: mean_count_per_rank=13.0, rank_mean_total_ms=1.943, rank_max_total_ms=2.177, rank_max_single_call_ms=0.457, bottleneck_by_slowest_rank=0.0%
Communication Scope: Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.
Communication Log File: /home/ubuntu/workspace/logs/log.json

---

# Distributed Training Benchmark Result
# Config Json has been reviewed

Mode: ZERO3
Model: Qwen/Qwen2.5-0.5B
Dataset Sample Size: 1000
Full Dataset Size: 36718
Selected Dataset Size: 1000
Selected Dataset Percentage: 2.72%
Epoch Processed Dataset Percentage: 2.72%
Max Sequence Length: 512
World Size: 2
Use ZeRO Init: True
Gradient Checkpointing Enabled: True
Per Device Train Batch Size: 1
Gradient Accumulation Steps: 8
Communication Profiler Enabled: False
Communication Profiler CUDA Sync Enabled: False
Training Time Seconds: 726.58
Trainer Reported Samples Per Second: 1.38
Average Non Padding Tokens Per Sample: 64.9
Trainer Based Tokens Per Second: 89
Manual Global Tokens Per Second: 89
Manual Global Total Samples: 1000
Manual Global Total Tokens: 64896
Train Loss: 2.8071
Perplexity: 16.5617
Peak PyTorch VRAM: 5.17GB
Average GPU Utilization: rank0: 93.95%; rank1: 93.37% | global avg: 93.66%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 7.92GB; rank1: 7.93GB | global avg: 7.92GB
Maximum GPU Memory: rank0: 8.22GB; rank1: 8.22GB | global max: 8.22GB
Communication Summary: Disabled. Set ENABLE_COMM_PROFILER=1 to capture Python-level torch.distributed calls.
Communication Scope: Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.
Communication Log File: Not generated

---