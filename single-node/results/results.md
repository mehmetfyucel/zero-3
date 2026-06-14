#Auto config + overlap_comm: true
# Distributed Training Benchmark Result

Mode: ZERO3_AUTO
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
Training Time Seconds: 220.54
Trainer Reported Samples Per Second: 5.01
Average Non Padding Tokens Per Sample: 64.9
Trainer Based Tokens Per Second: 325
Manual Global Tokens Per Second: 294
Manual Global Total Samples: 1000
Manual Global Total Tokens: 64896
Train Loss: 2.8387
Perplexity: 17.093
Peak PyTorch VRAM: rank0: 5.52GB; rank1: 5.52GB | global max: 5.52GB
Average GPU Utilization: rank0: 54.48%; rank1: 54.05% | global avg: 54.27%
Maximum GPU Utilization: rank0: 100.0%; rank1: 88.0% | global max: 100.0%
Average GPU Memory: rank0: 8.08GB; rank1: 8.12GB | global avg: 8.1GB
Maximum GPU Memory: rank0: 9.12GB; rank1: 9.16GB | global max: 9.16GB
Communication Summary: Disabled. Set ENABLE_COMM_PROFILER=1 to capture Python-level torch.distributed calls.
Communication Scope: Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.
Communication Log File: Not generated

---

#Manual config overlap_comm: true
# Distributed Training Benchmark Result

Mode: ZERO3_MANUAL
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
Training Time Seconds: 187.13
Trainer Reported Samples Per Second: 5.4
Average Non Padding Tokens Per Sample: 64.9
Trainer Based Tokens Per Second: 350
Manual Global Tokens Per Second: 346
Manual Global Total Samples: 1000
Manual Global Total Tokens: 64896
Train Loss: 2.8386
Perplexity: 17.0925
Peak PyTorch VRAM: rank0: 5.17GB; rank1: 5.17GB | global max: 5.17GB
Average GPU Utilization: rank0: 73.23%; rank1: 74.98% | global avg: 74.11%
Maximum GPU Utilization: rank0: 100.0%; rank1: 100.0% | global max: 100.0%
Average GPU Memory: rank0: 9.2GB; rank1: 9.23GB | global avg: 9.21GB
Maximum GPU Memory: rank0: 9.78GB; rank1: 9.77GB | global max: 9.78GB
Communication Summary: Disabled. Set ENABLE_COMM_PROFILER=1 to capture Python-level torch.distributed calls.
Communication Scope: Python-level torch.distributed only. DeepSpeed ZeRO-3 internal NCCL/C++ communication may not be fully captured.
Communication Log File: Not generated

---
