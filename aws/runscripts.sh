cd /home/ubuntu/workspace
source /home/ubuntu/workspace/configs/llm_env/bin/python3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp39s0
export GLOO_SOCKET_IFNAME=enp39s0
export DS_SKIP_CUDA_CHECK=1
export USE_ZERO_INIT=1
export DATASET_SAMPLE_SIZE=1000
export ENABLE_COMM_PROFILER=0
export SYNC_COMM_PROFILER=0

RUN_MODE=ZERO3 python3 -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=18.184.233.233 \
  --master_port=29500 \
  scripts/main.py


cd /home/ubuntu/workspace
source /home/ubuntu/workspace/configs/llm_env/bin/python3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp39s0
export GLOO_SOCKET_IFNAME=enp39s0
export DS_SKIP_CUDA_CHECK=1
export USE_ZERO_INIT=1
export DATASET_SAMPLE_SIZE=1000
export ENABLE_COMM_PROFILER=0
export SYNC_COMM_PROFILER=0

RUN_MODE=ZERO3 python3 -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=1 \
  --master_addr=18.184.233.233 \
  --master_port=29500 \
  scripts/main.py
