import sys
import time
import torch
import habana_frameworks.torch as ht

from torch.profiler import profile, ProfilerActivity

enable_profile = False

seq_len = 16384
experts = 64
topk = 8
partitions = 16
bins = experts
input_size = seq_len * topk # b*s*topk
partition_size = input_size // partitions;
num_iterations = 100

custom_op_lib_path = sys.argv[1]
torch.ops.load_library(custom_op_lib_path)

input = torch.randint(0, experts, (seq_len, topk), dtype=torch.int64, device='hpu')

# warmup for 10 iterations
for _ in range(10):
    #output = torch.bincount(input.view(-1), minlength=experts)
    bincount_stage1 = torch.ops.custom_op.custom_bincount_stage1(input.view(-1), partitions, partition_size, bins)
    output_hpu = torch.sum(bincount_stage1, dim=1)
    torch.hpu.synchronize()
torch.hpu.synchronize()

# benchmark for 100 iterations and calculate the average time
#start_time = time.time()
fwd_time_list = []

# profile
if enable_profile:
    activities = [ProfilerActivity.CPU, ProfilerActivity.HPU]
    profiler = profile(activities=activities,
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("hpu_profile_custom_op", use_gzip=True),
        with_stack=True, profile_memory=False, with_flops=False,
        with_modules=False, record_shapes=False)
    profiler.start()

for _ in range(num_iterations):
    start_time = time.time()

    #output = torch.bincount(input.view(-1), minlength=experts)
    bincount_stage1 = torch.ops.custom_op.custom_bincount_stage1(input.view(-1), partitions, partition_size, bins)
    output_hpu = torch.sum(bincount_stage1, dim=1)

    torch.hpu.synchronize()

    fwd_time = time.time() - start_time
    fwd_time_list.append(fwd_time)

torch.hpu.synchronize()

if enable_profile:
    profiler.step()
    profiler.stop()

#fwd_time = time.time() - start_time
#avg_fwd_time = fwd_time / num_iterations
avg_fwd_time = float(sum(fwd_time_list)) / len(fwd_time_list)

print(f"{fwd_time_list}")
print(f"Average time: {avg_fwd_time:.6f} s")
