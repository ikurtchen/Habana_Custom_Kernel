import torch
import habana_frameworks.torch.core

def test_custom_bincount_stage1_op_function(custom_op_lib_path):
    torch.ops.load_library(custom_op_lib_path)
    print(torch.ops.custom_op.custom_bincount_stage1)

    experts = 64 #8
    topk = 8 #2
    partitions = 24 #4
    seq_len = 16384 #4
    bins = experts
    input_size = seq_len * topk # b*s*topk
    partition_size = (input_size + partitions - 1) // partitions;

    input_cpu = torch.randint(0, experts, (seq_len, topk), dtype=torch.int32)
    input_hpu = input_cpu.to('hpu')

    # cpu
    output_cpu = torch.bincount(input_cpu.view(-1), minlength=bins)

    # hpu
    bincount_stage1 = torch.ops.custom_op.custom_bincount_stage1(input_hpu.view(-1), partitions, partition_size, bins)
    output_hpu = torch.sum(bincount_stage1, dim=1)

    torch.hpu.synchronize()

    print(f"output_cpu={output_cpu}")
    print(f"output_hpu={output_hpu}")

    assert(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))
