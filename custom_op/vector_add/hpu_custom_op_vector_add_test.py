import torch
import habana_frameworks.torch.core

def test_custom_vector_add_op_function(custom_op_lib_path):
    torch.ops.load_library(custom_op_lib_path)
    print(torch.ops.custom_op.custom_vector_add)
    a_cpu = torch.rand(1024)
    #print(f"a_cpu={a_cpu}")
    b_cpu = torch.rand(1024)
    #print(f"b_cpu={b_cpu}")
    a_hpu = a_cpu.to('hpu')
    b_hpu = b_cpu.to('hpu')
    c_hpu = torch.ops.custom_op.custom_vector_add(a_hpu, b_hpu)
    #print(f"c_hpu={c_hpu}")
    c_cpu = a_cpu + b_cpu
    #print(f"c_hpu={c_cpu}")
    assert(torch.equal(c_hpu.detach().cpu(), c_cpu.detach().cpu()))

