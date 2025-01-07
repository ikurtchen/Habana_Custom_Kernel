# Use in MLM

* Add following to launch script:
```bash
export GC_KERNEL_PATH="/mnt/ctrl/disk2/personal/kurt/develop/Habana_Custom_Kernel/build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so:/usr/lib/habanalabs/libtpc_scalar_kernels.so"
```
* Add following to `pretrain_gpt.py` `main` function:
```python
torch.ops.load_library("/mnt/ctrl/disk2/personal/kurt/develop/Habana_Custom_Kernel/custom_op/bincount_stage1/build/lib.linux-x86_64-cpython-310/hpu_custom_bincount_stage1.cpython-310-x86_64-linux-gnu.so")
```
* Use the kernel in `moe_utils.py`:
```python
     if capacity_factor is None:
         # TopK without capacity
         bincount_stage1 = torch.ops.custom_op.custom_bincount_stage1(top_indices.to(dtype=torch.int32).view(-1), 24, 0, num_experts)
         tokens_per_expert = torch.sum(bincount_stage1, dim=1)
```
