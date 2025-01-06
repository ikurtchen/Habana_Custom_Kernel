#!/bin/bash

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"

#pip install pytest

export GC_KERNEL_PATH=${MY_SCRIPT_DIR}/../../build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so:/usr/lib/habanalabs/libtpc_scalar_kernels.so

pytest -s --custom_op_lib ${MY_SCRIPT_DIR}/build/lib.linux-x86_64-cpython-310/hpu_custom_bincount_stage1.cpython-310-x86_64-linux-gnu.so hpu_custom_op_bincount_stage1_test.py
