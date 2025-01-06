#!/bin/bash

set -ex

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"
BUILD_DIR="${MY_SCRIPT_DIR}/build"

GDB=""
#GDB="gdb --args"

TEST_DEVICE="Gaudi2"
#TEST_KERNEL="VectorAddV1F32Gaudi2Test"
#TEST_KERNEL="VectorAddV2F32Gaudi2Test"
#TEST_KERNEL="MatrixAddV1F32Gaudi2Test"
#TEST_KERNEL="MatrixAddV2F32Gaudi2Test"
TEST_KERNEL="BincountStage1I32Gaudi2Test"

#export TPC_RUNNER=1

#${BUILD_DIR}/tests/tpc_kernel_tests -h
${GDB} ${BUILD_DIR}/tests/tpc_kernel_tests -d ${TEST_DEVICE} -t ${TEST_KERNEL}
