#ifndef _MATRIX_ADD_V1_F32_GAUDI2_HPP
#define _MATRIX_ADD_V1_F32_GAUDI2_HPP

#include <vector>
#include <cstring>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class MatrixAddV1F32Gaudi2
{
public:

    MatrixAddV1F32Gaudi2() {}
    virtual ~MatrixAddV1F32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);

private:
    MatrixAddV1F32Gaudi2(const MatrixAddV1F32Gaudi2& other) = delete;
    MatrixAddV1F32Gaudi2& operator=(const MatrixAddV1F32Gaudi2& other) = delete;
};

#endif // _MATRIX_ADD_V1_F32_GAUDI2_HPP
