#ifndef _MATRIX_ADD_V2_F32_GAUDI2_HPP
#define _MATRIX_ADD_V2_F32_GAUDI2_HPP

#include <vector>
#include <cstring>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class MatrixAddV2F32Gaudi2
{
public:

    MatrixAddV2F32Gaudi2() {}
    virtual ~MatrixAddV2F32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);

private:
    MatrixAddV2F32Gaudi2(const MatrixAddV2F32Gaudi2& other) = delete;
    MatrixAddV2F32Gaudi2& operator=(const MatrixAddV2F32Gaudi2& other) = delete;
};

#endif // _MATRIX_ADD_V2_F32_GAUDI2_HPP
