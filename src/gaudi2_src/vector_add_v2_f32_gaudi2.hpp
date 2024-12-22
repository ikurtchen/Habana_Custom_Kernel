#ifndef _VECTOR_ADD_V2_F32_GAUDI2_HPP
#define _VECTOR_ADD_V2_F32_GAUDI2_HPP

#include <vector>
#include <cstring>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class VectorAddV2F32Gaudi2
{
public:

    VectorAddV2F32Gaudi2() {}
    virtual ~VectorAddV2F32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);

    struct VectorAddV2Param
    {
        unsigned int partitionStep;
    };
private:
    VectorAddV2F32Gaudi2(const VectorAddV2F32Gaudi2& other) = delete;
    VectorAddV2F32Gaudi2& operator=(const VectorAddV2F32Gaudi2& other) = delete;
};

#endif // _VECTOR_ADD_V2_F32_GAUDI2_HPP
