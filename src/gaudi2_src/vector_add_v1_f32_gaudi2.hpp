#ifndef _VECTOR_ADD_V1_F32_GAUDI2_HPP
#define _VECTOR_ADD_V1_F32_GAUDI2_HPP

#include <vector>
#include <cstring>
#include <gc_interface.h>

class VectorAddV1F32Gaudi2
{
public:

    VectorAddV1F32Gaudi2() {}
    virtual ~VectorAddV1F32Gaudi2() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* params,
                                  gcapi::HabanaKernelInstantiation_t* instance);

     virtual gcapi::GlueCodeReturn_t GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME]);

private:
    VectorAddV1F32Gaudi2(const VectorAddV1F32Gaudi2& other) = delete;
    VectorAddV1F32Gaudi2& operator=(const VectorAddV1F32Gaudi2& other) = delete;
};

#endif // _VECTOR_ADD_V1_F32_GAUDI2_HPP
