#ifndef _MATRIX_ADD_V1_F32_GAUDI2_HPP
#define _MATRIX_ADD_V1_F32_GAUDI2_HPP

#include <vector>
#include <cstring>
#include <gc_interface.h>

class MatrixAddV1F32Gaudi2
{
public:

    MatrixAddV1F32Gaudi2() {}
    virtual ~MatrixAddV1F32Gaudi2() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* params,
                                  gcapi::HabanaKernelInstantiation_t* instance);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME]);

private:
    MatrixAddV1F32Gaudi2(const MatrixAddV1F32Gaudi2& other) = delete;
    MatrixAddV1F32Gaudi2& operator=(const MatrixAddV1F32Gaudi2& other) = delete;
};

#endif // _MATRIX_ADD_V1_F32_GAUDI2_HPP
