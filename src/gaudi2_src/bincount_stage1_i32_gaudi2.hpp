#ifndef _BINCOUNT_STAGE1_I32_GAUDI2_HPP
#define _BINCOUNT_STAGE1_I32_GAUDI2_HPP

#include <vector>
#include <cstring>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class BincountStage1I32Gaudi2
{
public:

    BincountStage1I32Gaudi2() {}
    virtual ~BincountStage1I32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);

    struct BincountStage1I32Param
    {
        unsigned int partitions;
        unsigned int partition_size;
        unsigned int bins;
    };
private:
    BincountStage1I32Gaudi2(const BincountStage1I32Gaudi2& other) = delete;
    BincountStage1I32Gaudi2& operator=(const BincountStage1I32Gaudi2& other) = delete;
};

#endif // _BINCOUNT_STAGE1_I32_GAUDI2_HPP
