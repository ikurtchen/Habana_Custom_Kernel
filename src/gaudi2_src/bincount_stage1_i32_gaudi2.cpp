#include "bincount_stage1_i32_gaudi2.hpp"

extern unsigned char _binary___bincount_stage1_i32_gaudi2_o_start;
extern unsigned char _binary___bincount_stage1_i32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn BincountStage1I32Gaudi2::GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_bincount_stage1_i32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn BincountStage1I32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    BincountStage1I32Param* userParams = static_cast<BincountStage1I32Param*>(params->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 1)
    {
        params->inputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr !=1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate tensor dimensions
    //if (params->inputTensors[0].geometry.maxSizes[0] % userParams->partitions != 0)
    //{
    //    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    //}
    if (params->outputTensors[0].geometry.maxSizes[0] != userParams->partitions
            || params->outputTensors[0].geometry.maxSizes[1] != userParams->bins)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // validate input and output data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    if (userParams->partition_size == 0)
    {
        userParams->partition_size = (params->inputTensors[0].geometry.maxSizes[0] + userParams->partitions -1) / userParams->partitions;
    }

    kernel->indexSpaceRank = 1;
    kernel->indexSpaceGeometry[0] = userParams->partitions;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].a  = userParams->partition_size;
    kernel->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].end_b    = userParams->partition_size - 1;

    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a  = 1;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b    = 0;
    //kernel->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    //kernel->outputTensorAccessPattern[0].mapping[1].a  = 1;
    //kernel->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    //kernel->outputTensorAccessPattern[0].mapping[1].end_b    = 0;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    kernel->kernel.paramsNr = sizeof(*userParams)/ sizeof(unsigned int);
    memcpy(&(kernel->kernel.scalarParams[0]), userParams, sizeof(*userParams));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___bincount_stage1_i32_gaudi2_o_end - &_binary___bincount_stage1_i32_gaudi2_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (kernel->kernel.kernelElf,
                &_binary___bincount_stage1_i32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
