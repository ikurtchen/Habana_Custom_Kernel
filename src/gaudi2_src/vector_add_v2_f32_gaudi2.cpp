#include <vector>
#include <cstring>
#include <iostream>
#include "vector_add_v2_f32_gaudi2.hpp"

extern unsigned char _binary___vector_add_v2_f32_gaudi2_o_start;
extern unsigned char _binary___vector_add_v2_f32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn VectorAddV2F32Gaudi2::GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_vector_add_v2_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn VectorAddV2F32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    VectorAddV2Param* userParams = static_cast<VectorAddV2Param*>(params->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr !=1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate tensor dimensions
    if (params->inputTensors[0].geometry.maxSizes[0] != params->inputTensors[1].geometry.maxSizes[0])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (params->outputTensors[0].geometry.maxSizes[0] != params->inputTensors[0].geometry.maxSizes[0])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // validate input and output data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        params->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    uint64_t outputSizes[tpc_lib_api::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    int elementsInVec = 64;
    uint64_t roundOutputSize0 = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    if (roundOutputSize0 < 8)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    kernel->indexSpaceRank = 1;
    kernel->indexSpaceGeometry[0] = 8;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    int elementsInPartition = roundOutputSize0 / 8;
    kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].a  = elementsInPartition;
    kernel->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInPartition - 1;

    kernel->inputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].a  = elementsInPartition;
    kernel->inputTensorAccessPattern[1].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].end_b    = elementsInPartition - 1;

    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a  = elementsInPartition;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInPartition - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    kernel->kernel.paramsNr = sizeof(*userParams)/ sizeof(unsigned int);
    memcpy(&(kernel->kernel.scalarParams[0]), userParams, sizeof(*userParams));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___vector_add_v2_f32_gaudi2_o_end - &_binary___vector_add_v2_f32_gaudi2_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (kernel->kernel.kernelElf,
                &_binary___vector_add_v2_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
