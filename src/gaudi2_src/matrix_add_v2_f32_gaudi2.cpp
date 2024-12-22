#include <vector>
#include <cstring>
#include <iostream>
#include "matrix_add_v2_f32_gaudi2.hpp"

extern unsigned char _binary___matrix_add_v2_f32_gaudi2_o_start;
extern unsigned char _binary___matrix_add_v2_f32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn MatrixAddV2F32Gaudi2::GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_matrix_add_v2_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn MatrixAddV2F32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
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
    int elementsInVec = 64;
    int unrollCount = 4;
    uint64_t outputSizes[tpc_lib_api::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    uint64_t size0 = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    uint64_t size1 = (outputSizes[1] + (unrollCount - 1)) / unrollCount;
    kernel->indexSpaceRank = 2;
    kernel->indexSpaceGeometry[0] = size0;
    kernel->indexSpaceGeometry[1] = size1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].a  = elementsInVec;
    kernel->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
    kernel->inputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    kernel->inputTensorAccessPattern[0].mapping[1].a  = unrollCount;
    kernel->inputTensorAccessPattern[0].mapping[1].start_b  = 0;
    kernel->inputTensorAccessPattern[0].mapping[1].end_b    = unrollCount - 1;

    kernel->inputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].a  = elementsInVec;
    kernel->inputTensorAccessPattern[1].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].end_b    = elementsInVec - 1;
    kernel->inputTensorAccessPattern[1].mapping[1].indexSpaceDim      = 1;
    kernel->inputTensorAccessPattern[1].mapping[1].a  = unrollCount;
    kernel->inputTensorAccessPattern[1].mapping[1].start_b  = 0;
    kernel->inputTensorAccessPattern[1].mapping[1].end_b    = unrollCount - 1;

    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a  = elementsInVec;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
    kernel->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    kernel->outputTensorAccessPattern[0].mapping[1].a  = unrollCount;
    kernel->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[1].end_b    = unrollCount - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___matrix_add_v2_f32_gaudi2_o_end - &_binary___matrix_add_v2_f32_gaudi2_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (kernel->kernel.kernelElf,
                &_binary___matrix_add_v2_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
