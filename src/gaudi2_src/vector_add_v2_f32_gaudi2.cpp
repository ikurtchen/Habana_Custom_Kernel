#include <vector>
#include <cstring>
#include <iostream>
#include "vector_add_v2_f32_gaudi2.hpp"

extern unsigned char _binary___vector_add_v2_f32_gaudi2_o_start;
extern unsigned char _binary___vector_add_v2_f32_gaudi2_o_end;

gcapi::GlueCodeReturn_t VectorAddV2F32Gaudi2::GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_vector_add_v2_f32_gaudi2");
    return gcapi::GLUE_SUCCESS;
}

gcapi::GlueCodeReturn_t VectorAddV2F32Gaudi2::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* params,
            gcapi::HabanaKernelInstantiation_t* instance)
{
    gcapi::GlueCodeReturn_t retVal;
    VectorAddV2Param* userParams = static_cast<VectorAddV2Param*>(params->NodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr !=1)
    {
        params->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    //validate tensor dimensions
    if (params->inputTensors[0].geometry.sizes[0] != params->inputTensors[1].geometry.sizes[0])
    {
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (params->outputTensors[0].geometry.sizes[0] != params->inputTensors[0].geometry.sizes[0])
    {
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // validate input and output data type
    if (params->inputTensors[0].dataType != gcapi::DATA_F32 ||
        params->inputTensors[1].dataType != gcapi::DATA_F32 ||
        params->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        params->inputTensors[0].dataType = gcapi::DATA_F32;
        params->inputTensors[1].dataType = gcapi::DATA_F32;
        params->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    int elementsInVec = 64;
    unsigned roundOutputSize0 = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    if (roundOutputSize0 < 8)
    {
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    instance->indexSpaceGeometry.dims = 1;
    instance->indexSpaceGeometry.sizes[0] = 8;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    int elementsInPartition = roundOutputSize0 / 8;
    instance->inputTensorAccessPattern[0].dim[0].dim      = 0;
    instance->inputTensorAccessPattern[0].dim[0].start_a  = elementsInPartition;
    instance->inputTensorAccessPattern[0].dim[0].end_a    = elementsInPartition;
    instance->inputTensorAccessPattern[0].dim[0].start_b  = 0;
    instance->inputTensorAccessPattern[0].dim[0].end_b    = elementsInPartition - 1;

    instance->inputTensorAccessPattern[1].dim[0].dim      = 0;
    instance->inputTensorAccessPattern[1].dim[0].start_a  = elementsInPartition;
    instance->inputTensorAccessPattern[1].dim[0].end_a    = elementsInPartition;
    instance->inputTensorAccessPattern[1].dim[0].start_b  = 0;
    instance->inputTensorAccessPattern[1].dim[0].end_b    = elementsInPartition - 1;

    instance->outputTensorAccessPattern[0].dim[0].dim      = 0;
    instance->outputTensorAccessPattern[0].dim[0].start_a  = elementsInPartition;
    instance->outputTensorAccessPattern[0].dim[0].end_a    = elementsInPartition;
    instance->outputTensorAccessPattern[0].dim[0].start_b  = 0;
    instance->outputTensorAccessPattern[0].dim[0].end_b    = elementsInPartition - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    instance->kernel.paramsNr = sizeof(*userParams)/ sizeof(unsigned int);
    memcpy(&(instance->kernel.scalarParams[0]), userParams, sizeof(*userParams));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___vector_add_v2_f32_gaudi2_o_end - &_binary___vector_add_v2_f32_gaudi2_o_start);
    unsigned givenBinarySize = instance->elfSize;
    instance->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (instance->kernelElf,
                &_binary___vector_add_v2_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}
