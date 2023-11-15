#include <vector>
#include <cstring>
#include <iostream>
#include "matrix_add_v2_f32_gaudi2.hpp"

extern unsigned char _binary___matrix_add_v2_f32_gaudi2_o_start;
extern unsigned char _binary___matrix_add_v2_f32_gaudi2_o_end;

gcapi::GlueCodeReturn_t MatrixAddV2F32Gaudi2::GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_matrix_add_v2_f32_gaudi2");
    return gcapi::GLUE_SUCCESS;
}

gcapi::GlueCodeReturn_t MatrixAddV2F32Gaudi2::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* params,
            gcapi::HabanaKernelInstantiation_t* instance)
{
    gcapi::GlueCodeReturn_t retVal;
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
    int elementsInVec = 64;
    int unrollCount = 4;
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    instance->indexSpaceGeometry.dims = 2;
    //round up to elementsInVec and divide by elementsInVec.
    unsigned size0 = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    unsigned size1 = (outputSizes[1] + (unrollCount - 1)) / unrollCount;
    instance->indexSpaceGeometry.sizes[0] = size0;
    instance->indexSpaceGeometry.sizes[1] = size1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    instance->inputTensorAccessPattern[0].dim[0].dim      = 0;
    instance->inputTensorAccessPattern[0].dim[0].start_a  = elementsInVec;
    instance->inputTensorAccessPattern[0].dim[0].end_a    = elementsInVec;
    instance->inputTensorAccessPattern[0].dim[0].start_b  = 0;
    instance->inputTensorAccessPattern[0].dim[0].end_b    = elementsInVec - 1;
    instance->inputTensorAccessPattern[0].dim[1].dim      = 1;
    instance->inputTensorAccessPattern[0].dim[1].start_a  = unrollCount;
    instance->inputTensorAccessPattern[0].dim[1].end_a    = unrollCount;
    instance->inputTensorAccessPattern[0].dim[1].start_b  = 0;
    instance->inputTensorAccessPattern[0].dim[1].end_b    = unrollCount - 1;

    instance->inputTensorAccessPattern[1].dim[0].dim      = 0;
    instance->inputTensorAccessPattern[1].dim[0].start_a  = elementsInVec;
    instance->inputTensorAccessPattern[1].dim[0].end_a    = elementsInVec;
    instance->inputTensorAccessPattern[1].dim[0].start_b  = 0;
    instance->inputTensorAccessPattern[1].dim[0].end_b    = elementsInVec - 1;
    instance->inputTensorAccessPattern[1].dim[1].dim      = 1;
    instance->inputTensorAccessPattern[1].dim[1].start_a  = unrollCount;
    instance->inputTensorAccessPattern[1].dim[1].end_a    = unrollCount;
    instance->inputTensorAccessPattern[1].dim[1].start_b  = 0;
    instance->inputTensorAccessPattern[1].dim[1].end_b    = unrollCount - 1;

    instance->outputTensorAccessPattern[0].dim[0].dim      = 0;
    instance->outputTensorAccessPattern[0].dim[0].start_a  = elementsInVec;
    instance->outputTensorAccessPattern[0].dim[0].end_a    = elementsInVec;
    instance->outputTensorAccessPattern[0].dim[0].start_b  = 0;
    instance->outputTensorAccessPattern[0].dim[0].end_b    = elementsInVec - 1;
    instance->outputTensorAccessPattern[0].dim[1].dim      = 1;
    instance->outputTensorAccessPattern[0].dim[1].start_a  = unrollCount;
    instance->outputTensorAccessPattern[0].dim[1].end_a    = unrollCount;
    instance->outputTensorAccessPattern[0].dim[1].start_b  = 0;
    instance->outputTensorAccessPattern[0].dim[1].end_b    = unrollCount - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___matrix_add_v2_f32_gaudi2_o_end - &_binary___matrix_add_v2_f32_gaudi2_o_start);
    unsigned givenBinarySize = instance->elfSize;
    instance->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (instance->kernelElf,
                &_binary___matrix_add_v2_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}
