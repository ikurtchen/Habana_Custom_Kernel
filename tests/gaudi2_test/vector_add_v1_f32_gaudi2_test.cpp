/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "vector_add_v1_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

void VectorAddV1F32Gaudi2Test::vector_add_f32_ref(
         const test::Tensor<float,1>& input0,
         const test::Tensor<float,1>& input1,
         test::Tensor<float,1>& output)
{
    int coords[1] = {0};
    for (unsigned d = 0; d < input0.Size(0); d += 1)
    {
        coords[0] = d;
        float x0 = input0.ElementAt(coords);
        float x1 = input1.ElementAt(coords);
        float y = x0 + x1;
        output.SetElement(coords, y);
    }
}

int VectorAddV1F32Gaudi2Test::runTest()
{
    const int vector_size = 1023;
    float_1DTensor input0({vector_size});
    input0.InitRand(-10.0f, 10.0f);
    float_1DTensor input1({vector_size});
    input1.InitRand(-10.0f, 10.0f);
    float_1DTensor output({vector_size});
    float_1DTensor output_ref({vector_size});

    // execute reference implementation of the kernel.
    vector_add_f32_ref(input0, input1, output_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;

    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input0);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input1);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_VECTOR_ADD_V1_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input0.GetTensorDescriptor());
    vec.push_back(input1.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    input0.Print(0);
    input1.Print(0);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Vector Add V1 F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Vector Add V1 F32 test pass!!" << std::endl;
    return 0;
}

