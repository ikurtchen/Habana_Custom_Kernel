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

#include "chatglm2_apply_rope_f32_gaudi2_test.hpp"
#include "entry_points.hpp"

void ChatGLM2ApplyRoPEF32Gaudi2Test::chatglm2_apply_rope_f32_ref(
         const test::Tensor<float,4>& input0,
         const test::Tensor<float,4>& input1,
         test::Tensor<float,4>& output)
{
    int coords[4] = {0};
    int coords_1[4] = {0};
    for (unsigned i = 0; i < input0.Size(0); i += 1)
    {
        coords[0] = i;
        coords_1[0] = i;
        for (unsigned j = 0; j < input0.Size(1); j += 1)
        {
            coords[1] = j;
            coords_1[1] = 0;
            for (unsigned k = 0; k < input0.Size(2); k += 1)
            {
                coords[2] = k;
                coords_1[2] = k;
                for (unsigned m = 0; m < input0.Size(3); m += 1)
                {
                    coords[3] = m;
                    coords_1[3] = m;
                    float x0 = input0.ElementAt(coords);
                    float x1 = input1.ElementAt(coords_1);
                    float y = x0 + x1;
                    output.SetElement(coords, y);
                }
            }
        }
    }
}

int ChatGLM2ApplyRoPEF32Gaudi2Test::runTest()
{
    float_4DTensor input0({64, 32, 1, 1024});
    input0.InitRand(-10.0f, 10.0f);
    float_4DTensor input1({64, 1, 1, 1024});
    input1.InitRand(-10.0f, 10.0f);
    float_4DTensor output({64, 32, 1, 1024});
    float_4DTensor output_ref({64, 32, 1, 1024});

    // execute reference implementation of the kernel.
    chatglm2_apply_rope_f32_ref(input0, input1, output_ref);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_CHATGLM2_APPLY_ROPE_F32]);
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
    // input0.Print(0);
    // input1.Print(0);
    // output.Print(0);
    // output_ref.Print(0);
    std::cout << "--- input 0 ---" << std::endl;
    PrintTensor(input0);
    std::cout << "--- input 1 ---" << std::endl;
    PrintTensor(input1);
    std::cout << "--- output ---" << std::endl;
    PrintTensor(output);
    std::cout << "--- output ref ---" << std::endl;
    PrintTensor(output_ref);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "ChatGLM2 apply RoPE F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "ChatGLM2 apply RoPE F32 test pass!!" << std::endl;
    return 0;
}

void ChatGLM2ApplyRoPEF32Gaudi2Test::PrintTensor(const test::Tensor<float,4>& tensor, std::ostream& os)
{
    os << std::endl;
    for (int d3 = 0; d3 < ((int) tensor.Size(3)); d3++)
    {
        for (int d2 = 0; d2 < ((int) tensor.Size(2)); d2++)
        {
            for (int d1 = 0; d1 < ((int) tensor.Size(1)); d1++)
            {
                for (int d0 = 0; d0 < ((int) tensor.Size(0)); d0++)
                {
                    int coords[] = { d0, d1, d2, d3, 0 };
                    os << std::fixed << std::setw(7) << std::setprecision(4)
                        << + (float)(tensor.ElementAt(coords)) << ",";
                }
            }
        }
        os << std::endl;
    }
}
