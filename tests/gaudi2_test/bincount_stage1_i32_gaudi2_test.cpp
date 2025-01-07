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

#include "bincount_stage1_i32_gaudi2_test.hpp"
#include "entry_points.hpp"

void BincountStage1I32Gaudi2Test::bincount_stage1_i32_ref(
         const test::Tensor<int,1>& input,
         test::Tensor<int,2>& output, int partiions, int partition_size, int bins)
{
    int input_coords[1] = {0};
    int output_coords[2] = {0};
    for (unsigned i = 0; i < input.Size(0); i += 1)
    {
        input_coords[0] = i;
        int x = input.ElementAt(input_coords);

        output_coords[0] = i / partition_size;
        output_coords[1] = x;

        int y = output.ElementAt(output_coords);
        y += 1;
        output.SetElement(output_coords, y);
    }
}

int BincountStage1I32Gaudi2Test::runTest()
{
    const unsigned int experts = 8; //64; //8
    const unsigned int topk = 2; //8; //2
    const unsigned int partitions = 4; //24; //6, 4
    const unsigned int seq_len = 8; //16384; //8
    const unsigned int bins = experts;
    const unsigned int input_size = seq_len * topk; // b*s*topk
    const unsigned int partition_size = (input_size + partitions - 1) / partitions;

    int32_1DTensor input({input_size});
    input.InitRand(0, experts - 1);

    int32_2DTensor output({partitions, bins});
    int32_2DTensor output_ref({partitions, bins});

    // execute reference implementation of the kernel.
    bincount_stage1_i32_ref(input, output_ref, partitions, partition_size, bins);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_BINCOUNT_STAGE1_I32].name);

    BincountStage1I32Gaudi2::BincountStage1I32Param def;
    def.partitions = partitions;
    def.partition_size = 0;
    def.bins = bins;
    m_in_defs.nodeParams.nodeParams = &def;

    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    input.Print(0);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Bincount stage 1 I32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Bincount stage 1 I32 test pass!!" << std::endl;
    return 0;
}

