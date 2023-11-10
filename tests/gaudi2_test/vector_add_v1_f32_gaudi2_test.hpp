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

#ifndef VECTOR_ADD_V1_F32_GAUDI2_TEST_HPP
#define VECTOR_ADD_V1_F32_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "vector_add_v1_f32_gaudi2.hpp"

class VectorAddV1F32Gaudi2Test : public TestBase
{
public:
    VectorAddV1F32Gaudi2Test() {}
    ~VectorAddV1F32Gaudi2Test() {}
    int runTest();

    static void vector_add_f32_ref(
         const test::Tensor<float,1>& input0,
         const test::Tensor<float,1>& input1,
         test::Tensor<float,1>& output);

private:
    VectorAddV1F32Gaudi2Test(const VectorAddV1F32Gaudi2Test& other) = delete;
    VectorAddV1F32Gaudi2Test& operator=(const VectorAddV1F32Gaudi2Test& other) = delete;

};

#endif /* VECTOR_ADD_V1_F32_GAUDI2_TEST_HPP */
