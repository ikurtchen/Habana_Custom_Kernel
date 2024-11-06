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

#ifndef MATRIX_ADD_V2_F32_GAUDI2_TEST_HPP
#define MATRIX_ADD_V2_F32_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "matrix_add_v2_f32_gaudi2.hpp"

class MatrixAddV2F32Gaudi2Test : public TestBase
{
public:
    MatrixAddV2F32Gaudi2Test() {}
    ~MatrixAddV2F32Gaudi2Test() {}
    int runTest();

    static void matrix_add_f32_ref(
         const test::Tensor<float,2>& input0,
         const test::Tensor<float,2>& input1,
         test::Tensor<float,2>& output);

private:
    MatrixAddV2F32Gaudi2Test(const MatrixAddV2F32Gaudi2Test& other) = delete;
    MatrixAddV2F32Gaudi2Test& operator=(const MatrixAddV2F32Gaudi2Test& other) = delete;

};

#endif /* MATRIX_ADD_V2_F32_GAUDI2_TEST_HPP */
