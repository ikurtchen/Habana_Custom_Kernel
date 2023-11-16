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

#ifndef CHATGLM2_APPLY_ROPE_F32_GAUDI2_TEST_HPP
#define CHATGLM2_APPLY_ROPE_F32_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "chatglm2_apply_rope_f32_gaudi2.hpp"

class ChatGLM2ApplyRoPEF32Gaudi2Test : public TestBase
{
public:
    ChatGLM2ApplyRoPEF32Gaudi2Test() {}
    ~ChatGLM2ApplyRoPEF32Gaudi2Test() {}
    int runTest();

    static void chatglm2_apply_rope_f32_ref(
         const test::Tensor<float,4>& input0,
         const test::Tensor<float,4>& input1,
         test::Tensor<float,4>& output);
    static void PrintTensor(
         const test::Tensor<float,4>& input0,
         std::ostream& os = std::cout);

private:
    ChatGLM2ApplyRoPEF32Gaudi2Test(const ChatGLM2ApplyRoPEF32Gaudi2Test& other) = delete;
    ChatGLM2ApplyRoPEF32Gaudi2Test& operator=(const ChatGLM2ApplyRoPEF32Gaudi2Test& other) = delete;

};

#endif /* CHATGLM2_APPLY_ROPE_F32_GAUDI2_TEST_HPP */
