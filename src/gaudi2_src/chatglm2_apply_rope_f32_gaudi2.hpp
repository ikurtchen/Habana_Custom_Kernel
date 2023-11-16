#ifndef _CHATGLM2_APPLY_ROPE_F32_GAUDI2_HPP
#define _CHATGLM2_APPLY_ROPE_F32_GAUDI2_HPP

#include <vector>
#include <cstring>
#include <gc_interface.h>

class ChatGLM2ApplyRoPEF32Gaudi2
{
public:

    ChatGLM2ApplyRoPEF32Gaudi2() {}
    virtual ~ChatGLM2ApplyRoPEF32Gaudi2() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* params,
                                  gcapi::HabanaKernelInstantiation_t* instance);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME]);

private:
    ChatGLM2ApplyRoPEF32Gaudi2(const ChatGLM2ApplyRoPEF32Gaudi2& other) = delete;
    ChatGLM2ApplyRoPEF32Gaudi2& operator=(const ChatGLM2ApplyRoPEF32Gaudi2& other) = delete;
};

#endif // _CHATGLM2_APPLY_ROPE_F32_GAUDI2_HPP
