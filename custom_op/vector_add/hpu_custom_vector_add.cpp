
// TODO: import exposed file
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

bool register_custom_vector_add() {
    // Registering ustom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_0_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_1_desc{
        habana::custom_op::input_type::TENSOR, 1};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_0_desc, input_1_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback

    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_vector_add", //schema name
        "custom_vector_add_v1_f32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_vector_add\n";
    return true;
}

at::Tensor custom_vector_add_execute(
    torch::Tensor input_0,
    torch::Tensor input_1) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_vector_add();
  TORCH_CHECK(registered, "custom_vector_add kernel not registered" );
  std::vector<c10::IValue> inputs{input_0, input_1};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_vector_add");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_vector_add(Tensor input_0, Tensor input_1) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_vector_add", custom_vector_add_execute);
}
