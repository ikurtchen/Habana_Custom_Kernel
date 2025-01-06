#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

typedef struct sBincountStage1Param{
    unsigned int partitions;
    unsigned int partition_size;
    unsigned int bins;
}bincountStage1Param;

bool register_custom_bincount_stage1() {
    // Registering ustom_op::custom_bincount_stage1
    // inputs desc
    habana::custom_op::InputDesc input_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_partitions_desc{
        habana::custom_op::input_type::SCALAR, 1};
    habana::custom_op::InputDesc input_partition_size_desc{
        habana::custom_op::input_type::SCALAR, 2};
    habana::custom_op::InputDesc input_bins_desc{
        habana::custom_op::input_type::SCALAR, 3};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_desc, input_partitions_desc,
        input_partition_size_desc, input_bins_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto partitions = inputs[1].toInt();
      auto bins = inputs[3].toInt();
      std::vector<int64_t> result_sizes = {bins, partitions};
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Int, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(bincountStage1Param);
      params->partitions = inputs[1].toInt();
      params->partition_size = inputs[2].toInt();
      params->bins = inputs[3].toInt();
      return params;
    };

    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_bincount_stage1", //schema name
        "custom_bincount_stage1_i32_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_bincount_stage1\n";
    return true;
}

at::Tensor custom_bincount_stage1_execute(
    torch::Tensor input, c10::Scalar partitions,
    c10::Scalar partition_size, c10::Scalar bins) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_bincount_stage1();
  TORCH_CHECK(registered, "custom_bincount_stage1 kernel not registered" );
  std::vector<c10::IValue> inputs{input, partitions, partition_size, bins};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_bincount_stage1");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_bincount_stage1(Tensor input, Scalar partitions, Scalar partition_size, Scalar bins) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_bincount_stage1", custom_bincount_stage1_execute);
}
