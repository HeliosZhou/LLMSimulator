#include "dram/dram_interface.h"
#include "dram/pimkernel/pim_kernel.h"

namespace llm_system {
namespace PIM_KERNEL {

namespace {

bool should_trace_channel(const Ramulator::AddrVec_t& addr_vec,
                          const PIMHWConfig& pim_hw_config) {
  if (addr_vec.empty()) {
    return false;
  }
  if (pim_hw_config.trace_all_channels || pim_hw_config.trace_channel_filter < 0) {
    return true;
  }
  return addr_vec.at(0) == pim_hw_config.trace_channel_filter;
}

}  // namespace

void Read_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                 DRAMRequest::PIM_Operand& operand,
                 const PIMHWConfig pim_hw_config) {
  auto read_operand = get_operand(operand, PIMOperandType::kDRAM);
  assertTrue((read_operand.size() == 1), "only one request can be read");

  Ramulator::AddrVec_t addr_vec;

  // std::cout << "size: " << std::to_string(read_operand[0]->getBundleSize())
  //           << std::endl;
  for (auto opnd : read_operand) {
    const int sample_stride = pim_hw_config.ramulator_sample_stride > 0
                                  ? pim_hw_config.ramulator_sample_stride
                                  : 1;
    long long eligible_commands = 0;
    long long sampled_commands = 0;
    for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
        addr_vec = opnd->getAddrVec(bundle_idx, pim_hw_config.type);
        if (should_trace_channel(addr_vec, pim_hw_config)) {
          eligible_commands++;
        }
        if (should_trace_channel(addr_vec, pim_hw_config) &&
            (eligible_commands - 1) % sample_stride == 0) {
          sampled_commands++;
          PIMCommand command(PIMCommandType::kRead, PIMOperandType::kDRAM,
                             addr_vec, &pim_request, dramreq_type);
          command.linear_addr = opnd->getTargetAddress(bundle_idx);
          command.bundle_idx = bundle_idx;
          pim_request.AddCommand(std::move(command));
        }
    }
    if (sampled_commands > 0) {
      pim_request.sample_scale =
          static_cast<double>(eligible_commands) / sampled_commands;
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system
