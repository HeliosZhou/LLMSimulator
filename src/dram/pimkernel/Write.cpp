#include <dram/dram_interface.h>
#include <dram/pimkernel/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

void Write_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                  DRAMRequest::PIM_Operand& operand,
                  const PIMHWConfig pim_hw_config) {
  auto write_operand = get_operand(operand, PIMOperandType::kDRAM);
  assertTrue((write_operand.size() == 1), "only one request can be written");
  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  // std::cout << "size: " << std::to_string(write_operand[0]->getBundleSize())
  //           << std::endl;
  for (auto opnd : write_operand) {
    const int sample_stride = pim_hw_config.ramulator_sample_stride > 0
                                  ? pim_hw_config.ramulator_sample_stride
                                  : 1;
    long long eligible_commands = 0;
    long long sampled_commands = 0;
    for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
      addr_vec = opnd->getAddrVec(bundle_idx, pim_hw_config.type);
      if (addr_vec.at(0) == 0) {
        eligible_commands++;
      }
      if (addr_vec.at(0) == 0 &&
          (eligible_commands - 1) % sample_stride == 0) {
        sampled_commands++;
        pim_request.AddCommand(PIMCommand(PIMCommandType::kWrite,
                                          PIMOperandType::kDRAM, addr_vec,
                                          &pim_request, dramreq_type));
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
