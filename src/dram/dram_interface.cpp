#include "dram/dram_interface.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <sstream>

#include "base/request.h"
#include "common/assert.h"
#include "dram/dram_request.h"

namespace llm_system {

namespace {

std::mutex mapped_trace_mutex;

std::string join_addr_vec(const Ramulator::AddrVec_t& addr_vec) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < addr_vec.size(); i++) {
    if (i != 0) {
      oss << ":";
    }
    oss << addr_vec[i];
  }
  return oss.str();
}

std::string to_hex_string(addr address) {
  if (address < 0) {
    return "";
  }
  std::ostringstream oss;
  oss << "0x" << std::hex << std::uppercase << address;
  return oss.str();
}

std::string layer_type_to_string(LayerType layer_type) {
  switch (layer_type) {
    case LayerType::LINEAR:
      return "LINEAR";
    case LayerType::BATCHED_LINEAR:
      return "BATCHED_LINEAR";
    case LayerType::ACTIVATION:
      return "ACTIVATION";
    case LayerType::ATTENTION_GEN:
      return "ATTENTION_GEN";
    case LayerType::ATTENTION_SUM:
      return "ATTENTION_SUM";
    case LayerType::ATTENTION_MIXED:
      return "ATTENTION_MIXED";
    case LayerType::MLA_GEN:
      return "MLA_GEN";
    case LayerType::MLA_SUM:
      return "MLA_SUM";
    case LayerType::MLA_MIXED:
      return "MLA_MIXED";
    case LayerType::ABSORBED_MLA_GEN:
      return "ABSORBED_MLA_GEN";
    case LayerType::ABSORBED_MLA_SUM:
      return "ABSORBED_MLA_SUM";
    case LayerType::MAX:
      return "MODULE";
  }
  return "UNKNOWN";
}

std::string processor_type_to_string(ProcessorType processor_type) {
  switch (processor_type) {
    case ProcessorType::GPU:
      return "GPU";
    case ProcessorType::LOGIC:
      return "LOGIC";
    case ProcessorType::PIM:
      return "PIM";
    case ProcessorType::NONE:
      return "NONE";
    case ProcessorType::MAX:
      return "MAX";
  }
  return "UNKNOWN";
}

}  // namespace

DRAMInterface::DRAMInterface(std::string config_path,
                             double memory_scale_factor,
                             int device_total_rank)
    : memory_scale_factor(memory_scale_factor),
      device_total_rank_(device_total_rank),
      exec_status() {
  std::vector<std::string> params;
  YAML::Node config = Ramulator::Config::parse_config_file(config_path, params);

  frontend = Ramulator::Factory::create_frontend(config);
  memory_system = Ramulator::Factory::create_memory_system(config);

  frontend->connect_memory_system(memory_system);
  memory_system->connect_frontend(frontend);

  frontend_tick = frontend->get_clock_ratio();
  mem_tick = memory_system->get_clock_ratio();

  tick_mult = frontend_tick * mem_tick;
  last_issued_dram_cmd_ = memory_system->get_issued_dram_cmd();

  PIM_KERNEL::init(kernel);

  StringMapInit();
  initializeMappedTrace();
}

void DRAMInterface::resetCounter() {
  exec_status = ExecStatus();
  per_channel_delta_.clear();
}

// Get requests and returns end time of each DRAM requests
void DRAMInterface::HandleRequest(const std::list<DRAMRequest::Ptr>& requests,
                                  cycle_t start_time_cycle) {
  resetCounter();
  for (auto& dram_req : requests) {
    PIMRequest pimrequest;

    // DRAM cycle to Core cycle
    GeneratePIMCommand(dram_req, pimrequest);
    SendRequest(pimrequest);
    run();
    updateStatus(pimrequest);
  }
}

void DRAMInterface::updateStatus(const PIMRequest& pimrequest) {
  cycle_t duration = pimrequest.end - pimrequest.start;  // dram cycle
  const double sample_scale = pimrequest.sample_scale;
  const auto scale_counter = [sample_scale](counter_t value) {
    return static_cast<counter_t>(std::llround(value * sample_scale));
  };
  const auto issued_dram_cmd = memory_system->get_issued_dram_cmd();
  std::vector<std::int64_t> delta_issued_dram_cmd(
      issued_dram_cmd.size(), 0);
  for (std::size_t i = 0; i < issued_dram_cmd.size(); i++) {
    const std::int64_t previous =
        i < last_issued_dram_cmd_.size() ? last_issued_dram_cmd_[i] : 0;
    delta_issued_dram_cmd[i] = issued_dram_cmd[i] - previous;
  }
  last_issued_dram_cmd_ = issued_dram_cmd;
  const auto get_delta = [&delta_issued_dram_cmd](DRAMCommandType command) {
    const auto idx = static_cast<std::size_t>(command);
    if (idx >= delta_issued_dram_cmd.size()) {
      return counter_t{0};
    }
    return static_cast<counter_t>(std::max<std::int64_t>(0, delta_issued_dram_cmd[idx]));
  };

  const time_ns scaled_duration = duration * memory_scale_factor * sample_scale;
  time += scaled_duration;
  exec_status.memory_duration += scaled_duration;
  exec_status.background_time += scaled_duration;

  exec_status.act_count += scale_counter(get_delta(DRAMCommandType::kACT));
  exec_status.read_count += scale_counter(get_delta(DRAMCommandType::kREAD));
  exec_status.write_count += scale_counter(get_delta(DRAMCommandType::kWRITE));
  exec_status.all_act_count +=
      scale_counter(get_delta(DRAMCommandType::kALL_ACT));
  exec_status.all_read_count +=
      scale_counter(get_delta(DRAMCommandType::kALL_READ));
  exec_status.all_write_count +=
      scale_counter(get_delta(DRAMCommandType::kALL_WRITE));
  exec_status.ref_count += scale_counter(get_delta(DRAMCommandType::kREF));

  // Per-channel command delta (accumulated across HandleRequest calls)
  const auto per_channel_cmd = memory_system->get_per_channel_dram_cmd();
  per_channel_delta_.resize(per_channel_cmd.size());
  if (per_channel_acc_.empty()) {
    per_channel_acc_.resize(per_channel_cmd.size());
  }
  for (std::size_t ch = 0; ch < per_channel_cmd.size(); ch++) {
    per_channel_delta_[ch].resize(per_channel_cmd[ch].size(), 0);
    if (per_channel_acc_[ch].empty()) {
      per_channel_acc_[ch].resize(per_channel_cmd[ch].size(), 0);
    }
    for (std::size_t i = 0; i < per_channel_cmd[ch].size(); i++) {
      const std::int64_t prev =
          (ch < last_per_channel_cmd_.size() && i < last_per_channel_cmd_[ch].size())
              ? last_per_channel_cmd_[ch][i] : 0;
      per_channel_delta_[ch][i] = std::max<std::int64_t>(0, per_channel_cmd[ch][i] - prev);
      per_channel_acc_[ch][i] += per_channel_delta_[ch][i];
    }
  }
  last_per_channel_cmd_ = per_channel_cmd;

  // For LPDDR5, order should be below
  /*
  exec_status.act_count += pimrequest.issued_dram_cmd[0];
  exec_status.read_count += pimrequest.issued_dram_cmd[9];
  exec_status.write_count += pimrequest.issued_dram_cmd[10];
  exec_status.all_act_count += pimrequest.issued_dram_cmd[2];
  exec_status.all_read_count += pimrequest.issued_dram_cmd[11];
  exec_status.all_write_count += pimrequest.issued_dram_cmd[12];
  exec_status.ref_count += pimrequest.issued_dram_cmd[15];
  */
}

void DRAMInterface::SendRequest(PIMRequest& pimrequest) {
  if (mapped_trace_enabled_) {
    for (const auto& command : pimrequest.command_queue) {
      traceMappedCommand(command, pimrequest);
    }
    mapped_trace_stream_.flush();
  }
  frontend->send(pimrequest);
}

void DRAMInterface::initializeMappedTrace() {
  const char* mapped_trace_env = std::getenv("LLMSIM_MAPPED_TRACE_PATH");
  if (mapped_trace_env == nullptr || std::string(mapped_trace_env).empty()) {
    return;
  }

  const char* all_devices_env = std::getenv("LLMSIM_TRACE_ALL_DEVICES");
  const bool trace_all_devices =
      all_devices_env != nullptr && std::string(all_devices_env) == "1";
  if (!trace_all_devices && device_total_rank_ != 0) {
    return;
  }

  const char* all_channels_env = std::getenv("LLMSIM_TRACE_ALL_CHANNELS");
  trace_all_channels_ =
      all_channels_env != nullptr && std::string(all_channels_env) == "1";
  const char* channel_filter_env = std::getenv("LLMSIM_TRACE_CHANNEL");
  if (channel_filter_env != nullptr && !std::string(channel_filter_env).empty()) {
    trace_channel_filter_ = std::stoi(channel_filter_env);
  } else if (trace_all_channels_) {
    trace_channel_filter_ = -1;
  } else {
    trace_channel_filter_ = 0;
  }

  mapped_trace_path_ = mapped_trace_env;
  std::filesystem::path path(mapped_trace_path_);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  mapped_trace_stream_.open(mapped_trace_path_, std::ios::out | std::ios::app);
  assertTrue(mapped_trace_stream_.is_open(),
             "Failed to open LLMSIM_MAPPED_TRACE_PATH: " + mapped_trace_path_);
  mapped_trace_enabled_ = true;
  writeMappedTraceHeader();
}

void DRAMInterface::writeMappedTraceHeader() {
  if (!mapped_trace_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mapped_trace_mutex);
  std::filesystem::path path(mapped_trace_path_);
  if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
    return;
  }

  mapped_trace_stream_
      << "event_id,device_rank,dram_req_type,pim_cmd_type,operand_type,"
      << "layer_type,processor_type,module_name,tensor_name,tensor_tag,"
      << "linear_address,linear_address_hex,bundle_idx,sample_scale,addr_vec,"
      << "level0,level1,level2,level3,level4,level5,level6\n";
  mapped_trace_stream_.flush();
}

void DRAMInterface::traceMappedCommand(const PIMCommand& command,
                                       const PIMRequest& pimrequest) {
  if (!mapped_trace_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mapped_trace_mutex);
  mapped_trace_stream_ << mapped_trace_event_id_++ << ","
                       << device_total_rank_ << ","
                       << dramreq_to_string[command.dramreq_type] << ","
                       << pimcmd_to_string[command.pimcmd_type] << ","
                       << pimoperand_to_string[command.op_type] << ","
                       << layer_type_to_string(pimrequest.trace_layer_type) << ","
                       << processor_type_to_string(pimrequest.trace_processor_type) << ","
                       << pimrequest.trace_module_name << ","
                       << pimrequest.trace_tensor_name << ","
                       << pimrequest.trace_tensor_tag << ","
                       << command.linear_addr << ","
                       << to_hex_string(command.linear_addr) << ","
                       << command.bundle_idx << ","
                       << pimrequest.sample_scale << ","
                       << join_addr_vec(command.addr_vec);
  for (std::size_t level = 0; level < 7; level++) {
    mapped_trace_stream_ << ",";
    if (level < command.addr_vec.size()) {
      mapped_trace_stream_ << command.addr_vec[level];
    }
  }
  mapped_trace_stream_ << "\n";
}

PIMRequest& DRAMInterface::GeneratePIMCommand(const DRAMRequest::Ptr request,
                                              PIMRequest& pimrequest) const {
  DRAMRequestType type = request->GetType();
  pimrequest.dramreq_type = type;
  pimrequest.trace_layer_type = request->GetTraceLayerType();
  pimrequest.trace_processor_type = request->GetTraceProcessorType();
  pimrequest.trace_module_name = request->GetTraceModuleName();
  pimrequest.trace_tensor_name = request->GetTraceTensorName();
  pimrequest.trace_tensor_tag = request->GetTraceTensorTag();

  try {
    kernel[int(type)](pimrequest, type, request->operands_, pim_hw_config);
  } catch (std::bad_function_call) {
    notYetImplemented("PIM_KERNEL " + std::to_string(int(type)));
  }

  return pimrequest;
}

void DRAMInterface::run() {
  for (uint64_t i = 0;; i++) {
    memory_system->tick();
    if (memory_system->is_finished()) {
      break;
    }
  }
}

void DRAMInterface::StringMapInit() {
  dramreq_to_string[DRAMRequestType::kRead] = "Read";
  dramreq_to_string[DRAMRequestType::kWrite] = "Write";
  dramreq_to_string[DRAMRequestType::kMove] = "Move";
  dramreq_to_string[DRAMRequestType::kMult] = "Mult";
  dramreq_to_string[DRAMRequestType::kAdd] = "Add";
  dramreq_to_string[DRAMRequestType::kMAD] = "MAD";
  dramreq_to_string[DRAMRequestType::kPMult] = "PMult";
  dramreq_to_string[DRAMRequestType::kCMult] = "CMult";
  dramreq_to_string[DRAMRequestType::kCAdd] = "CAdd";
  dramreq_to_string[DRAMRequestType::kCMAD] = "CMAD";
  dramreq_to_string[DRAMRequestType::kTensor] = "Tensor";
  dramreq_to_string[DRAMRequestType::kTensor_Square] = "Tensor_Square";
  dramreq_to_string[DRAMRequestType::kModup_Evkmult] = "Modup_Evkmult";
  dramreq_to_string[DRAMRequestType::kModDownEpilogue] = "ModDownEpilogue";
  dramreq_to_string[DRAMRequestType::kPMult_Accum] = "PMult_Accum";
  dramreq_to_string[DRAMRequestType::kCMult_Accum] = "CMult_Accum";

  pimcmd_to_string[PIMCommandType::kAdd] = "Add";
  pimcmd_to_string[PIMCommandType::kSub] = "Sub";
  pimcmd_to_string[PIMCommandType::kMult] = "Mult";
  pimcmd_to_string[PIMCommandType::kMAC] = "MAC";
  pimcmd_to_string[PIMCommandType::kDRAM2RF] = "DRAM2RF";
  pimcmd_to_string[PIMCommandType::kRF2DRAM] = "RF2DRAM";
  pimcmd_to_string[PIMCommandType::kRead] = "Read";
  pimcmd_to_string[PIMCommandType::kWrite] = "Write";

  pimoperand_to_string[PIMOperandType::kEvk] = "Evk";
  pimoperand_to_string[PIMOperandType::kModUp] = "ModUp";
  pimoperand_to_string[PIMOperandType::kRF] = "RF";
  pimoperand_to_string[PIMOperandType::kDRAM] = "DRAM";
}
}  // namespace llm_system
