#include "hardware/device.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {

namespace {

std::mutex trace_mutex;

std::string csv_escape(const std::string& value) {
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += "\"";
  return escaped;
}

std::string shape_to_string(const std::vector<int>& shape) {
  std::ostringstream oss;
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) {
      oss << "x";
    }
    oss << shape[i];
  }
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

std::string processor_list_to_string(const std::vector<ProcessorType>& processors) {
  std::ostringstream oss;
  for (size_t i = 0; i < processors.size(); i++) {
    if (i != 0) {
      oss << "+";
    }
    switch (processors[i]) {
      case ProcessorType::GPU:
        oss << "GPU";
        break;
      case ProcessorType::LOGIC:
        oss << "LOGIC";
        break;
      case ProcessorType::PIM:
        oss << "PIM";
        break;
      case ProcessorType::NONE:
        oss << "NONE";
        break;
      case ProcessorType::MAX:
        oss << "MAX";
        break;
    }
  }
  return oss.str();
}

}  // namespace

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if(!config.dram_config_path.empty()){
    dram_cfg_path = config.dram_config_path;
  }
  else if(config.gpu_gen == "H100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    // Select DRAM config based on memory bandwidth
    // HBM3E: 8 TB/s, HBM4: 16 TB/s, GDDR6: 512 GB/s, DDR5: 64 GB/s
    if(config.memory_bandwidth >= 8.0e12) {  // >= 8 TB/s -> HBM4
      dram_cfg_path = "./dram_config_HBM4_baseline.yaml";
    } else if(config.memory_bandwidth >= 4.0e12) {  // >= 4 TB/s -> HBM3E
      dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
    } else if(config.memory_bandwidth >= 256.0e9) {  // >= 256 GB/s -> GDDR6
      dram_cfg_path = "./dram_config_GDDR6.yaml";
    } else {  // < 256 GB/s -> DDR5
      dram_cfg_path = "./dram_config_DDR5.yaml";
    }
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // Select memory config based on bandwidth
    if(config.memory_bandwidth >= 8.0e12) {  // HBM4: 16 TB/s system
      memory_scale_factor = 0.5; // 8.0Gbps pin rate, tCK = 250ps, 1 / 4GHz = 0.25ns; scale = 0.5
      memory_config = hbm4_baseline;
      memory_config.num_cube = config.num_cube;
      memory_config.num_logic_cube = config.num_logic_cube;
    } else if(config.memory_bandwidth >= 4.0e12) {  // HBM3E
      memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
      memory_config = hbm3e_192GB;
      memory_config.num_cube = config.num_cube;
      memory_config.num_logic_cube = config.num_logic_cube;
    } else if(config.memory_bandwidth >= 256.0e9) {  // GDDR6
      memory_scale_factor = 0.5; // GDDR6 2Gbps
      memory_config = gddr6_192GB;
    } else {  // DDR5
      memory_scale_factor = 0.625; // DDR5 3.2Gbps
      memory_config = ddr5_192GB;
    }
  }
  if (config.memory_scale_factor_override > 0.0) {
    memory_scale_factor = config.memory_scale_factor_override;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  use_ramulator = config.use_ramulator;
  perform_execution = false;
  trace_enabled = false;
  trace_event_id = 0;
  initialize_trace();
}

void Device::initialize_trace() {
  const char* trace_env = std::getenv("LLMSIM_TRACE_PATH");
  if (trace_env == nullptr || std::string(trace_env).empty()) {
    return;
  }

  const char* all_devices_env = std::getenv("LLMSIM_TRACE_ALL_DEVICES");
  bool trace_all_devices =
      all_devices_env != nullptr && std::string(all_devices_env) == "1";
  if (!trace_all_devices && device_total_rank != 0) {
    return;
  }

  trace_path = trace_env;
  std::filesystem::path path(trace_path);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  trace_stream.open(trace_path, std::ios::out | std::ios::app);
  assertTrue(trace_stream.is_open(),
             "Failed to open LLMSIM_TRACE_PATH: " + trace_path);
  trace_enabled = true;
  write_trace_header();
}

void Device::write_trace_header() {
  std::lock_guard<std::mutex> lock(trace_mutex);
  std::filesystem::path path(trace_path);
  if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
    return;
  }
  trace_stream
      << "event_id,source,model,stage,device_rank,layer_type,module,tensor,"
      << "tag,category,shape,precision_byte,bytes,process_tokens,"
      << "sum_tokens,gen_tokens,total_sequence_length,average_sequence_length,"
      << "processor_type,parallel_execution,duplicated_input\n";
  trace_stream.flush();
}

void Device::set_dependency() { top_module_graph->set_dependency(); }

bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

void Device::restartGraph() { top_module_graph->restart_graph(); }

void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

void Device::reset_status() { status = StatusBoard(); }
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

void Device::setMemoryObject(Tensor::Ptr tensor) {
  mmap_controller->setMemoryObject(tensor);
}

void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    trace_tensor_accesses(layer_type, tensor_list, sequences_metadata,
                          layer_info, "execution");
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());

    // Accumulate per-layer-type DRAM stats from DRAMInterface
    // (DRAMInterface::exec_status is separate from Device::exec_status)
    const auto& dram_es = dram_interface->getExecStatus();
    auto& stats = per_layer_type_dram_stats_[layer_type];
    stats.act_count += dram_es.act_count;
    stats.read_count += dram_es.read_count;
    stats.write_count += dram_es.write_count;
    stats.all_act_count += dram_es.all_act_count;
    stats.all_read_count += dram_es.all_read_count;
    stats.all_write_count += dram_es.all_write_count;
    stats.ref_count += dram_es.ref_count;
    stats.memory_duration += dram_es.memory_duration;
  }
}

void Device::trace_tensor_accesses(
    LayerType layer_type, const std::vector<Tensor::Ptr>& tensor_list,
    const BatchedSequence::Ptr sequences_metadata, const LayerInfo layer_info,
    const std::string& source) {
  if (!trace_enabled || !perform_execution) {
    return;
  }

  std::string stage = "mixed";
  if (config.prefill_mode) {
    stage = "prefill";
  } else if (config.decode_mode) {
    stage = "decode";
  }
  const char* stage_env = std::getenv("LLMSIM_TRACE_STAGE");
  if (stage_env != nullptr && !std::string(stage_env).empty()) {
    stage = stage_env;
  }

  int process_tokens = 0;
  int sum_tokens = 0;
  int gen_tokens = 0;
  int total_sequence_length = 0;
  int average_sequence_length = 0;
  if (sequences_metadata != nullptr) {
    process_tokens = sequences_metadata->get_process_token();
    sum_tokens = sequences_metadata->get_sum_process_token();
    gen_tokens = sequences_metadata->get_gen_process_token();
    total_sequence_length = sequences_metadata->get_total_sequence_length();
    average_sequence_length = sequences_metadata->get_average_sequence_length();
  }

  std::lock_guard<std::mutex> lock(trace_mutex);
  for (const Tensor::Ptr& tensor : tensor_list) {
    if (tensor == nullptr) {
      continue;
    }
    std::string category = tensor->tag;
    if (category == "act") {
      category = "activation";
    } else if (category == "cache") {
      category = "kv_cache";
    }

    trace_stream << trace_event_id++ << ","
                 << csv_escape(source) << ","
                 << csv_escape(model_config.model_name) << ","
                 << csv_escape(stage) << ","
                 << device_total_rank << ","
                 << csv_escape(layer_type_to_string(layer_type)) << ","
                 << csv_escape(tensor->get_module_map_name()) << ","
                 << csv_escape(tensor->name) << ","
                 << csv_escape(tensor->tag) << ","
                 << csv_escape(category) << ","
                 << csv_escape(shape_to_string(tensor->shape)) << ","
                 << tensor->precision_byte << ","
                 << tensor->getSize() << ","
                 << process_tokens << ","
                 << sum_tokens << ","
                 << gen_tokens << ","
                 << total_sequence_length << ","
                 << average_sequence_length << ","
                 << csv_escape(processor_list_to_string(layer_info.processor_type))
                 << "," << (layer_info.parallel_execution ? 1 : 0)
                 << "," << (layer_info.duplicated_input ? 1 : 0)
                 << "\n";
  }
  trace_stream.flush();
}

void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

void Device::run_ramulator(DRAMRequest_Ptr dram_request) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  dram_interface->HandleRequest(request, 0);
}

void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = (total_read % num_cube == 0) ? (total_read / num_cube) : ((total_read / num_cube) + 1);

  long rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 % num_channel == 0) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);
  long rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 % num_channel == 1) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);

  dram_interface->resetCounter();
  dram_interface->getExecStatus().act_count = (((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) < num_col) ? 1 : ((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) / num_col));
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = std::max(
        1, mmap_controller->getConfig().num_cube *
               mmap_controller->getConfig().num_channel / 2);
  }
  else {
    num_pseudo_ch = std::max(
        1, mmap_controller->getConfig().num_cube *
               mmap_controller->getConfig().num_channel);
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREF_energy_j_ *= num_pseudo_ch;
  dramEnergy.kBACKGROUND_power_nW_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

};  // namespace llm_system
