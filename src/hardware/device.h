#pragma once
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_config.h"
#include "hardware/executor.h"
#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "module/status.h"
#include "scheduler/sequence.h"
#include "dram/power.h"

namespace llm_system {

class Device : public std::enable_shared_from_this<Device> {
  friend class Executor;
  friend class Cluster;

 public:
  using Ptr = std::shared_ptr<Device>;

  [[nodiscard]] static Ptr Create(SystemConfig config, int device_total_rank,
                                  Cluster_ptr cluster) {
    Device::Ptr ptr = Ptr(new Device(config, device_total_rank, cluster));
    ptr->connectTopModuleGraph();
    return ptr;
  }

  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  hw_metric memory_capacity;

  SystemConfig config;

  // rank in node
  int device_local_rank;
  // rank in cluster
  int device_total_rank;

  Device() = default;

  Device::Ptr get_ptr() { return shared_from_this(); }

  TopModuleGraph_ptr top_module_graph;

  void connectTopModuleGraph();
  void reset_status();
  void reset_timeboard();
  void set_dependency();

  bool check_module_graph_remain();
  void add_module(std::string name, Module_ptr module);

  void setModelConfig(ModelConfig& _model_config) {
    model_config = _model_config;
  }

  // Per-layer-type DRAM command statistics
  struct LayerTypeDramStats {
    counter_t act_count = 0;
    counter_t read_count = 0;
    counter_t write_count = 0;
    counter_t all_act_count = 0;
    counter_t all_read_count = 0;
    counter_t all_write_count = 0;
    counter_t ref_count = 0;
    time_ns memory_duration = 0;
  };

  const std::map<LayerType, LayerTypeDramStats>& getPerLayerTypeDramStats() const {
    return per_layer_type_dram_stats_;
  }

  time_ns get_time() { return status.device_time; }
  void set_time(time_ns time) { status.device_time = time; }

  // run with module_graph
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);

  void restartGraph();

  void setPerformExecution(bool perform) { perform_execution = perform; }

  // execute operations and update time;
  void execution(Tensor_Ptr input, Tensor_Ptr weight, Tensor_Ptr output);
  void execution(LayerType layer_type,
                 const std::vector<Tensor_Ptr>& tensor_list,
                 const BatchedSequence::Ptr sequences_metadata,
                 const LayerInfo layer_info);
  void trace_tensor_accesses(LayerType layer_type,
                             const std::vector<Tensor_Ptr>& tensor_list,
                             const BatchedSequence::Ptr sequences_metadata,
                             const LayerInfo layer_info,
                             const std::string& source);

  // allocate DataObject;
  void setMemoryObject(Tensor_Ptr tensor);

  void run_ramulator(DRAMRequest_Ptr dram_request);
  void run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor);

  void addExecutionCache(ExecStatus& exec_status, CacheKey key);

  void addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                         ProcessorType processor_type,
                         DRAMRequestType dram_reqeust_type, long size);

  bool checkExecutionCache(ExecStatus& exec_status, CacheKey key);
  bool checkExecutionCache(CacheKey key);


  void setExecStatus(ExecStatus& exec_status_);

  ExecStatus getExecStatus();
  ExecStatus getHighExecStatus();
  ExecStatus getLowExecStatus();

  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  Cluster_ptr cluster;
  DRAMInterface_Ptr dram_interface;

  StatusBoard status;
  ModelConfig model_config;

  MMapController_Ptr mmap_controller;

  bool perform_execution;

 private:
  ExecStatus high_exec_status;
  ExecStatus low_exec_status;
  ExecStatus exec_status;

  bool use_ramulator;
  std::map<LayerType, LayerTypeDramStats> per_layer_type_dram_stats_;
  bool trace_enabled;
  std::string trace_path;
  std::ofstream trace_stream;
  long trace_event_id;

  Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster);

  void initialize_trace();
  void write_trace_header();

  void execution_ramulator(LayerType layer_type,
                           std::vector<Tensor_Ptr> tensor_list);

  void execution_ideal(LayerType layer_type,
                       std::vector<Tensor_Ptr> tensor_list);
};
}  // namespace llm_system
