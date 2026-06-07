#include "module/module_graph.h"

#include <algorithm>
#include <cstdint>

#include "drampower/hbm3e_adapter.h"

namespace llm_system {
namespace {
constexpr double kNanojoulesPerJoule = 1.0e9;

struct LPDDR5PowerSpec {
  double vdd = 1.2;
  double idd0 = 56.25e-3;
  double idd2n = 33.75e-3;
  double idd3n = 35.0e-3;
  double idd4r = 157.5e-3;
  double idd4w = 135.0e-3;
  double idd5 = 118.0e-3;
  double ibeta = 56.25e-3;
  double tck_ns = 0.7692307692307693;
  double wck_to_ck = 2.0;
  double tras_cycles = 10.0;
  double trp_cycles = 10.0;
  double trfc_cycles = 25.0;
  double burst_length = 8.0;
  double data_rate = 2.0;
  double banks = 16.0;
  double bw_power_fact_rho = 0.5;
  double command_parallelism = 128.0;
};

Ramulator::DRAMPower::CommandCounters to_drampower_counters(
    const ExecStatus& exec_status) {
  return {
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.act_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.read_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.write_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.all_act_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.all_read_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.all_write_count)),
      static_cast<std::uint64_t>(std::max<counter_t>(0, exec_status.ref_count)),
  };
}

double seconds_from_ns(double ns) { return std::max(0.0, ns) * 1.0e-9; }

double seconds_from_cycles(double cycles, double tck_ns) {
  return std::max(0.0, cycles) * std::max(0.0, tck_ns) * 1.0e-9;
}

double safe_positive(double value) { return std::max(0.0, value); }

double lpddr5_act_energy_nj(const LPDDR5PowerSpec& spec,
                            std::uint64_t count) {
  const double tras = seconds_from_cycles(spec.tras_cycles, spec.tck_ns);
  const double trp = seconds_from_cycles(spec.trp_cycles, spec.tck_ns);
  const double i_theta =
      ((spec.idd0 * (trp + tras)) - (spec.ibeta * trp)) / tras;
  const double per_command_nj =
      spec.vdd * safe_positive(i_theta - spec.idd3n) * tras *
      kNanojoulesPerJoule;
  return per_command_nj * spec.command_parallelism *
         static_cast<double>(count);
}

double lpddr5_read_write_energy_nj(const LPDDR5PowerSpec& spec,
                                   std::uint64_t count, double idd4) {
  const double t_wck_ns = spec.tck_ns / spec.wck_to_ck;
  const double burst_time =
      seconds_from_cycles(spec.burst_length / spec.data_rate, t_wck_ns);
  const double i_rho =
      spec.bw_power_fact_rho * (spec.idd3n - spec.idd2n) + spec.idd2n;
  const double i_2 = spec.idd3n + (spec.idd3n - i_rho);
  const double per_command_nj =
      spec.vdd * safe_positive(idd4 - i_2) * burst_time *
      kNanojoulesPerJoule;
  return per_command_nj * spec.command_parallelism *
         static_cast<double>(count);
}

double lpddr5_ref_energy_nj(const LPDDR5PowerSpec& spec,
                            std::uint64_t count) {
  const double t_rfc = seconds_from_cycles(spec.trfc_cycles, spec.tck_ns);
  const double i_rho =
      spec.bw_power_fact_rho * (spec.idd3n - spec.idd2n) + spec.idd2n;
  const double approx_idd3n = i_rho + spec.banks * (spec.idd3n - i_rho);
  const double per_command_nj =
      (1.0 / spec.banks) * spec.vdd *
      safe_positive(spec.idd5 - approx_idd3n) * t_rfc *
      kNanojoulesPerJoule;
  return per_command_nj * spec.command_parallelism *
         static_cast<double>(count);
}

double lpddr5_background_energy_nj(const LPDDR5PowerSpec& spec,
                                   double background_time_ns) {
  return spec.vdd * spec.idd2n * seconds_from_ns(background_time_ns) *
         kNanojoulesPerJoule * spec.command_parallelism;
}

Ramulator::DRAMPower::EnergyBreakdown calculate_lpddr5_energy(
    const Ramulator::DRAMPower::CommandCounters& counters,
    double background_time_ns) {
  static const LPDDR5PowerSpec spec;
  Ramulator::DRAMPower::EnergyBreakdown energy;
  energy.act_nj = lpddr5_act_energy_nj(spec, counters.act);
  energy.read_nj = lpddr5_read_write_energy_nj(spec, counters.read, spec.idd4r);
  energy.write_nj =
      lpddr5_read_write_energy_nj(spec, counters.write, spec.idd4w);
  energy.all_act_nj = lpddr5_act_energy_nj(spec, counters.all_act);
  energy.all_read_nj =
      lpddr5_read_write_energy_nj(spec, counters.all_read, spec.idd4r);
  energy.all_write_nj =
      lpddr5_read_write_energy_nj(spec, counters.all_write, spec.idd4w);
  energy.ref_nj = lpddr5_ref_energy_nj(spec, counters.ref);
  energy.background_nj = lpddr5_background_energy_nj(spec, background_time_ns);
  energy.total_nj = energy.act_nj + energy.read_nj + energy.write_nj +
                    energy.all_act_nj + energy.all_read_nj +
                    energy.all_write_nj + energy.ref_nj +
                    energy.background_nj;
  return energy;
}

}  // namespace

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         Tensor::Ptr input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = false;
  stamped = false;
};

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         TensorVec input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input_vec(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = true;
  stamped = false;
};

bool ModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  if (module == nullptr || !module->execution()) {
    return true;
  } else if (check_ready() == true) {
    if (isTensorVec) {
      module->forward(input_vec, sequences_metadata);
    } else {
      module->forward(input, sequences_metadata);
    }
    // input->unset();
    return true;
  } else {
    return false;
  }
};

void ModuleGraph::set_dependency() { set_dependency_tensor(); }

bool ModuleGraph::checkListReady(TensorVec tensor_list) {
  for (Tensor::Ptr tensor : dependency_tensor_list) {
    if (tensor->ready == false) {
      return false;
    }
  }
  return true;
}

bool ModuleGraph::check_ready() {
  if (module->sync) {
    if (!checkListReady(dependency_tensor_list)) {
      return false;
    }
    // all operations are doned, we have to sync the devices
    sync_devices();
    return true;
  } else {
    if (input && input->ready) {
      return true;
    } else if (input_vec.size() != 0) {
      if (checkListReady(input_vec)) {
        return true;
      }
    }
  }
  return false;
}

void ModuleGraph::sync_devices() {
  // not yet synced
  if (input) {
    if (!input->timeboard_synced) {
      Device::Ptr device;
      time_ns time = 0;
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        device = tensor->get_device();
        time_ns device_time = device->get_time();
        time = std::max(time, device_time);
      }
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        tensor->timeboard_synced = true;
        device = tensor->get_device();
        device->set_time(time);
      }
      input->timeboard_synced = false;
    } else {
      input->timeboard_synced = false;
    }
  } else {
    fail("Module cannot be synced when it's inputs are TensorVector");
  }
}

void ModuleGraph::set_dependency_tensor() {
  // only when inputs are one Tensor pointer
  if (input && module && module->sync) {
    module->set_dependency_tensor(dependency_tensor_list, input);
  }
}

void ModuleGraph::print_graph() {
  if (module != nullptr) {
    for (int i = 0; i < module_level; i++) {
      std::cout << "\t";
    }
    std::cout << module->name << std::endl;
  }
}

TopModuleGraph::TopModuleGraph(StatusBoard& status)
    : status(status), module_graph(){};

void TopModuleGraph::push_module_graph(Module::Ptr module, Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::push_module_graph(Module::Ptr module, TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::set_dependency() {
  for (auto module : module_graph) {
    module->set_dependency();
  }
  restart_graph();
}

void TopModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  for (; current_module != module_graph.end(); current_module++) {
    set_stamp();
    // if execution is blocked because of sync
    if (!(*current_module)->run(sequences_metadata)) {
      break;
    }
  }
}

// deprecated
void TopModuleGraph::push_stamp() {
  fail("Cannot use push_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_pop()) {
    timeboard.push_timestamp(status, module_graph->get_name());
  }
}

// deprecated
void TopModuleGraph::pop_stamp() {
  fail("Cannot use pop_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (module_graph->is_pop()) {
    timeboard.pop_timestamp(status);
  }
}

void TopModuleGraph::set_stamp() {
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_stamped()) {
    if (module_graph->is_pop()) {
      set_pop_status();
      timeboard.pop_timestamp(status);
    } else {
      set_push_status();
      timeboard.push_timestamp(status, module_graph->get_name());
    }
    module_graph->set_stamped();
  }
}

void TopModuleGraph::set_push_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;

    status.device_time = std::max(status.device_time,
                                  std::max(status.low_time, status.high_time));
    status.low_time = status.device_time;
    status.high_time = status.device_time;
    status.parallel_execution = false;
    //

  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
      if (status.tensor->isPerformHigh()) {
        status.device_time = status.high_time;
      } else {
        status.device_time = status.low_time;
      }
    } else {
      status.device_time = std::max(
          status.device_time, std::max(status.low_time, status.high_time));
      status.low_time = status.device_time;
      status.high_time = status.device_time;
      status.parallel_execution = false;
    }
  }
}

void TopModuleGraph::set_pop_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;
  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
    } else {
      status.parallel_execution = false;
    }
  }

  ExecStatus exec_status = device->getExecStatus();

  if (status.parallel_execution) {
    if (exec_status.processor_type == ProcessorType::LOGIC ||
        exec_status.processor_type == ProcessorType::PIM) {
      status.low_time += exec_status.total_duration;
      status.device_time = status.low_time;
    } else if (exec_status.processor_type == ProcessorType::GPU) {
      status.high_time += exec_status.total_duration;
      status.device_time = status.high_time;
    }
  } else {
    status.device_time += exec_status.total_duration;
    // status.high_time = status.device_time;
    // status.low_time = status.device_time;
  }
  
  if (exec_status.processor_type == ProcessorType::PIM ||
      exec_status.processor_type == ProcessorType::LOGIC ||
      exec_status.processor_type == ProcessorType::GPU) {
    int processor_type = (int)exec_status.processor_type;
    status.act_energy +=
        exec_status.act_count * dram_powers[processor_type].kACT_energy_j_;
    status.read_energy +=
        exec_status.read_count * dram_powers[processor_type].kREAD_energy_j_;
    status.write_energy +=
        exec_status.write_count * dram_powers[processor_type].kWRITE_energy_j_;

    status.all_act_energy += exec_status.all_act_count *
                             dram_powers[processor_type].kALL_ACT_energy_j_;
    status.all_read_energy += exec_status.all_read_count *
                              dram_powers[processor_type].kALL_READ_energy_j_;
    status.all_write_energy += exec_status.all_write_count *
                               dram_powers[processor_type].kALL_WRITE_energy_j_;

    status.ref_energy +=
        exec_status.ref_count * dram_powers[processor_type].kREF_energy_j_;
    const time_ns background_time =
        exec_status.background_time > 0 ? exec_status.background_time
                                        : exec_status.total_duration;
    status.background_time += background_time;
    status.background_energy += background_time *
                                dram_powers[processor_type].kBACKGROUND_power_nW_ *
                                1e-9;

    if (device->config.use_drampower) {
      static const Ramulator::DRAMPower::HBM3EAdapter hbm3e_drampower;
      const auto counters = to_drampower_counters(exec_status);
      const auto drampower_energy =
          device->config.dram_power_model == "lpddr5"
              ? calculate_lpddr5_energy(counters, background_time)
              : hbm3e_drampower.calculate(counters, background_time);
      status.drampower_act_energy += drampower_energy.act_nj;
      status.drampower_read_energy += drampower_energy.read_nj;
      status.drampower_write_energy += drampower_energy.write_nj;
      status.drampower_all_act_energy += drampower_energy.all_act_nj;
      status.drampower_all_read_energy += drampower_energy.all_read_nj;
      status.drampower_all_write_energy += drampower_energy.all_write_nj;
      status.drampower_ref_energy += drampower_energy.ref_nj;
      status.drampower_background_energy += drampower_energy.background_nj;
      status.drampower_total_energy += drampower_energy.total_nj;
    }

    status.mac_energy +=
        exec_status.flops * dram_powers[processor_type].kMAC_energy_j_;
    ;  // 2flops per operation, energy per operation, pJ to nJ

    // Accumulate Ramulator detailed counters
    status.act_count += exec_status.act_count;
    status.read_count += exec_status.read_count;
    status.write_count += exec_status.write_count;
    status.all_act_count += exec_status.all_act_count;
    status.all_read_count += exec_status.all_read_count;
    status.all_write_count += exec_status.all_write_count;
    status.ref_count += exec_status.ref_count;
    status.memory_duration += exec_status.memory_duration;
  }

  // if (!exec_status.parallel_execution) {
  //   // status.device_time = std::max(status.device_time,
  //   //                               std::max(status.low_time,
  //   //                               status.high_time));
  //   // status.low_time = status.device_time;
  //   // status.high_time = status.device_time;
  //   status.parallel_execution = false;
  // } else {
  //   status.parallel_execution = true;
  // }

  // if (exec_status.processor_type == ProcessorType::LOGIC ||
  //     exec_status.processor_type == ProcessorType::PIM) {
  //   status.low_time += exec_status.total_duration;
  //   status.device_time = status.low_time;
  // } else if (exec_status.processor_type == ProcessorType::GPU) {
  //   status.high_time += exec_status.total_duration;
  //   status.device_time = status.high_time;
  // } else {
  //   status.low_time = status.device_time;
  //   status.high_time = status.device_time;
  // }

  status.compute_util = exec_status.compute_util;
  status.memory_util = exec_status.memory_util;
  status.processor_type = exec_status.processor_type;

  status.flops += exec_status.flops;
  status.memory_size += exec_status.memory_size;

  status.opb = exec_status.opb;
}

void TopModuleGraph::print_graph() {
  std::cout << "Print graph" << std::endl;
  for (auto module_graph_ : module_graph) {
    module_graph_->print_graph();
  }
}

void TopModuleGraph::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  if(dram_powers.size() == 0){
    for (int i = 0; i < (int)ProcessorType::MAX; i++) {
      DramEnergy temp;
      dram_powers.push_back(temp);
    }
  }
  dram_powers[ProcessorType] = dramEnergy;
}

std::vector<energy_nJ> TopModuleGraph::getDeviceEnergy(){
  std::vector<energy_nJ> device_energy {status.act_energy, status.read_energy, status.write_energy, 
                            status.all_act_energy, status.all_read_energy, status.all_write_energy,
                            status.mac_energy, status.act_energy + status.read_energy + status.write_energy + 
                            status.all_act_energy + status.all_read_energy + status.all_write_energy +
                            status.ref_energy + status.background_energy + status.mac_energy,
                            status.ref_energy, status.background_energy, status.background_time,
                            status.drampower_act_energy,
                            status.drampower_read_energy,
                            status.drampower_write_energy,
                            status.drampower_all_act_energy,
                            status.drampower_all_read_energy,
                            status.drampower_all_write_energy,
                            status.drampower_ref_energy,
                            status.drampower_background_energy,
                            status.drampower_total_energy};
  return device_energy;
}

void TopModuleGraph::restart_graph() {
  current_module = module_graph.begin();
  assertTrue(current_module != module_graph.end(),
             "No module in TopModuleGraph");
  (*current_module)->set_ready();

  for (auto module_graph_ : module_graph) {
    module_graph_->unset_tensor();
  }
};

}  // namespace llm_system
