#pragma once

#include <type_traits>

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

struct DramEnergy {
  double kACT_energy_j_ = 0.0;
  double kREAD_energy_j_ = 0.0;
  double kWRITE_energy_j_ = 0.0;
  double kALL_ACT_energy_j_ = 0.0;
  double kALL_READ_energy_j_ = 0.0;
  double kALL_WRITE_energy_j_ = 0.0;
  double kREF_energy_j_ = 0.0;
  double kBACKGROUND_power_nW_ = 0.0;
  double kMAC_energy_j_ = 0.0;
};

// 2017 MICRO FGDRAM, Table 3
// https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
// ACT energy = 0.909 nJ
// HBM2 read/write energy = pre-GSA 1.51pJ/b + post-GSA 1.17pJ/b
//                         + I/O 0.80pJ/b = 3.48pJ/b

// energy per bit of Read(RD) and Write(WR) assumed to be the same
// we multiply this value to the number of count (not number of bits)
// for example, energy for Read operation in HBM2 is
// 3.48 (= 1.51 + 1.17 + 0.80) pJ/b. HBM's granularity is
// 32Byte (256bit), so multiply 256 and divide by 1000 to get nJ
// (= 3.48 * 256 / 1000 = 0.89088 nJ)
//
// FGDRAM does not provide refresh or background/standby energy constants.
// We model REF as one activation-equivalent DRAM event and leave background
// power at 0 nW by default. Replace kBACKGROUND_power_nW_ with a DRAMPower
// or datasheet-derived value when standby energy should be charged.

static DramEnergy gpuEnergy{0.909, 0.891, 0.891, 0, 0, 0, 0.909, 0, 0.46 / 2 / 1000};

static DramEnergy logicEnergy{0.909,     0.464,     0.464,
                              0.909 * 8, 0.464 * 8, 0.464 * 8,
                              0.909,     0,
                              0.46 / 2 / 1000};  // X4

static DramEnergy pimBankgroupEnergy{0.909,     0.686,     0.686,
                                     0.909 * 8, 0.686 * 8, 0.686 * 8,
                                     0.909,     0,
                                     0.46 / 2 / 1000};  // X4

static DramEnergy pimEnergy{0.909,      0.187,      0.187,
                            0.909 * 32, 0.187 * 32, 0.187 * 32,
                            0.909,      0,
                            0.46 / 2 / 1000};  // X16

}  // namespace llm_system
