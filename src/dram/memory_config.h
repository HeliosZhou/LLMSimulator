#pragma once

namespace llm_system {
class MemoryConfig {
 public:
 MemoryConfig& operator=(const MemoryConfig& rhs) = default;

  // 16GB per cube, total 80GB HBM configuration & 1 cube, 16Gb density, 2 channel, LPDDR5
  MemoryConfig(int num_cube = 5, int num_logic_cube = 5, int num_channel = 32, int num_rank = 2,
               int num_bankgroup = 4, int num_bank = 4, int num_row = 16384,
               int num_col = 32, int granul = 32, int ramulator_addr_levels = 0)
        : num_cube(num_cube),
          num_logic_cube(num_logic_cube),
          num_channel(num_channel),
          num_rank(num_rank),
          num_bankgroup(num_bankgroup),
          num_bank(num_bank),
          num_row(num_row),
          num_col(num_col),
          granul(granul),
          ramulator_addr_levels(ramulator_addr_levels){};

    int num_cube;
    int num_logic_cube;
    
    int num_channel;
    int num_rank;
    int num_bankgroup;
    int num_bank;
    int num_row;
    int num_col;
    int granul;
    int ramulator_addr_levels;
  };

  static MemoryConfig hbm3_80GB = MemoryConfig(
    5,      // num_cube 
    5,      // num_logic_cube
    32,     // num_channel
    2,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  static MemoryConfig hbm3e_192GB = MemoryConfig(
    8,      // num_cube
    8,      // num_logic_cube
    32,     // num_channel
    3,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  // DDR5 configuration: no cube/pseudochannel, one logical 192GB address space.
  static MemoryConfig ddr5_192GB = MemoryConfig(
    1,      // num_cube (没有 pseudochannel)
    1,      // num_logic_cube
    1,      // num_channel
    96,     // num_rank (96 * 2GB logical ranks = 192GB)
    8,      // num_bankgroup
    4,      // num_bank
    65536,  // num_row
    32,     // num_col
    32,     // granul
    6       // Ramulator levels: channel, rank, bankgroup, bank, row, column
  );

  // GDDR6 configuration: no rank level in Ramulator's GDDR6 model.
  static MemoryConfig gddr6_192GB = MemoryConfig(
    1,        // num_cube
    1,        // num_logic_cube
    2,        // num_channel
    1,        // num_rank
    4,        // num_bankgroup
    4,        // num_bank
    6291456,  // num_row
    32,       // num_col
    32,       // granul
    5         // Ramulator levels: channel, bankgroup, bank, row, column
  );

  // HBM4 baseline configuration (Folded Banks / RoMe projected)
  // 64 pseudo channels, 16 banks/channel, 1KB row, 32B access granularity
  // Per cube: 64 channels × 16 banks × 1024 rows × 32 cols × 32B × 8 prefetch = 1Gb density
  // 8 cubes × 32GB = 256GB total, 8 cubes × 2TB/s = 16TB/s total bandwidth
  static MemoryConfig hbm4_baseline = MemoryConfig(
    8,      // num_cube (8 cubes for 256GB system)
    8,      // num_logic_cube
    64,     // num_channel (64 pseudo channels per cube)
    4,      // num_rank (4 ranks for 16-Hi stack)
    1,      // num_bankgroup (HBM4: no bankgroup level, set to 1)
    16,     // num_bank (16 banks per channel)
    1024,   // num_row (1KB row = 1024 rows)
    32,     // num_col
    32,     // granul = 32B access granularity
    6       // Ramulator levels: channel, pseudochannel, rank, bank, row, column
  );

}  // namespace llm_system
