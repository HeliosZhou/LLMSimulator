# DRAM Energy Model Notes

LLMSimulator currently reports DRAM energy with a fixed event-energy model:

```text
energy = command_count * fixed_energy_per_command
```

Ramulator ON changes the command counters and memory timing, but the CSV energy
columns are still computed by LLMSimulator.

## FGDRAM Constants

The default GPU constants in `src/dram/power.h` are based on MICRO 2017
Fine-Grained DRAM Table 3 HBM2/QB-HBM values:

```text
ACT = 909 pJ = 0.909 nJ
READ/WRITE = (1.51 + 1.17 + 0.80) pJ/bit * 32 B * 8 / 1000
           = 0.89088 nJ
```

The paper does not provide REF energy, background power, standby current, or an
IDD-style model. It only lists row activation, data movement, and I/O access
energy. Therefore `ref_energy` and `background_energy` are model extensions, not
direct FGDRAM paper values.

## REF Energy

`ref_energy` is estimated as:

```text
ref_energy = ref_count * kREF_energy_j_
```

The default `kREF_energy_j_` is activation-equivalent (`0.909 nJ` before device
organization scaling). Replace it with a datasheet or DRAMPower-derived value
for quantitative refresh analysis.

## Background Energy

`background_time` is the time base for DRAM background energy. It is accumulated
from each executed layer/request `total_duration`, not from `memory_duration`.
This keeps it distinct from the DRAM service time:

```text
memory_duration    = modeled DRAM access service time
background_time    = DRAM powered time charged for background energy
background_energy  = background_time_ns * kBACKGROUND_power_nW_ * 1e-9
```

The default `kBACKGROUND_power_nW_` is `0`, so background energy remains zero
until a technology-specific background power is supplied.
