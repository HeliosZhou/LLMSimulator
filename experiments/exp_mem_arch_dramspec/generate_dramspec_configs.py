#!/usr/bin/env python3
"""Generate DRAMSpec-calibrated HBM3E-like LLMSimulator configs."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
DRAMSPEC_DIR = EXP_DIR / "tools" / "DRAMSpec"
DRAMSPEC_BIN = DRAMSPEC_DIR / "build" / "release" / "dramspec"
TECH_INPUT = EXP_DIR / "inputs" / "tech_hbm3e_calibrated_10nm.json"
ARCH_INPUT = EXP_DIR / "inputs" / "arch_hbm3e_like_b200_24gb_8gbps.json"
RAW_DIR = EXP_DIR / "generated" / "dramspec_raw"
RAMULATOR_CONFIG = EXP_DIR / "generated" / "dram_config_HBM3E_DRAMSpec.yaml"
POWER_CONFIG = EXP_DIR / "generated" / "dramspec_hbm3e_like_power.yaml"
SUMMARY_JSON = EXP_DIR / "generated" / "dramspec_hbm3e_like_summary.json"


def run_dramspec() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in DRAMSPEC_DIR.glob("*result_*.json"):
        path.unlink()
    cmd = [
        str(DRAMSPEC_BIN),
        "-t",
        str(TECH_INPUT),
        "-p",
        str(ARCH_INPUT),
    ]
    completed = subprocess.run(
        cmd,
        cwd=DRAMSPEC_DIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (RAW_DIR / "dramspec_stdout.txt").write_text(completed.stdout)
    (RAW_DIR / "dramspec_stderr.txt").write_text(completed.stderr)
    for name in ["timingresult_1.json", "timingnsresult_1.json", "currentresult_1.json"]:
        shutil.copy2(DRAMSPEC_DIR / name, RAW_DIR / name)
    timing = json.loads((RAW_DIR / "timingresult_1.json").read_text())
    timing_ns = json.loads((RAW_DIR / "timingnsresult_1.json").read_text())
    current = json.loads((RAW_DIR / "currentresult_1.json").read_text())
    return timing, timing_ns, current


def cycles_from_ns(ns: float, tck_ns: float) -> int:
    return max(1, int(math.ceil(ns / tck_ns)))


def build_ramulator_config(timing_ns: dict[str, float]) -> dict:
    base = yaml.safe_load((PROJECT_DIR / "dram_config_HBM3E_192GB.yaml").read_text())
    dram = base["MemorySystem"]["DRAM"]
    dram["impl"] = "HBM3"
    dram["org"] = {
        "preset": "HBM3_24Gb_3R",
        "channel": 1,
        "rank": 3,
    }
    dram["timing"] = {
        "rate": 8000,
        "nBL": 2,
        "tCL": float(timing_ns["tcl"]),
        "tRCDRD": float(timing_ns["trcd"]),
        "tRCDWR": float(timing_ns["trcd"]),
        "tRP": float(timing_ns["trp"]),
        "tRAS": float(timing_ns["tras"]),
        "tRC": float(timing_ns["trc"]),
        "tWR": float(timing_ns["twr"]),
        "nRTPS": 8,
        "nRTPL": 12,
        "nCWL": 8,
        "nCCDS": 2,
        "nCCDL": 4,
        "nCCDAB": 6,
        "nCCDSB": 6,
        "nRRDS": 2,
        "nRRDL": 4,
        "nWTRS": 12,
        "nWTRL": 16,
        "nRTW": 3,
        "nFAW": 60,
        "tRFC": float(timing_ns["trfc"]),
        "nRFCSB": 400,
        "tREFI": float(timing_ns["trefI"]),
        "nRREFD": 16,
    }
    return base


def read_technology_value(path: Path, key: str) -> float:
    technology = json.loads(path.read_text())
    return float(technology[key])


def build_power_config(timing_ns: dict[str, float], current: dict[str, float]) -> dict:
    tck_ns = 0.5
    vdd = read_technology_value(TECH_INPUT, "Vdd[V]")
    return {
        "source": {
            "model": "DRAMSpec-calibrated HBM3E-like",
            "technology_input": str(TECH_INPUT.relative_to(EXP_DIR)),
            "architecture_input": str(ARCH_INPUT.relative_to(EXP_DIR)),
            "notes": [
                "Technology parameters use the 10nm HBM3E-like calibration documented in DRAMSPEC_PARAMETERS.md.",
                "Architecture parameters are B200/HBM3E-like assumptions aligned to the current LLMSimulator HBM3_24Gb_3R organization.",
                "This is not a vendor datasheet-level HBM3E memspec.",
            ],
        },
        "power": {
            "vdd": vdd,
            "idd0": float(current["IDD0"]) * 1.0e-3,
            "idd2n": float(current["IDD2n"]) * 1.0e-3,
            "idd3n": float(current["IDD3n"]) * 1.0e-3,
            "idd4r": float(current["IDD4R"]) * 1.0e-3,
            "idd4w": float(current["IDD4W"]) * 1.0e-3,
            "idd5": float(current["IDD5B"]) * 1.0e-3,
            "tck_ns": tck_ns,
            "trcd_cycles": cycles_from_ns(float(timing_ns["trcd"]), tck_ns),
            "tras_cycles": cycles_from_ns(float(timing_ns["tras"]), tck_ns),
            "trp_cycles": cycles_from_ns(float(timing_ns["trp"]), tck_ns),
            "trfc_cycles": cycles_from_ns(float(timing_ns["trfc"]), tck_ns),
            "burst_length_cycles": 2.0,
            "command_parallelism": 128.0,
            "fallback_act_nj": 0.0,
            "fallback_read_nj": 0.0,
            "fallback_write_nj": 0.0,
            "fallback_ref_nj": 0.0,
        },
    }


def main() -> None:
    if not DRAMSPEC_BIN.exists():
        raise SystemExit(f"DRAMSpec binary not found: {DRAMSPEC_BIN}")
    timing, timing_ns, current = run_dramspec()
    RAMULATOR_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    POWER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    ramulator_config = build_ramulator_config(timing_ns)
    power_config = build_power_config(timing_ns, current)
    RAMULATOR_CONFIG.write_text(yaml.safe_dump(ramulator_config, sort_keys=False))
    POWER_CONFIG.write_text(yaml.safe_dump(power_config, sort_keys=False))
    summary = {
        "timing_cycles_from_dramspec": timing,
        "timing_ns_from_dramspec": timing_ns,
        "current_mA_from_dramspec": current,
        "generated_ramulator_config": str(RAMULATOR_CONFIG.relative_to(EXP_DIR)),
        "generated_power_config": str(POWER_CONFIG.relative_to(EXP_DIR)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {RAMULATOR_CONFIG}")
    print(f"Wrote {POWER_CONFIG}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
