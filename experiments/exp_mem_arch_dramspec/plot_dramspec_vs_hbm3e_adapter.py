#!/usr/bin/env python3
"""Compare reorder-on HBM3EAdapter and DRAMSpec-calibrated energy breakdowns."""

from __future__ import annotations

import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
BASELINE_CSV = PROJECT_DIR / "experiments" / "exp_mem_arch" / "data" / "energy_breakdown_ramulator_on_drampower_ref.csv"
DRAMSPEC_CSV = EXP_DIR / "data" / "summary_dramspec_hbm3e_like.csv"
OUT_CSV = EXP_DIR / "data" / "dramspec_vs_hbm3e_adapter_reorder_on.csv"
PLOT_DIR = EXP_DIR / "plots"
OUT_ENERGY_PNG = PLOT_DIR / "figure_dramspec_vs_hbm3e_adapter_reorder_on_energy.png"
OUT_SHARE_PNG = PLOT_DIR / "figure_dramspec_vs_hbm3e_adapter_reorder_on_share.png"
OUT_DRAM_ONLY_ENERGY_PNG = PLOT_DIR / "figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_energy.png"
OUT_DRAM_ONLY_SHARE_PNG = PLOT_DIR / "figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_share.png"
OLD_COMBINED_PNG = PLOT_DIR / "figure_dramspec_vs_hbm3e_adapter_reorder_on.png"
PPT_DRAM_HEATMAP_SIZE = (9.15, 5.15)
PPT_TOTAL_HEATMAP_SIZE = (10.7, 5.15)
PLOT_DPI = 220

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]
DRAM_COMPONENTS = ["ACT", "READ", "WRITE", "REF", "BG"]
COMPONENTS = DRAM_COMPONENTS + ["MAC"]
ENERGY_KEYS = {
    "ACT": "act_J_step",
    "READ": "read_J_step",
    "WRITE": "write_J_step",
    "REF": "ref_J_step",
    "BG": "background_J_step",
    "MAC": "mac_J_step",
}
SHARE_KEYS = {
    "ACT": "act_pct_total_plus_mac",
    "READ": "read_pct_total_plus_mac",
    "WRITE": "write_pct_total_plus_mac",
    "REF": "ref_pct_total_plus_mac",
    "BG": "background_pct_total_plus_mac",
    "MAC": "mac_pct_total_plus_mac",
}
DRAM_SHARE_KEYS = {
    "ACT": "act_pct_dram",
    "READ": "read_pct_dram",
    "WRITE": "write_pct_dram",
    "REF": "ref_pct_dram",
    "BG": "background_pct_dram",
}


def read_hbm3e_adapter_rows() -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with BASELINE_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row["reorder"] != "on" or row["ramulator"] != "on":
                continue
            batch = int(row["batch_per_gpu"])
            seq = int(row["seq_len"])
            rows[(batch, seq)] = {key: float(value) for key, value in row.items() if value and key not in {"source_result", "reorder", "ramulator"}}
    return rows


def read_dramspec_rows() -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with DRAMSPEC_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row["reorder"] != "on" or row["ramulator"] != "on":
                continue
            batch = int(row["batch_size"])
            seq = int(row["seq_len"])
            out = {
                "latency_ms": float(row["latency_ms"]),
                "tokens_per_step": float(row["sim_batchsize"]),
                "act_J_step": float(row["drampower_act_energy_nJ"]) / 1.0e9,
                "read_J_step": float(row["drampower_read_energy_nJ"]) / 1.0e9,
                "write_J_step": float(row["drampower_write_energy_nJ"]) / 1.0e9,
                "ref_J_step": float(row["drampower_ref_energy_nJ"]) / 1.0e9,
                "background_J_step": float(row["drampower_background_energy_nJ"]) / 1.0e9,
                "mac_J_step": float(row["mac_energy_nJ"]) / 1.0e9,
            }
            dram_total = sum(out[ENERGY_KEYS[c]] for c in COMPONENTS if c != "MAC")
            total_plus_mac = dram_total + out["mac_J_step"]
            out["dram_total_J_step"] = dram_total
            out["total_plus_mac_J_step"] = total_plus_mac
            for component in COMPONENTS:
                out[SHARE_KEYS[component]] = 0.0 if total_plus_mac <= 0 else out[ENERGY_KEYS[component]] / total_plus_mac * 100.0
            for component in DRAM_COMPONENTS:
                out[DRAM_SHARE_KEYS[component]] = 0.0 if dram_total <= 0 else out[ENERGY_KEYS[component]] / dram_total * 100.0
            rows[(batch, seq)] = out
    return rows


def write_comparison_csv(
    hbm: dict[tuple[int, int], dict[str, float]],
    dramspec: dict[tuple[int, int], dict[str, float]],
) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_per_gpu",
        "seq_len",
        "component",
        "hbm3e_adapter_J_step",
        "dramspec_J_step",
        "dramspec_over_hbm3e_adapter_energy",
        "hbm3e_adapter_pct_total_plus_mac",
        "dramspec_pct_total_plus_mac",
        "dramspec_minus_hbm3e_adapter_pct_points",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for batch in BATCH_PER_GPU:
            for seq in SEQ_LENGTHS:
                hbm_row = hbm[(batch, seq)]
                dram_row = dramspec[(batch, seq)]
                for component in COMPONENTS:
                    hbm_energy = hbm_row[ENERGY_KEYS[component]]
                    dram_energy = dram_row[ENERGY_KEYS[component]]
                    hbm_share = hbm_row[SHARE_KEYS[component]]
                    dram_share = dram_row[SHARE_KEYS[component]]
                    writer.writerow(
                        {
                            "batch_per_gpu": batch,
                            "seq_len": seq,
                            "component": component,
                            "hbm3e_adapter_J_step": hbm_energy,
                            "dramspec_J_step": dram_energy,
                            "dramspec_over_hbm3e_adapter_energy": "" if hbm_energy == 0 else dram_energy / hbm_energy,
                            "hbm3e_adapter_pct_total_plus_mac": hbm_share,
                            "dramspec_pct_total_plus_mac": dram_share,
                            "dramspec_minus_hbm3e_adapter_pct_points": dram_share - hbm_share,
                        }
                    )


def _format_energy(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_heatmap(
    hbm: dict[tuple[int, int], dict[str, float]],
    dramspec: dict[tuple[int, int], dict[str, float]],
    *,
    kind: str,
    output: Path,
    components: list[str],
    share_keys: dict[str, str],
    figsize: tuple[float, float],
    share_note: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.linewidth": 0.75,
        }
    )

    if kind not in {"energy", "share"}:
        raise ValueError(f"Unknown plot kind: {kind}")
    row_defs = [("HBM3EAdapter", hbm), ("DRAMSpec", dramspec)]
    fig, axes = plt.subplots(
        len(row_defs),
        len(components),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    all_energy_values = [
        dataset[(batch, seq)][ENERGY_KEYS[component]]
        for _, dataset in row_defs
        for component in components
        for batch in BATCH_PER_GPU
        for seq in SEQ_LENGTHS
    ]
    positive = [value for value in all_energy_values if value > 0]
    energy_norm = LogNorm(vmin=max(min(positive), 1e-3), vmax=max(positive))
    share_norm = Normalize(vmin=0, vmax=100)

    for row_idx, (model, dataset) in enumerate(row_defs):
        for col_idx, component in enumerate(components):
            ax = axes[row_idx][col_idx]
            key = ENERGY_KEYS[component] if kind == "energy" else share_keys[component]
            matrix = np.array(
                [
                    [dataset[(batch, seq)][key] for seq in SEQ_LENGTHS]
                    for batch in BATCH_PER_GPU
                ],
                dtype=float,
            )
            if kind == "energy":
                norm = energy_norm
                cmap = "YlOrRd"
                formatter = _format_energy
            else:
                norm = share_norm
                cmap = "YlGnBu"
                formatter = lambda value: f"{value:.1f}%"
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(BATCH_PER_GPU)):
                for j in range(len(SEQ_LENGTHS)):
                    value = matrix[i, j]
                    norm_value = norm(max(value, norm.vmin)) if kind == "energy" else norm(value)
                    ax.text(
                        j,
                        i,
                        formatter(value),
                        ha="center",
                        va="center",
                        fontsize=6.7,
                        color="white" if norm_value > 0.58 else "#111111",
                    )
            if row_idx == 0:
                ax.set_title(component, fontsize=9.3)
            if col_idx == 0:
                ax.set_ylabel(f"{model}\nB/GPU", fontsize=7.8)
            ax.set_xticks(range(len(SEQ_LENGTHS)))
            ax.set_xticklabels([str(seq) for seq in SEQ_LENGTHS], fontsize=7)
            ax.set_yticks(range(len(BATCH_PER_GPU)))
            ax.set_yticklabels([str(batch) for batch in BATCH_PER_GPU], fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(SEQ_LENGTHS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(BATCH_PER_GPU), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.7)
            ax.tick_params(which="minor", bottom=False, left=False)

    for ax in axes[-1]:
        ax.set_xlabel("Seq", fontsize=7.8)

    is_dram_only = components == DRAM_COMPONENTS
    if kind == "energy":
        if is_dram_only:
            title = "DRAM command energy (J/step): HBM3EAdapter vs DRAMSpec"
        else:
            title = "Component energy including MAC (J/step): HBM3EAdapter vs DRAMSpec"
        note = "Ramulator=on; cell = J/step"
    else:
        if is_dram_only:
            title = "DRAM command energy share: HBM3EAdapter vs DRAMSpec"
        else:
            title = "Energy share including MAC: HBM3EAdapter vs DRAMSpec"
        note = f"Ramulator=on; cell = share of {share_note}"
    fig.suptitle(title, fontsize=10.2, y=0.985)
    fig.text(
        0.976,
        0.965,
        note,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.058, right=0.992, top=0.875, bottom=0.105, hspace=0.24, wspace=0.065)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    hbm = read_hbm3e_adapter_rows()
    dramspec = read_dramspec_rows()
    expected = {(batch, seq) for batch in BATCH_PER_GPU for seq in SEQ_LENGTHS}
    missing_hbm = sorted(expected - set(hbm))
    missing_dramspec = sorted(expected - set(dramspec))
    if missing_hbm or missing_dramspec:
        raise SystemExit(f"Missing rows: hbm={missing_hbm}, dramspec={missing_dramspec}")
    write_comparison_csv(hbm, dramspec)
    plot_heatmap(
        hbm,
        dramspec,
        kind="energy",
        output=OUT_DRAM_ONLY_ENERGY_PNG,
        components=DRAM_COMPONENTS,
        share_keys=DRAM_SHARE_KEYS,
        figsize=PPT_DRAM_HEATMAP_SIZE,
        share_note="DRAM",
    )
    plot_heatmap(
        hbm,
        dramspec,
        kind="share",
        output=OUT_DRAM_ONLY_SHARE_PNG,
        components=DRAM_COMPONENTS,
        share_keys=DRAM_SHARE_KEYS,
        figsize=PPT_DRAM_HEATMAP_SIZE,
        share_note="DRAM",
    )
    plot_heatmap(
        hbm,
        dramspec,
        kind="energy",
        output=OUT_ENERGY_PNG,
        components=COMPONENTS,
        share_keys=SHARE_KEYS,
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        share_note="DRAM+MAC",
    )
    plot_heatmap(
        hbm,
        dramspec,
        kind="share",
        output=OUT_SHARE_PNG,
        components=COMPONENTS,
        share_keys=SHARE_KEYS,
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        share_note="DRAM+MAC",
    )
    if OLD_COMBINED_PNG.exists():
        OLD_COMBINED_PNG.unlink()
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
