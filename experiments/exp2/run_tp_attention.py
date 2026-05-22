#!/usr/bin/env python3
"""Figure 8: MLA attention latency as TP degree changes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import SimPoint, add_common_args, attention_breakdown_from_csv, run_simulation, write_summary_csv  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

BATCHES = [32, 64, 128]
TP_DEGREES = [1, 2, 4, 8]
SEQ_LEN = 4096


def result_name(batch: int, tp: int, absorb: str) -> str:
    return f"result_b{batch}_l{SEQ_LEN}_tp{tp}_absorb_{absorb}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_b*_l*_tp*_absorb_*.csv")):
        parts = path.stem.split("_")
        try:
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            tp = int(next(p[2:] for p in parts if p.startswith("tp")))
            absorb = parts[-1]
        except (StopIteration, ValueError):
            continue
        b = attention_breakdown_from_csv(path)
        rows.append({"batch_size": batch, "seq_len": seq_len, "tp_degree": tp, "absorb": absorb, **b})
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = {(r["batch_size"], r["tp_degree"], r["absorb"]): float(r["total"]) / 1e6 for r in rows}

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"on": ("o-", "w/ reordering"), "off": ("s--", "w/o reordering")}
    for batch in BATCHES:
        for absorb, (style, label) in styles.items():
            y = [data.get((batch, tp, absorb), float("nan")) for tp in TP_DEGREES]
            ax.plot(TP_DEGREES, y, style, label=f"B={batch}, {label}", linewidth=1.8, markersize=5)

    ax.set_yscale("log")
    ax.set_xticks(TP_DEGREES)
    ax.set_xlabel("TP degree")
    ax.set_ylabel("Attention block latency (ms)")
    ax.set_title("Figure 8 style: TP impact on MLA attention, L=4096")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = PLOT_DIR / "figure8_tp_attention.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    batches = [32] if args.quick else BATCHES
    tps = [1, 2] if args.quick else TP_DEGREES

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for batch in batches:
            for tp in tps:
                for absorb_label, use_absorb in [("on", True), ("off", False)]:
                    point = SimPoint(
                        batch_size=batch,
                        seq_len=SEQ_LEN,
                        use_absorb=use_absorb,
                        none_expert_tp=tp,
                        expert_tp=1,
                        num_node=1,
                        num_device=8,
                    )
                    print(f"[Figure 8] B={batch} TP={tp} absorb={absorb_label}")
                    run_simulation(point, DATA_DIR, result_name(batch, tp, absorb_label), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_tp_attention.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No Figure 8 data found.")
        plot(rows)


if __name__ == "__main__":
    main()
