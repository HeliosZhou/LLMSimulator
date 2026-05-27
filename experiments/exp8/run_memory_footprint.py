#!/usr/bin/env python3
"""Analytical Figure 4 memory-footprint reconstruction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

TOKENS = 8_000_000
BYTES_PER_PARAM = 2

MODELS = [
    {
        "model": "GPT-3",
        "hidden_dim": 12288,
        "num_layers": 96,
        "num_heads": 96,
        "num_kv_heads": 96,
        "total_params_b": 175.0,
        "activated_params_b": 175.0,
        "kv_per_token_bytes": 4.5 * 1024 * 1024,
    },
    {
        "model": "Llama4-Maverick",
        "hidden_dim": 5120,
        "num_layers": 48,
        "num_heads": 40,
        "num_kv_heads": 8,
        "total_params_b": 400.0,
        "activated_params_b": 17.0,
        "kv_per_token_bytes": 192.0 * 1024,
    },
    {
        "model": "DeepSeek-R1",
        "hidden_dim": 7168,
        "num_layers": 60,
        "num_heads": 128,
        "num_kv_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "total_params_b": 671.0,
        "activated_params_b": 37.0,
        "kv_per_token_bytes": 68.6 * 1024,
    },
]


def attention_params_b(item: dict[str, float | int | str]) -> float:
    hidden = int(item["hidden_dim"])
    layers = int(item["num_layers"])
    num_heads = int(item["num_heads"])
    num_kv_heads = int(item["num_kv_heads"])
    head_dim = hidden // num_heads
    if "kv_lora_rank" in item:
        q_lora = int(item["q_lora_rank"])
        kv_lora = int(item["kv_lora_rank"])
        rope = int(item["qk_rope_head_dim"])
        per_layer = (
            hidden * q_lora
            + q_lora * num_heads * head_dim
            + hidden * (kv_lora + rope)
            + kv_lora * num_heads * head_dim * 2
            + hidden * hidden
        )
    else:
        per_layer = hidden * hidden + 2 * hidden * num_kv_heads * head_dim + hidden * hidden
    return per_layer * layers / 1e9


def build_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for item in MODELS:
        attn_params_b = attention_params_b(item)
        total_params_b = float(item["total_params_b"])
        activated_params_b = float(item["activated_params_b"])
        activated_attn_params_b = min(attn_params_b, activated_params_b)
        activated_ffn_moe_params_b = max(0.0, activated_params_b - activated_attn_params_b)
        attention_weight_gb = attn_params_b * BYTES_PER_PARAM
        ffn_moe_weight_gb = max(0.0, total_params_b - attn_params_b) * BYTES_PER_PARAM
        activated_attention_gb = activated_attn_params_b * BYTES_PER_PARAM
        activated_ffn_moe_gb = activated_ffn_moe_params_b * BYTES_PER_PARAM
        total_weight_gb = attention_weight_gb + ffn_moe_weight_gb
        activated_weight_gb = activated_attention_gb + activated_ffn_moe_gb
        kv_total_gb = float(item["kv_per_token_bytes"]) * TOKENS / 1e9
        rows.append(
            {
                "model": str(item["model"]),
                "total_params_b": total_params_b,
                "activated_params_b": activated_params_b,
                "attention_weight_gb": attention_weight_gb,
                "ffn_moe_weight_gb": ffn_moe_weight_gb,
                "activated_attention_weight_gb": activated_attention_gb,
                "activated_ffn_moe_weight_gb": activated_ffn_moe_gb,
                "activated_weight_gb": activated_weight_gb,
                "total_weight_gb": total_weight_gb,
                "kv_per_token_kb": float(item["kv_per_token_bytes"]) / 1024.0,
                "kv_for_8m_tokens_gb": kv_total_gb,
                "total_memory_gb": total_weight_gb + kv_total_gb,
            }
        )
    return rows


def write_summary(rows: list[dict[str, float | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "summary_memory_footprint.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [str(r["model"]) for r in rows]
    y = np.arange(len(labels))
    height = 0.32

    act_attn = [float(r["activated_attention_weight_gb"]) for r in rows]
    act_ffn = [float(r["activated_ffn_moe_weight_gb"]) for r in rows]
    total_attn = [float(r["attention_weight_gb"]) for r in rows]
    total_ffn = [float(r["ffn_moe_weight_gb"]) for r in rows]
    kv = [float(r["kv_for_8m_tokens_gb"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y - height / 2, act_attn, height, label="Attention weight", color="#d7191c", edgecolor="black", linewidth=0.4)
    ax.barh(y - height / 2, act_ffn, height, left=act_attn, label="FFN/MoE weight", color="#2c7bb6", edgecolor="black", linewidth=0.4)
    ax.barh(y + height / 2, total_attn, height, color="#d7191c", edgecolor="black", linewidth=0.4)
    total_attn_ffn = [a + b for a, b in zip(total_attn, total_ffn)]
    ax.barh(y + height / 2, total_ffn, height, left=total_attn, color="#2c7bb6", edgecolor="black", linewidth=0.4)
    ax.barh(y + height / 2, kv, height, left=total_attn_ffn, label="KV cache for 8M tokens", color="#f2f2f2", edgecolor="black", linewidth=0.4)

    for i, row in enumerate(rows):
        ax.annotate(
            f"{float(row['kv_per_token_kb']):.1f}KB/token",
            (total_attn_ffn[i] + kv[i], y[i] + height / 2),
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Memory footprint (GB)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title("Figure 4 style memory footprint comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "figure4_memory_footprint.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    rows = build_rows()
    write_summary(rows)
    if args.plot or args.run or args.all:
        plot(rows)


if __name__ == "__main__":
    main()
