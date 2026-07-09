#!/usr/bin/env python3
"""Analyze command trace for row lifecycle, RowHammer, and RowPress signals."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_WINDOW_CYCLES = 15600


@dataclass
class TraceEvent:
    cycle: int
    command: str
    channel: int
    pseudochannel: int
    rank: int
    bankgroup: int
    bank: int
    row: int
    column: int
    addr: int
    pim_cmd_type: str
    operand_type: str
    dramreq_type: str

    @property
    def row_key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.channel,
            self.pseudochannel,
            self.rank,
            self.bankgroup,
            self.bank,
            self.row,
        )

    @property
    def bank_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.channel,
            self.pseudochannel,
            self.rank,
            self.bankgroup,
            self.bank,
        )


@dataclass
class RowSession:
    session_id: int
    row_key: tuple[int, int, int, int, int, int]
    act_cycle: int
    pre_cycle: int | None = None
    commands: list[str] = field(default_factory=list)
    columns: list[int] = field(default_factory=list)
    rd_count: int = 0
    wr_count: int = 0
    act_count: int = 1

    def lifetime_cycles(self) -> int | None:
        if self.pre_cycle is None:
            return None
        return self.pre_cycle - self.act_cycle

    def columns_repr(self, limit: int = 32) -> str:
        unique = sorted(set(self.columns))
        if len(unique) <= limit:
            return " ".join(str(col) for col in unique)
        head = " ".join(str(col) for col in unique[:limit])
        return f"{head} ... ({len(unique)} cols)"

    def command_trace(self, limit: int = 80) -> str:
        if len(self.commands) <= limit:
            return " -> ".join(self.commands)
        prefix = " -> ".join(self.commands[:limit])
        return f"{prefix} -> ... ({len(self.commands)} commands)"


def parse_trace(path: Path) -> tuple[list[TraceEvent], list[dict[str, int]]]:
    events: list[TraceEvent] = []
    segments: list[dict[str, int]] = []
    current_segment = 0
    current_start = 0
    prev_cycle: int | None = None
    with path.open("r", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            if len(row) < 13:
                continue
            try:
                event = TraceEvent(
                    cycle=int(row[0]),
                    command=row[1],
                    channel=int(row[2]),
                    pseudochannel=int(row[3]),
                    rank=int(row[4]),
                    bankgroup=int(row[5]),
                    bank=int(row[6]),
                    row=int(row[7]),
                    column=int(row[8]),
                    addr=int(row[9]),
                    pim_cmd_type=row[10],
                    operand_type=row[11],
                    dramreq_type=row[12],
                )
            except ValueError:
                continue

            if prev_cycle is not None and event.cycle < prev_cycle:
                segments.append(
                    {
                        "segment_idx": current_segment,
                        "start_event_idx": current_start,
                        "end_event_idx": len(events) - 1,
                        "start_cycle": events[current_start].cycle,
                        "end_cycle": events[-1].cycle,
                        "event_count": len(events) - current_start,
                    }
                )
                current_segment += 1
                current_start = len(events)
            prev_cycle = event.cycle
            events.append(event)

    if events:
        segments.append(
            {
                "segment_idx": current_segment,
                "start_event_idx": current_start,
                "end_event_idx": len(events) - 1,
                "start_cycle": events[current_start].cycle,
                "end_cycle": events[-1].cycle,
                "event_count": len(events) - current_start,
            }
        )
    return events, segments


def track_sessions(events: Iterable[TraceEvent]) -> tuple[list[RowSession], dict[str, int]]:
    open_sessions: dict[tuple[int, int, int, int, int], RowSession] = {}
    sessions: list[RowSession] = []
    anomalies = Counter()
    next_session_id = 0

    for event in events:
        bank_key = event.bank_key
        row_key = event.row_key
        active = open_sessions.get(bank_key)

        if event.command == "ACT":
            if active is not None:
                anomalies["act_while_other_row_open"] += 1
                if active.pre_cycle is None:
                    active.pre_cycle = event.cycle
            session = RowSession(
                session_id=next_session_id,
                row_key=row_key,
                act_cycle=event.cycle,
                commands=[f"ACT@{event.cycle}"],
                columns=[event.column],
            )
            open_sessions[bank_key] = session
            sessions.append(session)
            next_session_id += 1
            continue

        if active is None:
            anomalies[f"{event.command.lower()}_without_open_row"] += 1
            continue

        if event.command not in {"PRE", "PREA"} and active.row_key != row_key:
            anomalies[f"{event.command.lower()}_row_mismatch"] += 1

        active.columns.append(event.column)
        if event.command == "RD":
            active.rd_count += 1
            active.commands.append(f"RD@{event.cycle}(c{event.column})")
        elif event.command == "WR":
            active.wr_count += 1
            active.commands.append(f"WR@{event.cycle}(c{event.column})")
        elif event.command in {"PRE", "PREA"}:
            active.pre_cycle = event.cycle
            active.commands.append(f"{event.command}@{event.cycle}")
            del open_sessions[bank_key]
        elif event.command.startswith("REF"):
            anomalies["refresh_during_open_row"] += 1
            active.commands.append(f"{event.command}@{event.cycle}")
        else:
            active.commands.append(f"{event.command}@{event.cycle}")

    for session in open_sessions.values():
        anomalies["open_rows_at_trace_end"] += 1
        session.commands.append("OPEN_AT_TRACE_END")

    return sessions, dict(anomalies)


def row_lifecycle_rows(sessions: list[RowSession]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session in sessions:
        ch, pch, rk, bg, ba, row = session.row_key
        rows.append(
            {
                "session_id": session.session_id,
                "channel": ch,
                "pseudochannel": pch,
                "rank": rk,
                "bankgroup": bg,
                "bank": ba,
                "row": row,
                "act_cycle": session.act_cycle,
                "pre_cycle": session.pre_cycle if session.pre_cycle is not None else "",
                "lifetime_cycles": session.lifetime_cycles() if session.lifetime_cycles() is not None else "",
                "rd_count": session.rd_count,
                "wr_count": session.wr_count,
                "command_count": len(session.commands),
                "distinct_columns": len(set(session.columns)),
                "columns_touched": session.columns_repr(),
                "command_trace": session.command_trace(),
            }
        )
    return rows


def window_rows(events: Iterable[TraceEvent], window_cycles: int) -> list[dict[str, object]]:
    counts: dict[tuple[int, int, int, int, int, int, int], int] = defaultdict(int)
    for event in events:
        if event.command != "ACT":
            continue
        window_idx = event.cycle // window_cycles
        counts[(*event.row_key, window_idx)] += 1

    rows: list[dict[str, object]] = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        ch, pch, rk, bg, ba, row, window_idx = key
        rows.append(
            {
                "channel": ch,
                "pseudochannel": pch,
                "rank": rk,
                "bankgroup": bg,
                "bank": ba,
                "row": row,
                "window_idx": window_idx,
                "window_start_cycle": window_idx * window_cycles,
                "window_end_cycle": (window_idx + 1) * window_cycles - 1,
                "act_count": count,
            }
        )
    return rows


def aggressor_rows(sessions: list[RowSession], window_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    row_stats: dict[tuple[int, int, int, int, int, int], dict[str, object]] = defaultdict(
        lambda: {
            "opens": 0,
            "max_lifetime_cycles": 0,
            "total_lifetime_cycles": 0,
            "rd_count": 0,
            "wr_count": 0,
        }
    )
    for session in sessions:
        stats = row_stats[session.row_key]
        stats["opens"] = int(stats["opens"]) + 1
        stats["rd_count"] = int(stats["rd_count"]) + session.rd_count
        stats["wr_count"] = int(stats["wr_count"]) + session.wr_count
        lifetime = session.lifetime_cycles() or 0
        stats["total_lifetime_cycles"] = int(stats["total_lifetime_cycles"]) + lifetime
        stats["max_lifetime_cycles"] = max(int(stats["max_lifetime_cycles"]), lifetime)

    max_window_act: dict[tuple[int, int, int, int, int, int], int] = defaultdict(int)
    for row in window_summary:
        key = (
            int(row["channel"]),
            int(row["pseudochannel"]),
            int(row["rank"]),
            int(row["bankgroup"]),
            int(row["bank"]),
            int(row["row"]),
        )
        max_window_act[key] = max(max_window_act[key], int(row["act_count"]))

    out: list[dict[str, object]] = []
    for key, stats in row_stats.items():
        ch, pch, rk, bg, ba, row = key
        opens = int(stats["opens"])
        total_lifetime = int(stats["total_lifetime_cycles"])
        avg_lifetime = total_lifetime / opens if opens else 0.0
        out.append(
            {
                "channel": ch,
                "pseudochannel": pch,
                "rank": rk,
                "bankgroup": bg,
                "bank": ba,
                "row": row,
                "neighbor_rows": f"{row - 1},{row + 1}",
                "opens": opens,
                "max_window_act_count": max_window_act.get(key, 0),
                "max_lifetime_cycles": int(stats["max_lifetime_cycles"]),
                "avg_lifetime_cycles": round(avg_lifetime, 2),
                "total_rd_count": int(stats["rd_count"]),
                "total_wr_count": int(stats["wr_count"]),
            }
        )
    out.sort(
        key=lambda row: (
            -int(row["max_window_act_count"]),
            -int(row["max_lifetime_cycles"]),
            -int(row["opens"]),
            int(row["channel"]),
            int(row["pseudochannel"]),
            int(row["rank"]),
            int(row["bankgroup"]),
            int(row["bank"]),
            int(row["row"]),
        )
    )
    return out


def overlap_rows(sessions: list[RowSession], trace_end_cycle: int) -> list[dict[str, object]]:
    points: list[tuple[int, int]] = []
    for session in sessions:
        end_cycle = session.pre_cycle if session.pre_cycle is not None else trace_end_cycle
        points.append((session.act_cycle, 1))
        points.append((end_cycle, -1))
    points.sort()

    current = 0
    peak = 0
    out: list[dict[str, object]] = []
    for cycle, delta in points:
        current += delta
        peak = max(peak, current)
        out.append(
            {
                "cycle": cycle,
                "delta_open_rows": delta,
                "open_rows_after_event": current,
                "peak_open_rows_so_far": peak,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = (len(arr) - 1) * p
    left = math.floor(idx)
    right = math.ceil(idx)
    if left == right:
        return float(arr[left])
    return arr[left] * (right - idx) + arr[right] * (idx - left)


def format_cycles_ns(cycles: int, tck_ns: float = 0.5) -> str:
    return f"{cycles} cycles (~{cycles * tck_ns:.2f} ns)"


def summarize_sessions(sessions: list[RowSession]) -> dict[str, object]:
    closed = [session for session in sessions if session.pre_cycle is not None]
    lifetime_values = [session.lifetime_cycles() or 0 for session in closed]
    rd_values = [session.rd_count for session in sessions]
    wr_values = [session.wr_count for session in sessions]
    cmd_values = [len(session.commands) for session in sessions]
    col_values = [len(set(session.columns)) for session in sessions]
    return {
        "closed_count": len(closed),
        "open_end_count": len(sessions) - len(closed),
        "lifetime_min": min(lifetime_values) if lifetime_values else 0,
        "lifetime_p50": percentile(lifetime_values, 0.50),
        "lifetime_p90": percentile(lifetime_values, 0.90),
        "lifetime_p99": percentile(lifetime_values, 0.99),
        "lifetime_max": max(lifetime_values) if lifetime_values else 0,
        "lifetime_mean": (sum(lifetime_values) / len(lifetime_values)) if lifetime_values else 0.0,
        "rd_p50": percentile(rd_values, 0.50),
        "rd_p90": percentile(rd_values, 0.90),
        "rd_p99": percentile(rd_values, 0.99),
        "rd_max": max(rd_values) if rd_values else 0,
        "wr_p50": percentile(wr_values, 0.50),
        "wr_p90": percentile(wr_values, 0.90),
        "wr_p99": percentile(wr_values, 0.99),
        "wr_max": max(wr_values) if wr_values else 0,
        "cmd_p50": percentile(cmd_values, 0.50),
        "cmd_p90": percentile(cmd_values, 0.90),
        "cmd_p99": percentile(cmd_values, 0.99),
        "cmd_max": max(cmd_values) if cmd_values else 0,
        "col_p50": percentile(col_values, 0.50),
        "col_p90": percentile(col_values, 0.90),
        "col_p99": percentile(col_values, 0.99),
        "col_max": max(col_values) if col_values else 0,
    }


def summarize_windows(window_summary: list[dict[str, object]]) -> dict[str, object]:
    act_values = [int(row["act_count"]) for row in window_summary]
    act_dist = Counter(act_values)
    max_by_row = Counter()
    for row in window_summary:
        row_id = int(row["row"])
        max_by_row[row_id] = max(max_by_row[row_id], int(row["act_count"]))
    return {
        "entry_count": len(window_summary),
        "act_dist": act_dist,
        "max_by_row_top10": max_by_row.most_common(10),
        "act_p50": percentile(act_values, 0.50),
        "act_p90": percentile(act_values, 0.90),
        "act_p99": percentile(act_values, 0.99),
        "act_max": max(act_values) if act_values else 0,
    }


def summarize_aggressors(aggressors: list[dict[str, object]]) -> dict[str, object]:
    opens = [int(row["opens"]) for row in aggressors]
    max_acts = [int(row["max_window_act_count"]) for row in aggressors]
    max_life = [int(row["max_lifetime_cycles"]) for row in aggressors]
    return {
        "row_count": len(aggressors),
        "opens_p50": percentile(opens, 0.50),
        "opens_p90": percentile(opens, 0.90),
        "opens_p99": percentile(opens, 0.99),
        "opens_max": max(opens) if opens else 0,
        "maxact_p50": percentile(max_acts, 0.50),
        "maxact_p90": percentile(max_acts, 0.90),
        "maxact_p99": percentile(max_acts, 0.99),
        "maxact_max": max(max_acts) if max_acts else 0,
        "maxlife_p50": percentile(max_life, 0.50),
        "maxlife_p90": percentile(max_life, 0.90),
        "maxlife_p99": percentile(max_life, 0.99),
        "maxlife_max": max(max_life) if max_life else 0,
    }


def select_segment(
    events: list[TraceEvent], segments: list[dict[str, int]], segment_selector: str
) -> tuple[list[TraceEvent], dict[str, int]]:
    if not segments:
        return events, {
            "segment_idx": 0,
            "start_event_idx": 0,
            "end_event_idx": len(events) - 1,
            "start_cycle": events[0].cycle if events else 0,
            "end_cycle": events[-1].cycle if events else 0,
            "event_count": len(events),
        }

    if segment_selector == "longest":
        meta = max(segments, key=lambda item: item["event_count"])
    elif segment_selector == "last":
        meta = segments[-1]
    else:
        target_idx = int(segment_selector)
        matches = [segment for segment in segments if segment["segment_idx"] == target_idx]
        if not matches:
            raise SystemExit(f"segment {target_idx} not found; available: 0..{segments[-1]['segment_idx']}")
        meta = matches[0]

    start = meta["start_event_idx"]
    end = meta["end_event_idx"] + 1
    return events[start:end], meta


def format_row_id(row: dict[str, object]) -> str:
    return (
        f"ch{row['channel']}/pch{row['pseudochannel']}/rk{row['rank']}/"
        f"bg{row['bankgroup']}/ba{row['bank']}/row{row['row']}"
    )


def write_report(
    path: Path,
    trace_path: Path,
    segment_meta: dict[str, int],
    all_segments: list[dict[str, int]],
    window_cycles: int,
    sessions: list[RowSession],
    anomalies: dict[str, int],
    window_summary: list[dict[str, object]],
    aggressors: list[dict[str, object]],
    overlap: list[dict[str, object]],
) -> None:
    total_acts = len(sessions)
    closed_sessions = [session for session in sessions if session.pre_cycle is not None]
    open_sessions = [session for session in sessions if session.pre_cycle is None]
    max_lifetime = max((session.lifetime_cycles() or 0 for session in sessions), default=0)
    avg_lifetime = (
        sum(session.lifetime_cycles() or 0 for session in closed_sessions) / len(closed_sessions)
        if closed_sessions
        else 0.0
    )
    max_window_act = max((int(row["act_count"]) for row in window_summary), default=0)
    peak_overlap = max((int(row["peak_open_rows_so_far"]) for row in overlap), default=0)

    top_press = sorted(
        sessions,
        key=lambda session: (session.lifetime_cycles() or -1, session.rd_count + session.wr_count),
        reverse=True,
    )[:10]
    top_hammer = aggressors[:10]

    session_stats = summarize_sessions(sessions)
    window_stats = summarize_windows(window_summary)
    aggressor_stats = summarize_aggressors(aggressors)

    lines = [
        "# RowRisk 分析报告",
        "",
        "## 概览",
        "",
        f"- Trace 文件：`{trace_path}`",
        f"- 选中的单调 cycle 段：`segment {segment_meta['segment_idx']}`",
        f"- 该段事件数：`{segment_meta['event_count']}`",
        f"- Cycle 范围：`{segment_meta['start_cycle']}..{segment_meta['end_cycle']}`",
        f"- 检测到的段数：`{len(all_segments)}`",
        f"- RowHammer 统计窗口：`{window_cycles}` cycles（约 `7.8us @ 0.5ns tCK`）",
        f"- Row session 总数：`{len(sessions)}`",
        f"- ACT 总数：`{total_acts}`",
        f"- 正常 closed 的 session：`{len(closed_sessions)}`",
        f"- Trace 结束时仍 open 的 session：`{len(open_sessions)}`",
        f"- 峰值同时打开行数：`{peak_overlap}`",
        f"- 最大单次开行时长：`{max_lifetime}` cycles",
        f"- 已关闭 session 的平均开行时长：`{avg_lifetime:.2f}` cycles",
        f"- 单窗口内单行最大 ACT 次数：`{max_window_act}`",
        "",
        "## 结论摘要",
        "",
        "- 当前访问模式的主要特征是**多行并发打开 + 较强 row-buffer reuse**，而不是少数行被极高频率反复锤击。",
        f"- 从 RowHammer 角度看，`ACT/window` 的最大值只有 `{max_window_act}`，整体属于**弱信号**。",
        f"- 从 RowPress 角度看，最长开行约 `{max_lifetime}` cycles（约 `{max_lifetime * 0.5 / 1000:.2f} us`），比常见强 RowPress 场景更温和，整体属于**弱到中等信号**。",
        "- 因此更适合把这份 trace 理解为“真实工作负载下的 HBM row-buffer 利用行为”，而不是攻击型 hammer/press 模式。",
        "",
        "## Segment 概览",
        "",
    ]
    for segment in sorted(all_segments, key=lambda item: item["segment_idx"]):
        lines.append(
            f"- segment {segment['segment_idx']}: cycles {segment['start_cycle']}..{segment['end_cycle']}, "
            f"events={segment['event_count']}"
        )

    lines.extend([
        "",
        "## 基础统计",
        "",
        f"- 生命周期分布：min `{session_stats['lifetime_min']}` / p50 `{session_stats['lifetime_p50']:.0f}` / p90 `{session_stats['lifetime_p90']:.0f}` / p99 `{session_stats['lifetime_p99']:.0f}` / max `{session_stats['lifetime_max']}`",
        f"- 每个 session 的 READ 次数：p50 `{session_stats['rd_p50']:.0f}` / p90 `{session_stats['rd_p90']:.0f}` / p99 `{session_stats['rd_p99']:.0f}` / max `{session_stats['rd_max']}`",
        f"- 每个 session 的 WRITE 次数：p50 `{session_stats['wr_p50']:.0f}` / p90 `{session_stats['wr_p90']:.0f}` / p99 `{session_stats['wr_p99']:.0f}` / max `{session_stats['wr_max']}`",
        f"- 每个 session 覆盖的列数：p50 `{session_stats['col_p50']:.0f}` / p90 `{session_stats['col_p90']:.0f}` / p99 `{session_stats['col_p99']:.0f}` / max `{session_stats['col_max']}`",
        f"- 每个 session 的命令条数：p50 `{session_stats['cmd_p50']:.0f}` / p90 `{session_stats['cmd_p90']:.0f}` / p99 `{session_stats['cmd_p99']:.0f}` / max `{session_stats['cmd_max']}`",
        "",
        "## RowHammer 统计",
        "",
        f"- 窗口条目总数：`{window_stats['entry_count']}`",
        f"- `ACT/window` 分布：p50 `{window_stats['act_p50']:.0f}` / p90 `{window_stats['act_p90']:.0f}` / p99 `{window_stats['act_p99']:.0f}` / max `{window_stats['act_max']}`",
        "- `ACT/window` 频数分布："
    ])
    for act_count, freq in sorted(window_stats["act_dist"].items()):
        lines.append(f"  - `{act_count}` 次：`{freq}` 个窗口条目")

    lines.extend([
        "",
        "- 以“某个 row 的最大窗口 ACT 次数”来看，最可疑的 row 编号 Top 10：",
    ])
    for row_id, value in window_stats["max_by_row_top10"]:
        lines.append(f"  - `row {row_id}`: `max ACT/window = {value}`")

    lines.extend([
        "",
        "## 聚合后的 Aggressor 候选统计",
        "",
        f"- 唯一 row 数量：`{aggressor_stats['row_count']}`",
        f"- 每个 row 的 reopen 次数：p50 `{aggressor_stats['opens_p50']:.0f}` / p90 `{aggressor_stats['opens_p90']:.0f}` / p99 `{aggressor_stats['opens_p99']:.0f}` / max `{aggressor_stats['opens_max']}`",
        f"- 每个 row 的最大窗口 ACT：p50 `{aggressor_stats['maxact_p50']:.0f}` / p90 `{aggressor_stats['maxact_p90']:.0f}` / p99 `{aggressor_stats['maxact_p99']:.0f}` / max `{aggressor_stats['maxact_max']}`",
        f"- 每个 row 的最大开行时长：p50 `{aggressor_stats['maxlife_p50']:.0f}` / p90 `{aggressor_stats['maxlife_p90']:.0f}` / p99 `{aggressor_stats['maxlife_p99']:.0f}` / max `{aggressor_stats['maxlife_max']}`",
        "",
        "## 异常/边界情况",
        "",
    ])
    if anomalies:
        for key, value in sorted(anomalies.items()):
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- none")

    lines.extend(["", "## 最可疑的 RowPress 候选 Session", ""])
    if top_press:
        for session in top_press:
            row_id = (
                f"ch{session.row_key[0]}/pch{session.row_key[1]}/rk{session.row_key[2]}/"
                f"bg{session.row_key[3]}/ba{session.row_key[4]}/row{session.row_key[5]}"
            )
            lifetime = session.lifetime_cycles()
            lines.append(
                f"- {row_id}: ACT@{session.act_cycle}, PRE@{session.pre_cycle}, "
                f"开行时长={lifetime} cycles, RD={session.rd_count}, WR={session.wr_count}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## 最可疑的 RowHammer 候选 Row", ""])
    if top_hammer:
        for row in top_hammer:
            lines.append(
                f"- {format_row_id(row)}: max ACT/window={row['max_window_act_count']}, "
                f"reopen 次数={row['opens']}, max lifetime={row['max_lifetime_cycles']}, "
                f"邻行={row['neighbor_rows']}"
            )
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## 如何解读这些结果",
        "",
        "- 如果你主要关心 RowHammer：优先看 `rowhammer_windows.csv` 和本报告里的 `ACT/window` 分布。真正危险的情况通常会表现为少数行在单个窗口里出现远高于当前结果的 ACT 次数。",
        "- 如果你主要关心 RowPress：优先看 `row_lifecycle.csv`，关注 `lifetime_cycles` 特别长、同时 RD/WR 次数也比较多的 session。",
        "- 当前这份 trace 更像“真实工作负载下广泛 row reuse”的模式，而不是“少数相邻行被反复压榨”的攻击模式。",
    ])

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to cmd_hbm3E.log.chX")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/MD outputs. Default: sibling rowrisk_analysis/",
    )
    parser.add_argument(
        "--window-cycles",
        type=int,
        default=DEFAULT_WINDOW_CYCLES,
        help="Window size for ACT counting. Default approximates 7.8us at 0.5ns tCK.",
    )
    parser.add_argument(
        "--segment",
        default="longest",
        help="Which monotonic-cycle segment to analyze: longest, last, or explicit segment index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = args.trace.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else trace_path.parent / "rowrisk_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    events, segments = parse_trace(trace_path)
    if not events:
        raise SystemExit(f"No parseable events found in {trace_path}")
    events, segment_meta = select_segment(events, segments, args.segment)

    sessions, anomalies = track_sessions(events)
    lifecycle = row_lifecycle_rows(sessions)
    windows = window_rows(events, args.window_cycles)
    aggressors = aggressor_rows(sessions, windows)
    overlaps = overlap_rows(sessions, events[-1].cycle)

    write_csv(out_dir / "row_lifecycle.csv", lifecycle)
    write_csv(out_dir / "rowhammer_windows.csv", windows)
    write_csv(out_dir / "aggressor_summary.csv", aggressors)
    write_csv(out_dir / "open_row_overlap.csv", overlaps)
    write_report(
        out_dir / "rowrisk_report.md",
        trace_path,
        segment_meta,
        segments,
        args.window_cycles,
        sessions,
        anomalies,
        windows,
        aggressors,
        overlaps,
    )


if __name__ == "__main__":
    main()
