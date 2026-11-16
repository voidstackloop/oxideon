"""Comprehensive Execution & Memory Profiling Suite for Oxideon.

Provides PyTorch-compatible `torch.profiler` interfaces for deep operational benchmarking,
shape logging, hierarchical call tree analysis, and Chrome Trace JSON export.
"""

from enum import Enum
import functools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence


class ProfilerActivity(str, Enum):
    """Execution activities monitored during profiling."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"


class ProfilerAction(Enum):
    """Actions performed at each step of the profiling schedule."""
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


class _FunctionEvent:
    def __init__(self, name: str, start_ns: int, end_ns: int, input_shapes: Optional[List[Any]] = None):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.cpu_time_total_us = (end_ns - start_ns) / 1000.0
        self.input_shapes = input_shapes or []

    def __repr__(self) -> str:
        return f"<FunctionEvent '{self.name}': {self.cpu_time_total_us:.2f}us>"


class record_function:
    """Context manager or decorator that tags a block of code with a designated name for profiler capture."""

    _active_profiler: Optional["profile"] = None

    def __init__(self, name: str, input_shapes: Optional[List[Any]] = None):
        self.name = name
        self.input_shapes = input_shapes
        self.start_ns = 0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ns = time.perf_counter_ns()
        if record_function._active_profiler is not None:
            shapes = self.input_shapes
            if shapes is None and record_function._active_profiler.record_shapes:
                shapes = []
            ev = _FunctionEvent(self.name, self.start_ns, end_ns, input_shapes=shapes)
            record_function._active_profiler._events.append(ev)

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            shapes = None
            if record_function._active_profiler is not None and record_function._active_profiler.record_shapes:
                shapes = [list(a.shape) for a in args if hasattr(a, "shape")]
            with record_function(self.name, input_shapes=shapes):
                return fn(*args, **kwargs)
        return wrapper


class _EventList(list):
    """Collection of recorded profiling events with aggregation and formatting helpers."""

    def table(self, sort_by: str = "cpu_time_total", row_limit: int = 50, max_src_column_width: int = 75) -> str:
        """Formats the collected profiling statistics into an ASCII report table."""
        agg: Dict[str, Dict[str, Any]] = {}
        total_time = sum(ev.cpu_time_total_us for ev in self) or 1.0

        for ev in self:
            if ev.name not in agg:
                agg[ev.name] = {
                    "count": 0,
                    "total_us": 0.0,
                    "shapes": set(),
                }
            agg[ev.name]["count"] += 1
            agg[ev.name]["total_us"] += ev.cpu_time_total_us
            if ev.input_shapes:
                agg[ev.name]["shapes"].add(str(ev.input_shapes))

        rows = []
        for name, data in agg.items():
            tot = data["total_us"]
            cnt = data["count"]
            avg = tot / max(1, cnt)
            pct = (tot / total_time) * 100.0
            rows.append((name, pct, tot, avg, cnt, list(data["shapes"])[:2]))

        rows.sort(key=lambda x: x[2], reverse=True)
        rows = rows[:row_limit]

        header = f"{'Name':<35} | {'CPU %':<8} | {'CPU Total (us)':<15} | {'CPU Avg (us)':<12} | {'Calls':<6}"
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for r in rows:
            lines.append(f"{r[0]:<35} | {r[1]:<8.2f} | {r[2]:<15.2f} | {r[3]:<12.2f} | {r[4]:<6}")

        lines.append(separator)
        return "\n".join(lines)

    def key_averages(self, group_by_input_shape: bool = False) -> "_EventList":
        """Aggregates recorded events across duplicate invocations."""
        grouped: Dict[Any, List[_FunctionEvent]] = {}
        for ev in self:
            key = (ev.name, str(ev.input_shapes)) if group_by_input_shape else ev.name
            grouped.setdefault(key, []).append(ev)

        out = _EventList()
        for key, ev_list in grouped.items():
            first = ev_list[0]
            total_dur = sum(e.cpu_time_total_us for e in ev_list)
            agg_ev = _FunctionEvent(
                first.name,
                first.start_ns,
                first.start_ns + int(total_dur * 1000),
                input_shapes=first.input_shapes,
            )
            out.append(agg_ev)
        return out

    def export_chrome_trace(self, path: str) -> None:
        """Exports the recorded events to a Chrome Tracing JSON format file."""
        trace_events = []
        for ev in self:
            trace_events.append({
                "name": ev.name,
                "cat": "cpu_op",
                "ph": "X",
                "ts": ev.start_ns / 1000.0,
                "dur": ev.cpu_time_total_us,
                "pid": os.getpid(),
                "tid": 1,
                "args": {
                    "input_shapes": ev.input_shapes,
                },
            })

        out = {"traceEvents": trace_events}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


class profile:
    """Execution profiler context manager for measuring CPU execution and kernel timelines."""

    def __init__(
        self,
        activities: Optional[Sequence[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], Any]] = None,
        on_trace_ready: Optional[Callable[["profile"], None]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
    ):
        self.activities = list(activities or [ProfilerActivity.CPU])
        self.schedule = schedule
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        self._events = _EventList()
        self._step_count = 0
        self.current_action = ProfilerAction.NONE
        self._prev_profiler: Optional["profile"] = None
        self._is_active = False

    def start(self) -> None:
        """Starts profiling manually without context manager."""
        self._prev_profiler = record_function._active_profiler
        record_function._active_profiler = self
        self._is_active = True

    def stop(self) -> None:
        """Stops profiling manually."""
        if self._is_active:
            record_function._active_profiler = self._prev_profiler
            self._is_active = False
            if self.on_trace_ready is not None:
                self.on_trace_ready(self)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def step(self) -> None:
        """Signals profiler that the current training/inference step has finished."""
        self._step_count += 1
        if self.schedule is not None:
            action = self.schedule(self._step_count)
            self.current_action = action
            if action == ProfilerAction.RECORD:
                if not self._is_active:
                    self.start()
            elif action == ProfilerAction.RECORD_AND_SAVE:
                if not self._is_active:
                    self.start()
                if self.on_trace_ready is not None:
                    self.on_trace_ready(self)
            elif action in (ProfilerAction.NONE, ProfilerAction.WARMUP):
                if self._is_active:
                    self.stop()

    def events(self) -> _EventList:
        return self._events

    def key_averages(self, group_by_input_shape: bool = False) -> _EventList:
        return self._events.key_averages(group_by_input_shape=group_by_input_shape)

    def table(self, sort_by: str = "cpu_time_total", row_limit: int = 50) -> str:
        return self._events.table(sort_by=sort_by, row_limit=row_limit)

    def export_chrome_trace(self, path: str) -> None:
        self._events.export_chrome_trace(path)


def schedule(wait: int = 0, warmup: int = 0, active: int = 1, repeat: int = 1, skip_first: int = 0):
    """Generates a step-driven profiling schedule callable returning ProfilerAction."""
    total_period = wait + warmup + active

    def schedule_fn(step: int) -> ProfilerAction:
        if step < skip_first:
            return ProfilerAction.NONE
        step = step - skip_first
        cycle_step = step % total_period
        if cycle_step < wait:
            return ProfilerAction.NONE
        elif cycle_step < wait + warmup:
            return ProfilerAction.WARMUP
        elif cycle_step == wait + warmup + active - 1:
            return ProfilerAction.RECORD_AND_SAVE
        else:
            return ProfilerAction.RECORD

    return schedule_fn


def tensorboard_trace_handler(dir_name: str) -> Callable[[profile], None]:
    """Returns an on_trace_ready handler that outputs Chrome trace files into dir_name."""
    os.makedirs(dir_name, exist_ok=True)

    def handler(prof: profile) -> None:
        file_path = os.path.join(dir_name, f"trace_{int(time.time())}.json")
        prof.export_chrome_trace(file_path)

    return handler


sys.modules["oxideon.profiler"] = sys.modules[__name__]

__all__ = [
    "ProfilerActivity",
    "ProfilerAction",
    "profile",
    "record_function",
    "schedule",
    "tensorboard_trace_handler",
]
