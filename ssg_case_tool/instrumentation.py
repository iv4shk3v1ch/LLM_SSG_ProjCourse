from __future__ import annotations

import json
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import request

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

from .utils import now_iso


def run_build_measurement(
    command: str,
    project_dir: Path,
    cpu_watt_assumption: float = 35.0,
    shell: bool = False,
) -> dict[str, Any]:
    started_at = now_iso()
    start = time.perf_counter()

    args: str | list[str] = command if shell else shlex.split(command, posix=False)
    if psutil is not None:
        process = psutil.Popen(
            args,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell,
        )
        stdout, stderr = process.communicate()
        exit_code = int(process.returncode)
    else:
        process = subprocess.Popen(
            args,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell,
        )
        stdout, stderr = process.communicate()
        exit_code = int(process.returncode)
    end = time.perf_counter()
    wall_seconds = end - start

    cpu_seconds: float | None
    if psutil is None:
        cpu_seconds = None
    else:
        try:
            cpu_times = process.cpu_times()
            cpu_seconds = float(cpu_times.user + cpu_times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cpu_seconds = None

    energy_proxy_joules = None
    if cpu_seconds is not None:
        energy_proxy_joules = cpu_seconds * cpu_watt_assumption

    return {
        "metric_type": "build",
        "started_at_utc": started_at,
        "finished_at_utc": now_iso(),
        "command": command,
        "project_dir": str(project_dir),
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "energy_proxy": {
            "label": "cpu_time_seconds_x_assumed_cpu_watts",
            "assumed_cpu_watts": cpu_watt_assumption,
            "joules": energy_proxy_joules,
        },
        "notes": (
            []
            if psutil is not None
            else ["psutil not installed: cpu_seconds unavailable, energy proxy omitted."]
        ),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _tail(value: str, max_chars: int = 2500) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def record_llm_call(
    log_path: Path,
    session: str,
    step: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_chars: int | None = None,
) -> dict[str, Any]:
    call = {
        "metric_type": "llm_call",
        "timestamp_utc": now_iso(),
        "session": session,
        "step": step,
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "prompt_chars": int(prompt_chars) if prompt_chars is not None else None,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(call, ensure_ascii=True) + "\n")
    return call


def aggregate_llm_calls(log_path: Path, session: str | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return {
            "metric_type": "llm_aggregate",
            "session": session,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "steps": {},
        }

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if session and row.get("session") != session:
            continue
        rows.append(row)

    steps: dict[str, dict[str, int]] = {}
    for row in rows:
        step = str(row.get("step", "unknown"))
        current = steps.setdefault(step, {"calls": 0, "tokens": 0})
        current["calls"] += 1
        current["tokens"] += int(row.get("total_tokens", 0))

    return {
        "metric_type": "llm_aggregate",
        "session": session,
        "calls": len(rows),
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
        "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in rows),
        "total_tokens": sum(int(r.get("total_tokens", 0)) for r in rows),
        "steps": steps,
    }


def measure_runtime_url(url: str, runs: int = 5, timeout_sec: float = 10.0) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for _ in range(runs):
        started = time.perf_counter()
        ttfb_ms: float | None = None
        status: int | None = None
        bytes_total = 0
        error: str | None = None
        try:
            req = request.Request(url, headers={"User-Agent": "ssg-case-tool/0.1"})
            with request.urlopen(req, timeout=timeout_sec) as response:
                status = int(getattr(response, "status", 200))
                first = response.read(1)
                ttfb_ms = (time.perf_counter() - started) * 1000.0
                rest = response.read()
                bytes_total = len(first) + len(rest)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency_ms = (time.perf_counter() - started) * 1000.0
        attempts.append(
            {
                "status": status,
                "ttfb_ms": ttfb_ms,
                "latency_ms": latency_ms,
                "transfer_bytes": bytes_total,
                "error": error,
            }
        )

    successful = [a for a in attempts if a["error"] is None]
    latencies = [float(a["latency_ms"]) for a in successful]
    transfers = [int(a["transfer_bytes"]) for a in successful]
    ttfbs = [float(a["ttfb_ms"]) for a in successful if a["ttfb_ms"] is not None]

    return {
        "url": url,
        "runs": runs,
        "successes": len(successful),
        "failures": runs - len(successful),
        "attempts": attempts,
        "summary": {
            "avg_latency_ms": _safe_mean(latencies),
            "p95_latency_ms": _safe_p95(latencies),
            "avg_ttfb_ms": _safe_mean(ttfbs),
            "avg_transfer_bytes": _safe_mean(transfers),
        },
    }


def measure_runtime_comparison(
    static_url: str,
    dynamic_url: str | None,
    runs: int = 5,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    static_metrics = measure_runtime_url(static_url, runs=runs, timeout_sec=timeout_sec)
    dynamic_metrics = (
        measure_runtime_url(dynamic_url, runs=runs, timeout_sec=timeout_sec) if dynamic_url else None
    )

    comparison = None
    if dynamic_metrics:
        s_lat = static_metrics["summary"]["avg_latency_ms"]
        d_lat = dynamic_metrics["summary"]["avg_latency_ms"]
        s_transfer = static_metrics["summary"]["avg_transfer_bytes"]
        d_transfer = dynamic_metrics["summary"]["avg_transfer_bytes"]
        comparison = {
            "latency_ratio_dynamic_over_static": _safe_ratio(d_lat, s_lat),
            "transfer_ratio_dynamic_over_static": _safe_ratio(d_transfer, s_transfer),
        }

    return {
        "metric_type": "runtime",
        "measured_at_utc": now_iso(),
        "static": static_metrics,
        "dynamic": dynamic_metrics,
        "comparison": comparison,
    }


def _safe_mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_p95(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = max(0, int(round(0.95 * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator / denominator)
