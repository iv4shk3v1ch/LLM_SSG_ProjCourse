from __future__ import annotations

import functools
import json
import shlex
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

from .utils import now_iso

DEFAULT_BUILD_ARTIFACT_DIRS = {
    "hugo": "public",
    "eleventy": "_site",
    "zola": "public",
    "pelican": "output",
}

PLACEHOLDER_TOKEN_MARKERS = {
    "echo",
    "printf",
    "write-host",
    "write-output",
}


def run_build_measurement(
    command: str,
    project_dir: Path,
    cpu_watt_assumption: float = 65.0,
    shell: bool = False,
    ssg: str | None = None,
    artifact_dir: Path | None = None,
    capture_full_output: bool = False,
) -> dict[str, Any]:
    started_at = now_iso()
    start = time.perf_counter()

    warnings: list[str] = _placeholder_command_warnings(command)
    args: str | list[str] = command if shell else shlex.split(command, posix=False)
    process: Any | None = None
    stdout = ""
    stderr = ""
    exit_code = 127
    spawn_error: str | None = None
    monitored_cpu_seconds: float | None = None
    monitored_pids: list[int] = []
    cpu_tree_breakdown: dict[str, Any] | None = None

    try:
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
            monitored_cpu_seconds, monitored_pids, cpu_tree_breakdown = _collect_cpu_seconds_after_completion(
                process
            )
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
    except OSError as exc:
        spawn_error = str(exc)
        stderr = spawn_error

    wall_seconds = time.perf_counter() - start

    if psutil is None or process is None:
        cpu_seconds = None
    elif monitored_cpu_seconds is None:
        # psutil path should always set this; fallback to numeric zero if monitoring failed unexpectedly.
        cpu_seconds = 0.0
    else:
        cpu_seconds = float(monitored_cpu_seconds)

    energy_basis = "cpu_time_seconds"
    energy_seconds = cpu_seconds
    if energy_seconds is None or energy_seconds == 0:
        energy_basis = "wall_time_seconds"
        energy_seconds = wall_seconds
    energy_proxy_joules = float(energy_seconds) * float(cpu_watt_assumption)

    artifact_metrics = _build_artifact_metrics(project_dir=project_dir, ssg=ssg, artifact_dir=artifact_dir)
    notes: list[str] = []
    if psutil is None:
        notes.append("psutil not installed: cpu_seconds unavailable; energy proxy uses wall time.")
    if spawn_error is not None:
        notes.append("Build command failed to start. Check that the tool is installed and on PATH.")
    if artifact_metrics.get("exists") is not True:
        notes.append("Build artifact directory not found after build.")
    if cpu_seconds in {None, 0.0}:
        notes.append("Energy proxy basis switched to wall_time_seconds because cpu_seconds was zero/unavailable.")
    if cpu_seconds is not None and cpu_seconds < 0.001:
        warnings.append("CPU time below 0.001s: build likely too fast for reliable energy estimation.")

    result = {
        "metric_type": "build",
        "started_at_utc": started_at,
        "finished_at_utc": now_iso(),
        "ssg": ssg,
        "build_pid": int(process.pid) if process is not None else None,
        "observed_process_tree_pids": monitored_pids if psutil is not None else [],
        "process_tree_cpu": cpu_tree_breakdown if cpu_tree_breakdown is not None else None,
        "command": command,
        "project_dir": str(project_dir),
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_display": _format_small_float(cpu_seconds),
        "energy_proxy": {
            "label": "energy_seconds_x_assumed_system_watts",
            "energy_basis": energy_basis,
            "energy_seconds": energy_seconds,
            "assumed_system_watts": cpu_watt_assumption,
            "joules": energy_proxy_joules,
            "joules_display": _format_small_float(energy_proxy_joules),
        },
        "build_artifact": artifact_metrics,
        "warnings": warnings,
        "notes": notes,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }
    if capture_full_output:
        result["stdout_full"] = stdout
        result["stderr_full"] = stderr
    return result


def _collect_cpu_seconds_after_completion(
    process: Any,
) -> tuple[float, list[int], dict[str, Any]]:
    """Collect process-tree CPU after the build subprocess has completed."""
    if psutil is None:
        return 0.0, [], {}

    root_pid = int(getattr(process, "pid", -1))
    root_cpu = 0.0
    child_cpu = 0.0
    pids: list[int] = []

    if root_pid > 0:
        pids.append(root_pid)

    try:
        root_times = process.cpu_times()
        root_cpu = float(root_times.user + root_times.system)
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        root_cpu = 0.0

    children: list[Any] = []
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
        children = []

    seen = {root_pid} if root_pid > 0 else set()
    for child in children:
        try:
            child_pid = int(child.pid)
        except (TypeError, ValueError, AttributeError):
            continue
        if child_pid in seen:
            continue
        seen.add(child_pid)
        pids.append(child_pid)
        try:
            child_times = child.cpu_times()
            child_cpu += float(child_times.user + child_times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue

    total_cpu = root_cpu + child_cpu
    breakdown = {
        "root_pid": int(root_pid),
        "root_cpu_seconds": root_cpu,
        "children_cpu_seconds": child_cpu,
        "total_cpu_seconds": total_cpu,
        "tracked_pid_count": len(set(pids)),
        "collection_mode": "post_completion_cpu_times",
    }
    return total_cpu, sorted(set(pids)), breakdown


def _format_small_float(value: float | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "0.00000000"
    if abs(value) < 0.0005:
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return f"{value:.8f}"


def _placeholder_command_warnings(command: str) -> list[str]:
    cmd_lower = command.strip().lower()
    parts = shlex.split(cmd_lower, posix=False) if cmd_lower else []
    warnings: list[str] = []

    if not parts:
        return ["Build command is empty."]

    first_token = parts[0]
    if first_token in PLACEHOLDER_TOKEN_MARKERS:
        warnings.append("Build command appears to be a placeholder (echo/printf/write-*).")
    if "cmd" in parts and "/c" in parts and "echo" in parts:
        warnings.append("Build command uses `cmd /c echo`, likely a placeholder.")
    if "build-ok" in cmd_lower or "placeholder" in cmd_lower:
        warnings.append("Build command contains marker text often used for placeholder runs.")

    return warnings


def _build_artifact_metrics(
    project_dir: Path,
    ssg: str | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    resolved = _resolve_artifact_dir(project_dir=project_dir, ssg=ssg, artifact_dir=artifact_dir)
    if resolved is None:
        return {
            "path": None,
            "exists": False,
            "size_bytes": None,
            "file_count": 0,
        }

    if not resolved.exists() or not resolved.is_dir():
        return {
            "path": str(resolved),
            "exists": False,
            "size_bytes": None,
            "file_count": 0,
        }

    total_size = 0
    file_count = 0
    for path in resolved.rglob("*"):
        if path.is_file():
            file_count += 1
            total_size += path.stat().st_size

    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": total_size,
        "file_count": file_count,
    }


def _resolve_artifact_dir(
    project_dir: Path,
    ssg: str | None = None,
    artifact_dir: Path | None = None,
) -> Path | None:
    if artifact_dir is not None:
        return artifact_dir if artifact_dir.is_absolute() else (project_dir / artifact_dir)

    ssg_name = (ssg or "").lower()
    if ssg_name in DEFAULT_BUILD_ARTIFACT_DIRS:
        return project_dir / DEFAULT_BUILD_ARTIFACT_DIRS[ssg_name]

    for candidate in ("public", "_site", "output", "dist", "build"):
        candidate_path = project_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path
    return None


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
    static_url: str | None,
    dynamic_url: str | None,
    static_dir: Path | None = None,
    dynamic_local: bool = False,
    local_host: str = "127.0.0.1",
    runs: int = 5,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    if static_url is None and static_dir is None:
        raise ValueError("Provide either static_url or static_dir.")
    if dynamic_local and static_dir is None:
        raise ValueError("dynamic_local requires static_dir.")

    static_server: dict[str, Any] | None = None
    dynamic_server: dict[str, Any] | None = None
    notes: list[str] = []

    try:
        if static_dir is not None:
            static_server = _start_local_static_server(static_dir, host=local_host)
            static_target = static_server["url"]
            notes.append(f"Started local static server at {static_target}")
        else:
            static_target = str(static_url)

        if dynamic_local:
            html_payload = _load_static_html_payload(static_dir)
            dynamic_server = _start_local_dynamic_server(html_payload, host=local_host)
            dynamic_target = dynamic_server["url"]
            notes.append(f"Started local dynamic server at {dynamic_target}")
        else:
            dynamic_target = dynamic_url

        static_metrics = measure_runtime_url(static_target, runs=runs, timeout_sec=timeout_sec)
        dynamic_metrics = (
            measure_runtime_url(dynamic_target, runs=runs, timeout_sec=timeout_sec)
            if dynamic_target
            else None
        )
    finally:
        if dynamic_server is not None:
            _stop_local_server(dynamic_server)
            notes.append("Stopped local dynamic server.")
        if static_server is not None:
            _stop_local_server(static_server)
            notes.append("Stopped local static server.")

    comparison = None
    if dynamic_metrics:
        s_lat = static_metrics["summary"]["avg_latency_ms"]
        d_lat = dynamic_metrics["summary"]["avg_latency_ms"]
        s_transfer = static_metrics["summary"]["avg_transfer_bytes"]
        d_transfer = dynamic_metrics["summary"]["avg_transfer_bytes"]
        comparison = {
            "latency_ratio": _safe_ratio(d_lat, s_lat),
            "transfer_ratio": _safe_ratio(d_transfer, s_transfer),
            "latency_ratio_dynamic_over_static": _safe_ratio(d_lat, s_lat),
            "transfer_ratio_dynamic_over_static": _safe_ratio(d_transfer, s_transfer),
        }

    return {
        "metric_type": "runtime",
        "measured_at_utc": now_iso(),
        "mode": {
            "static_source": "local_dir" if static_dir is not None else "url",
            "dynamic_source": "local_dynamic_server" if dynamic_local else ("url" if dynamic_url else None),
        },
        "servers": {
            "static_local": _server_summary(static_server),
            "dynamic_local": _server_summary(dynamic_server),
        },
        "static": static_metrics,
        "dynamic": dynamic_metrics,
        "comparison": comparison,
        "notes": notes,
    }


class _SilentStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def _make_dynamic_handler(payload: bytes) -> type[BaseHTTPRequestHandler]:
    class _DynamicHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            # Minimal dynamic rendering path; same baseline HTML payload on each request.
            body = payload
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _DynamicHandler


def _start_local_static_server(directory: Path, host: str = "127.0.0.1") -> dict[str, Any]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Static directory does not exist: {directory}")

    handler_cls = functools.partial(_SilentStaticHandler, directory=str(directory))
    return _start_local_server(handler_cls, host=host, server_type="static", root=directory)


def _start_local_dynamic_server(payload: bytes, host: str = "127.0.0.1") -> dict[str, Any]:
    handler_cls = _make_dynamic_handler(payload)
    return _start_local_server(handler_cls, host=host, server_type="dynamic", root=None)


def _start_local_server(
    handler_cls: type[BaseHTTPRequestHandler] | Any,
    host: str,
    server_type: str,
    root: Path | None,
) -> dict[str, Any]:
    httpd = ThreadingHTTPServer((host, 0), handler_cls)
    port = int(httpd.server_port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return {
        "type": server_type,
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}/",
        "root": str(root) if root else None,
        "httpd": httpd,
        "thread": thread,
    }


def _stop_local_server(server: dict[str, Any]) -> None:
    httpd = server.get("httpd")
    thread = server.get("thread")
    if httpd is not None:
        httpd.shutdown()
        httpd.server_close()
    if thread is not None and isinstance(thread, threading.Thread):
        thread.join(timeout=2.0)


def _server_summary(server: dict[str, Any] | None) -> dict[str, Any] | None:
    if server is None:
        return None
    return {
        "type": server.get("type"),
        "host": server.get("host"),
        "port": server.get("port"),
        "url": server.get("url"),
        "root": server.get("root"),
    }


def _load_static_html_payload(static_dir: Path | None) -> bytes:
    if static_dir is None:
        return b"<html><body><h1>Dynamic baseline</h1></body></html>"

    candidates = [static_dir / "index.html"]
    candidates.extend(static_dir.rglob("index.html"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.read_bytes()

    html_files = list(static_dir.rglob("*.html"))
    if html_files:
        return html_files[0].read_bytes()

    return b"<html><body><h1>Dynamic baseline</h1></body></html>"


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
