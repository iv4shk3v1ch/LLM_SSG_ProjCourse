from __future__ import annotations

from typing import Any

from .utils import now_iso


def combine_report(
    spec: dict[str, Any] | None = None,
    scaffold: dict[str, Any] | None = None,
    build: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    llm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_insights = _derive_runtime_insights(runtime)
    return {
        "report_generated_at_utc": now_iso(),
        "runtime_environment": runtime_insights.get("runtime_environment"),
        "runtime_warning": runtime_insights.get("warning"),
        "runtime_summary": runtime_insights.get("summary"),
        "spec": spec,
        "scaffold": scaffold,
        "measurements": {
            "build": build,
            "runtime": runtime,
            "llm": llm,
        },
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    spec = report.get("spec") or {}
    site = spec.get("site") or {}
    build = ((report.get("measurements") or {}).get("build")) or {}
    runtime = ((report.get("measurements") or {}).get("runtime")) or {}
    llm = ((report.get("measurements") or {}).get("llm")) or {}
    energy_proxy = (build.get("energy_proxy") or {}) if build else {}
    artifact = (build.get("build_artifact") or {}) if build else {}
    static_summary = ((runtime.get("static") or {}).get("summary") or {}) if runtime else {}
    dynamic = runtime.get("dynamic") if runtime else None
    dynamic_summary = (dynamic.get("summary") or {}) if dynamic else {}
    runtime_environment = report.get("runtime_environment", "unknown")
    runtime_warning = report.get("runtime_warning")
    runtime_summary = report.get("runtime_summary") or {}

    lines = [
        "# SSG Case-Study Run Summary",
        "",
        "## Site",
        f"- Name: {site.get('name', 'N/A')}",
        f"- Type: {site.get('type', 'N/A')}",
        f"- Deployment target: {site.get('deployment_target', 'N/A')}",
        "",
        "## Build Metrics",
        f"- Build command: {build.get('command', 'N/A')}",
        f"- Exit code: {build.get('exit_code', 'N/A')}",
        f"- Wall time (s): {_fmt_num(build.get('wall_seconds'))}",
        f"- CPU time (s): {_fmt_num(build.get('cpu_seconds'))}",
        f"- Energy proxy (J): {_fmt_num(energy_proxy.get('joules'))} (cpu_time_seconds_x_assumed_cpu_watts)",
        f"- Build artifact path: {artifact.get('path', 'N/A')}",
        f"- Build artifact size (bytes): {_fmt_num(artifact.get('size_bytes'))}",
        f"- Build artifact file count: {_fmt_num(artifact.get('file_count'))}",
        "",
        "## LLM Usage Proxy",
        f"- Calls: {llm.get('calls', 'N/A')}",
        f"- Prompt tokens: {llm.get('prompt_tokens', 'N/A')}",
        f"- Completion tokens: {llm.get('completion_tokens', 'N/A')}",
        f"- Total tokens: {llm.get('total_tokens', 'N/A')}",
        "",
        "## Runtime Proxy",
        f"- Runtime environment: {runtime_environment}",
        f"- Static request count: {_fmt_num((runtime.get('static') or {}).get('successes'))}",
        f"- Static transfer bytes (total): {_fmt_num((runtime_summary.get('static') or {}).get('transfer_bytes'))}",
        f"- Static avg transfer (bytes/request): {_fmt_num(static_summary.get('avg_transfer_bytes'))}",
    ]

    if dynamic:
        lines.extend(
            [
                f"- Dynamic request count: {_fmt_num((runtime.get('dynamic') or {}).get('successes'))}",
                f"- Dynamic transfer bytes (total): {_fmt_num((runtime_summary.get('dynamic') or {}).get('transfer_bytes'))}",
                f"- Dynamic avg transfer (bytes/request): {_fmt_num(dynamic_summary.get('avg_transfer_bytes'))}",
                f"- Transfer ratio (dynamic/static): {_fmt_num(runtime_summary.get('transfer_ratio'))}",
                (
                    "- Transfer size identical: "
                    f"{_fmt_bool(runtime_summary.get('transfer_size_identical'))}"
                ),
            ]
        )
    if runtime_warning:
        lines.append(f"- Runtime warning: {runtime_warning}")

    lines.extend(
        [
            "",
            "## Notes",
            "- This prototype excludes AI content generation by design (templates/placeholders only).",
            "- Energy values are proxies unless hardware joule telemetry is integrated.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return str(value)


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "N/A"


def _derive_runtime_insights(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if not runtime:
        return {
            "runtime_environment": "unknown",
            "warning": None,
            "summary": None,
        }

    mode = runtime.get("mode") or {}
    static = runtime.get("static") or {}
    dynamic = runtime.get("dynamic") or {}
    static_attempts = static.get("attempts") or []
    dynamic_attempts = dynamic.get("attempts") or []

    both_local = (
        mode.get("static_source") == "local_dir"
        and mode.get("dynamic_source") == "local_dynamic_server"
    )
    runtime_environment = "local_loopback" if both_local else "mixed_or_remote"
    warning = (
        "Local runtime comparison does not reflect real-world hosting differences."
        if both_local
        else None
    )

    static_transfer_bytes = sum(
        int(a.get("transfer_bytes", 0)) for a in static_attempts if a.get("error") is None
    )
    dynamic_transfer_bytes = sum(
        int(a.get("transfer_bytes", 0)) for a in dynamic_attempts if a.get("error") is None
    )
    static_request_count = int(static.get("successes", 0))
    dynamic_request_count = int(dynamic.get("successes", 0)) if dynamic else 0
    transfer_ratio = None
    if dynamic and static_transfer_bytes > 0:
        transfer_ratio = float(dynamic_transfer_bytes / static_transfer_bytes)

    transfer_size_identical = None
    if dynamic:
        transfer_size_identical = bool(dynamic_transfer_bytes == static_transfer_bytes)

    summary = {
        "static": {
            "request_count": static_request_count,
            "transfer_bytes": static_transfer_bytes,
        },
        "dynamic": (
            {
                "request_count": dynamic_request_count,
                "transfer_bytes": dynamic_transfer_bytes,
            }
            if dynamic
            else None
        ),
        "transfer_ratio": transfer_ratio,
        "transfer_size_identical": transfer_size_identical,
    }

    return {
        "runtime_environment": runtime_environment,
        "warning": warning,
        "summary": summary,
    }
