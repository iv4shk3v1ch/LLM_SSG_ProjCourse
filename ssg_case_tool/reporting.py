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
    return {
        "report_generated_at_utc": now_iso(),
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
    static_summary = ((runtime.get("static") or {}).get("summary") or {}) if runtime else {}
    dynamic = runtime.get("dynamic") if runtime else None
    dynamic_summary = (dynamic.get("summary") or {}) if dynamic else {}
    runtime_comparison = runtime.get("comparison") or {} if runtime else {}

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
        "",
        "## LLM Usage Proxy",
        f"- Calls: {llm.get('calls', 'N/A')}",
        f"- Prompt tokens: {llm.get('prompt_tokens', 'N/A')}",
        f"- Completion tokens: {llm.get('completion_tokens', 'N/A')}",
        f"- Total tokens: {llm.get('total_tokens', 'N/A')}",
        "",
        "## Runtime Proxy",
        f"- Static avg latency (ms): {_fmt_num(static_summary.get('avg_latency_ms'))}",
        f"- Static avg transfer (bytes): {_fmt_num(static_summary.get('avg_transfer_bytes'))}",
    ]

    if dynamic:
        lines.extend(
            [
                f"- Dynamic avg latency (ms): {_fmt_num(dynamic_summary.get('avg_latency_ms'))}",
                f"- Dynamic avg transfer (bytes): {_fmt_num(dynamic_summary.get('avg_transfer_bytes'))}",
                (
                    "- Latency ratio (dynamic/static): "
                    f"{_fmt_num(runtime_comparison.get('latency_ratio_dynamic_over_static'))}"
                ),
                (
                    "- Transfer ratio (dynamic/static): "
                    f"{_fmt_num(runtime_comparison.get('transfer_ratio_dynamic_over_static'))}"
                ),
            ]
        )

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
