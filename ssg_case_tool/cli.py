from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .instrumentation import (
    DEFAULT_BUILD_ARTIFACT_DIRS,
    aggregate_llm_calls,
    measure_runtime_comparison,
    record_llm_call,
    run_build_measurement,
)
from .reporting import combine_report, render_markdown_summary
from .scaffold import SUPPORTED_SSGS, scaffold_project
from .spec_validation import validate_spec_schema
from .utils import dump_json, load_json, load_spec, now_iso, slug_to_path

DEFAULT_BUILD_COMMANDS: dict[str, str] = {
    "hugo": "hugo --minify",
    "eleventy": "npx @11ty/eleventy",
    "zola": "zola build",
    "pelican": "pelican content -o output -s pelicanconf.py",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssg-case",
        description="LLM-assisted SSG prototype with energy/performance instrumentation.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scaffold = sub.add_parser("scaffold", help="Generate an SSG scaffold from a site spec.")
    p_scaffold.add_argument("--spec", required=True, type=Path, help="Path to spec (.json/.yaml/.yml).")
    p_scaffold.add_argument("--out", required=True, type=Path, help="Output directory.")
    p_scaffold.add_argument("--ssg", choices=sorted(SUPPORTED_SSGS), help="SSG to use.")
    p_scaffold.add_argument(
        "--page-multiplier",
        type=int,
        default=1,
        help="Generate N synthetic content files per base page.",
    )
    p_scaffold.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p_scaffold.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/scaffold_manifest.json"),
        help="Where to write scaffold manifest JSON.",
    )

    p_spec = sub.add_parser(
        "spec-from-prompt",
        help="Generate a structured site spec from an informal prompt (replay mode supported).",
    )
    p_spec.add_argument("--prompt", help="Informal user prompt text.")
    p_spec.add_argument("--prompt-file", type=Path, help="Path to a file containing informal prompt text.")
    p_spec.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help="LLM generation mode. `live` is reserved for future provider integration.",
    )
    p_spec.add_argument("--replay-fixture", type=Path, help="Replay fixture JSON path (required in replay mode).")
    p_spec.add_argument("--session", required=True, help="Session id for telemetry.")
    p_spec.add_argument("--step", default="spec_generation", help="Telemetry step label.")
    p_spec.add_argument("--out-spec", required=True, type=Path, help="Output path for generated spec JSON.")
    p_spec.add_argument(
        "--out-telemetry",
        required=True,
        type=Path,
        help="Output path for deterministic replay telemetry JSON.",
    )
    p_spec.add_argument(
        "--llm-log",
        type=Path,
        default=Path("reports/llm_calls.jsonl"),
        help="JSONL file to write call rows (overwritten for deterministic replay output).",
    )

    p_build = sub.add_parser("measure-build", help="Run build and record duration/cpu/energy proxy.")
    p_build.add_argument("--project", required=True, type=Path, help="Project directory.")
    p_build.add_argument("--ssg", choices=sorted(SUPPORTED_SSGS), help="SSG for default build command.")
    p_build.add_argument("--build-command", help="Build command override, e.g. 'hugo --minify'.")
    p_build.add_argument("--artifact-dir", type=Path, help="Build output directory override.")
    p_build.add_argument(
        "--system-watts",
        type=float,
        default=65.0,
        help="Assumed system watts for energy proxy (used with cpu or wall-time basis).",
    )
    p_build.add_argument("--cpu-watts", dest="system_watts", type=float, help=argparse.SUPPRESS)
    p_build.add_argument("--shell", action="store_true", help="Run command via shell.")
    p_build.add_argument("--out", required=True, type=Path, help="Build metric output JSON path.")

    p_llm = sub.add_parser("record-llm", help="Record one LLM call usage row.")
    p_llm.add_argument("--log", required=True, type=Path, help="JSONL log path.")
    p_llm.add_argument("--session", required=True, help="Session id, e.g. portfolio-hugo-run1.")
    p_llm.add_argument("--step", required=True, help="Workflow step, e.g. scaffold/migration/report.")
    p_llm.add_argument("--model", required=True, help="LLM model id.")
    p_llm.add_argument("--prompt-tokens", required=True, type=int, help="Prompt token count.")
    p_llm.add_argument("--completion-tokens", required=True, type=int, help="Completion token count.")
    p_llm.add_argument("--prompt-chars", type=int, help="Prompt character count.")

    p_runtime = sub.add_parser(
        "measure-runtime", help="Measure static URL and optional dynamic URL latency/transfer proxy."
    )
    static_source = p_runtime.add_mutually_exclusive_group(required=True)
    static_source.add_argument("--static-url", help="Static hosted site URL.")
    static_source.add_argument("--static-dir", type=Path, help="Local static artifact directory to serve temporarily.")
    dynamic_source = p_runtime.add_mutually_exclusive_group(required=False)
    dynamic_source.add_argument("--dynamic-url", help="Dynamic site URL.")
    dynamic_source.add_argument(
        "--dynamic-local",
        action="store_true",
        help="Run a temporary local dynamic endpoint serving the same HTML payload.",
    )
    p_runtime.add_argument("--local-host", default="127.0.0.1", help="Host interface for temporary local servers.")
    p_runtime.add_argument("--runs", type=int, default=5, help="Number of runs per URL.")
    p_runtime.add_argument("--timeout-sec", type=float, default=10.0, help="Request timeout per run.")
    p_runtime.add_argument("--out", required=True, type=Path, help="Runtime metric output JSON.")

    p_compare = sub.add_parser(
        "compare-ssg",
        help="Scaffold and benchmark all SSG candidates, then rank by median energy proxy and artifact size.",
    )
    p_compare.add_argument("--spec", required=True, type=Path, help="Site spec path.")
    p_compare.add_argument(
        "--out-root",
        type=Path,
        default=Path("demos/compare"),
        help="Root directory for generated candidate projects.",
    )
    p_compare.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of build runs per candidate.",
    )
    p_compare.add_argument(
        "--page-multiplier",
        type=int,
        default=1,
        help="Generate N synthetic content files per base page before benchmarking.",
    )
    p_compare.add_argument(
        "--system-watts",
        type=float,
        default=65.0,
        help="Assumed system watts for energy proxy (used with cpu or wall-time basis).",
    )
    p_compare.add_argument("--cpu-watts", dest="system_watts", type=float, help=argparse.SUPPRESS)
    p_compare.add_argument(
        "--out",
        type=Path,
        default=Path("reports/ssg_compare.json"),
        help="Comparison output JSON path.",
    )
    p_compare.add_argument("--force", action="store_true", help="Overwrite existing scaffold files.")

    p_run = sub.add_parser(
        "run-case",
        help="Run single-SSG end-to-end flow: prompt->spec->scaffold->build->runtime->report.",
    )
    p_run.add_argument("--session", required=True, help="Session id used for all generated artifacts.")
    p_run.add_argument("--prompt", help="Informal user prompt text.")
    p_run.add_argument("--prompt-file", type=Path, help="Path to prompt text file.")
    p_run.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help="LLM generation mode. `live` is reserved for future provider integration.",
    )
    p_run.add_argument("--replay-fixture", type=Path, help="Replay fixture path (required in replay mode).")
    p_run.add_argument("--ssg", choices=sorted(SUPPORTED_SSGS), help="Selected SSG. Defaults to spec preferred.")
    p_run.add_argument(
        "--out-root",
        type=Path,
        default=Path("demos/runs"),
        help="Root folder for generated single-SSG project.",
    )
    p_run.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated measurement/report artifacts.",
    )
    p_run.add_argument(
        "--page-multiplier",
        type=int,
        default=1,
        help="Generate N synthetic content files per base page.",
    )
    p_run.add_argument("--build-command", help="Optional build command override.")
    p_run.add_argument("--artifact-dir", type=Path, help="Optional build artifact directory override.")
    p_run.add_argument("--runtime-runs", type=int, default=5, help="Number of runtime request runs.")
    p_run.add_argument("--timeout-sec", type=float, default=10.0, help="Runtime request timeout.")
    p_run.add_argument(
        "--system-watts",
        type=float,
        default=65.0,
        help="Assumed system watts for energy proxy (used with cpu or wall-time basis).",
    )
    p_run.add_argument("--cpu-watts", dest="system_watts", type=float, help=argparse.SUPPRESS)
    p_run.add_argument("--force", action="store_true", help="Overwrite scaffold files when they exist.")

    p_report = sub.add_parser("report", help="Assemble final JSON+Markdown report.")
    p_report.add_argument("--spec", type=Path, help="Spec path.")
    p_report.add_argument("--scaffold", type=Path, help="Scaffold manifest JSON.")
    p_report.add_argument("--build", type=Path, help="Build measurement JSON.")
    p_report.add_argument("--runtime", type=Path, help="Runtime measurement JSON.")
    p_report.add_argument("--compare", type=Path, help="Optional multi-SSG comparison JSON.")
    p_report.add_argument("--llm-log", type=Path, help="LLM JSONL log.")
    p_report.add_argument("--session", help="Session id filter for llm log.")
    p_report.add_argument("--out-json", required=True, type=Path, help="Combined report JSON output path.")
    p_report.add_argument("--out-md", required=True, type=Path, help="Human summary Markdown output path.")

    return parser


def _maybe_load(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        return None
    return load_json(path)


def _infer_ssg_from_project(project: Path) -> str | None:
    if (project / "pelicanconf.py").exists():
        return "pelican"
    if (project / ".eleventy.js").exists():
        return "eleventy"
    if (project / "templates").exists() and (project / "content").exists():
        return "zola"
    if (project / "layouts").exists() and (project / "content").exists():
        return "hugo"
    return None


def _compute_median(values: list[float | int | None]) -> float | None:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _rank_key(candidate: dict[str, Any]) -> tuple[float, float]:
    med = candidate.get("medians") or {}
    joules = med.get("joules")
    artifact = med.get("artifact_size_bytes")
    energy_rank = float(joules) if isinstance(joules, (int, float)) else float("inf")
    artifact_rank = float(artifact) if isinstance(artifact, (int, float)) else float("inf")
    return (energy_rank, artifact_rank)


def _artifact_dir_for_ssg(project_dir: Path, ssg: str) -> Path:
    return project_dir / DEFAULT_BUILD_ARTIFACT_DIRS.get(ssg, "public")


def _resolve_npm_diagnostics() -> dict[str, Any]:
    path_value = os.environ.get("PATH", "")
    is_windows = os.name == "nt"
    candidates = ["npm.cmd", "npm.exe", "npm"] if is_windows else ["npm"]
    resolved = []
    for candidate in candidates:
        found = shutil.which(candidate)
        resolved.append({"candidate": candidate, "resolved": found})
        if found:
            return {
                "found": True,
                "is_windows": is_windows,
                "executable": found,
                "candidate": candidate,
                "path": path_value,
                "attempts": resolved,
            }
    return {
        "found": False,
        "is_windows": is_windows,
        "executable": None,
        "candidate": None,
        "path": path_value,
        "attempts": resolved,
    }


def _resolve_compare_build_invocation(ssg: str, project_dir: Path, npm_diag: dict[str, Any]) -> dict[str, Any]:
    if ssg != "eleventy":
        return {"command": DEFAULT_BUILD_COMMANDS[ssg], "shell": False, "resolved_executable": None}

    if not npm_diag.get("found"):
        return {"command": "npm run build", "shell": True, "resolved_executable": None}

    npm_exec = str(npm_diag["executable"])
    return {
        "command": "npm run build",
        "shell": True,
        "resolved_executable": npm_exec,
    }


def _ensure_eleventy_dependencies(project_dir: Path, npm_diag: dict[str, Any]) -> dict[str, Any]:
    if not npm_diag.get("found"):
        stderr = (
            "npm executable not found for Eleventy dependency install.\n"
            f"PATH={npm_diag.get('path')}\n"
            f"attempts={npm_diag.get('attempts')}"
        )
        return {
            "executed": True,
            "exit_code": 127,
            "stdout": "",
            "stderr": stderr,
            "resolved_executable": None,
            "path": npm_diag.get("path"),
        }

    npm_exec = str(npm_diag["executable"])
    try:
        completed = subprocess.run(
            [npm_exec, "install"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "executed": True,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
            "resolved_executable": npm_exec,
            "path": npm_diag.get("path"),
        }
    return {
        "executed": True,
        "exit_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "resolved_executable": npm_exec,
        "path": npm_diag.get("path"),
    }


def _artifact_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_dir():
        return None
    file_count = 0
    total_bytes = 0
    max_mtime_ns = 0
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            file_count += 1
            total_bytes += int(stat.st_size)
            max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "max_mtime_ns": max_mtime_ns,
    }


def _render_compare_markdown(compare: dict[str, Any]) -> str:
    lines = [
        "# SSG Comparison Ranking",
        "",
        "| Rank | SSG | Median CPU (s) | Median Wall (s) | Median Energy Proxy (J) | Artifact Size (bytes) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(compare.get("ranked", []), start=1):
        med = row.get("medians") or {}
        lines.append(
            f"| {idx} | {row.get('ssg')} | {_fmt_md_num(med.get('cpu_seconds'))} | "
            f"{_fmt_md_num(med.get('wall_seconds'))} | {_fmt_md_num(med.get('joules'))} | "
            f"{_fmt_md_num(med.get('artifact_size_bytes'))} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_md_num(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value == 0:
            return "0.00000000"
        if abs(value) < 0.0005:
            return f"{value:.12f}".rstrip("0").rstrip(".")
        return f"{value:.8f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _load_prompt_text(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise SystemExit("Pass either --prompt or --prompt-file, not both.")
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    if prompt is not None:
        return str(prompt).strip()
    return ""


def _canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(data)).hexdigest()


def _overwrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _load_replay_fixture(path: Path) -> dict[str, Any]:
    fixture = load_json(path)
    if not isinstance(fixture.get("spec"), dict):
        raise SystemExit("Replay fixture is missing object field: spec")
    if not isinstance(fixture.get("llm_calls"), list) or not fixture["llm_calls"]:
        raise SystemExit("Replay fixture must include non-empty array field: llm_calls")
    return fixture


def _build_replay_calls(
    fixture_calls: list[dict[str, Any]],
    session: str,
    step: str,
    replay_fixture: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_fixture_str = str(replay_fixture)
    for idx, raw in enumerate(fixture_calls):
        prompt_tokens = int(raw.get("prompt_tokens", 0))
        completion_tokens = int(raw.get("completion_tokens", 0))
        row = {
            "metric_type": "llm_call",
            "timestamp_utc": str(raw.get("timestamp_utc", "1970-01-01T00:00:00+00:00")),
            "session": session,
            "step": str(raw.get("step", step)),
            "model": str(raw.get("model", "replay-model")),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_chars": int(raw["prompt_chars"]) if raw.get("prompt_chars") is not None else None,
            "mode": "replay",
            "replay_fixture": replay_fixture_str,
            "call_index": idx,
        }
        rows.append(row)
    return rows


def _validate_spec_or_exit(spec: dict[str, Any]) -> dict[str, Any]:
    validation = validate_spec_schema(spec=spec, supported_ssgs=SUPPORTED_SSGS)
    if validation.get("schema_validation_pass") is True:
        return validation

    lines = ["Spec validation failed:"]
    for err in validation.get("errors", []):
        lines.append(f"- [{err.get('code')}] {err.get('path')}: {err.get('message')}")
    raise SystemExit("\n".join(lines))


def _generate_spec_from_replay(
    prompt_text: str,
    replay_fixture: Path,
    session: str,
    step: str,
    llm_log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _load_replay_fixture(replay_fixture)
    fixture_prompt = str(fixture.get("prompt", "")).strip()
    prompt_matches_fixture = None
    if prompt_text:
        prompt_matches_fixture = prompt_text == fixture_prompt

    spec = fixture["spec"]
    validation = _validate_spec_or_exit(spec)
    replay_calls = _build_replay_calls(
        fixture_calls=fixture["llm_calls"],
        session=session,
        step=step,
        replay_fixture=replay_fixture,
    )
    _overwrite_jsonl(llm_log_path, replay_calls)
    llm_summary = aggregate_llm_calls(llm_log_path, session=session)

    telemetry_payload = {
        "metric_type": "spec_from_prompt",
        "mode": "replay",
        "session": session,
        "step": step,
        "prompt_text": fixture_prompt if not prompt_text else prompt_text,
        "prompt_matches_fixture": prompt_matches_fixture,
        "replay_fixture": str(replay_fixture),
        "llm_calls": replay_calls,
        "llm_calls_sha256": _sha256_json(replay_calls),
        "llm_summary": llm_summary,
        "spec_sha256": _sha256_json(spec),
        "schema_validation_pass": validation.get("schema_validation_pass"),
        "schema_error_count": validation.get("schema_error_count"),
        "schema_error_types": validation.get("schema_error_types"),
        "notes": [
            "Replay mode writes deterministic spec and telemetry outputs for reproducible audits.",
        ],
    }
    return spec, telemetry_payload


def _session_slug(session: str) -> str:
    return slug_to_path(session.lower().replace(" ", "-"))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "spec-from-prompt":
        prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
        if args.mode == "live":
            raise SystemExit(
                "Live LLM mode is not configured in this prototype. Use --mode replay with --replay-fixture."
            )
        if not args.replay_fixture:
            raise SystemExit("--replay-fixture is required when --mode replay.")

        spec, telemetry_payload = _generate_spec_from_replay(
            prompt_text=prompt_text,
            replay_fixture=args.replay_fixture,
            session=args.session,
            step=args.step,
            llm_log_path=args.llm_log,
        )
        dump_json(args.out_spec, spec)
        dump_json(args.out_telemetry, telemetry_payload)
        print(f"Spec generated from replay fixture: {args.out_spec}")
        print(f"Deterministic telemetry written: {args.out_telemetry}")
        print(f"LLM calls JSONL (overwritten): {args.llm_log}")
        return

    if args.cmd == "run-case":
        if args.page_multiplier < 1:
            raise SystemExit("--page-multiplier must be >= 1.")
        prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
        if args.mode == "live":
            raise SystemExit(
                "Live LLM mode is not configured in this prototype. Use --mode replay with --replay-fixture."
            )
        if not args.replay_fixture:
            raise SystemExit("--replay-fixture is required when --mode replay.")

        session_slug = _session_slug(args.session)
        reports_dir = args.reports_dir
        spec_path = reports_dir / f"{session_slug}_generated_spec.json"
        telemetry_path = reports_dir / f"{session_slug}_replay_telemetry.json"
        llm_log_path = reports_dir / f"{session_slug}_llm_calls.jsonl"
        scaffold_manifest_path = reports_dir / f"{session_slug}_scaffold.json"
        build_path = reports_dir / f"{session_slug}_build.json"
        runtime_path = reports_dir / f"{session_slug}_runtime.json"
        report_json_path = reports_dir / f"{session_slug}_report.json"
        report_md_path = reports_dir / f"{session_slug}_report.md"
        manifest_path = reports_dir / f"{session_slug}_case_run_manifest.json"

        spec, telemetry = _generate_spec_from_replay(
            prompt_text=prompt_text,
            replay_fixture=args.replay_fixture,
            session=args.session,
            step="spec_generation",
            llm_log_path=llm_log_path,
        )
        dump_json(spec_path, spec)
        dump_json(telemetry_path, telemetry)

        selected_ssg = args.ssg or str((spec.get("ssg") or {}).get("preferred", "hugo")).lower()
        if selected_ssg not in SUPPORTED_SSGS:
            raise SystemExit(f"Unsupported selected SSG: {selected_ssg}")

        project_dir = args.out_root / f"{session_slug}_{selected_ssg}"
        scaffold_result = scaffold_project(
            spec=spec,
            out_dir=project_dir,
            ssg=selected_ssg,
            force=args.force,
            spec_base_dir=args.replay_fixture.parent,
            page_multiplier=args.page_multiplier,
        )
        dump_json(scaffold_manifest_path, scaffold_result)

        build_command = args.build_command or DEFAULT_BUILD_COMMANDS[selected_ssg]
        build_shell = False
        eleventy_setup: dict[str, Any] | None = None
        if selected_ssg == "eleventy":
            npm_diag = _resolve_npm_diagnostics()
            eleventy_setup = _ensure_eleventy_dependencies(project_dir, npm_diag)
            invocation = _resolve_compare_build_invocation(selected_ssg, project_dir, npm_diag)
            build_command = str(invocation["command"])
            build_shell = bool(invocation["shell"])

        build_metric = run_build_measurement(
            command=build_command,
            project_dir=project_dir,
            cpu_watt_assumption=args.system_watts,
            shell=build_shell,
            ssg=selected_ssg,
            artifact_dir=args.artifact_dir,
            capture_full_output=(selected_ssg == "eleventy"),
        )
        if eleventy_setup is not None:
            build_metric["eleventy_setup"] = {
                "executed": bool(eleventy_setup.get("executed")),
                "exit_code": int(eleventy_setup.get("exit_code", 0)),
                "stdout_tail": str(eleventy_setup.get("stdout", ""))[-2500:],
                "stderr_tail": str(eleventy_setup.get("stderr", ""))[-2500:],
                "resolved_executable": eleventy_setup.get("resolved_executable"),
            }
        dump_json(build_path, build_metric)

        if args.artifact_dir is not None:
            runtime_artifact_dir = (
                args.artifact_dir if args.artifact_dir.is_absolute() else (project_dir / args.artifact_dir)
            )
        else:
            runtime_artifact_dir = _artifact_dir_for_ssg(project_dir, selected_ssg)

        if runtime_artifact_dir.exists() and runtime_artifact_dir.is_dir():
            runtime_metric = measure_runtime_comparison(
                static_url=None,
                dynamic_url=None,
                static_dir=runtime_artifact_dir,
                dynamic_local=True,
                runs=args.runtime_runs,
                timeout_sec=args.timeout_sec,
            )
        else:
            runtime_metric = {
                "metric_type": "runtime",
                "measured_at_utc": now_iso(),
                "mode": {
                    "static_source": "local_dir",
                    "dynamic_source": "local_dynamic_server",
                },
                "static": None,
                "dynamic": None,
                "comparison": None,
                "notes": [
                    f"Runtime measurement skipped because artifact directory was not found: {runtime_artifact_dir}"
                ],
            }
        dump_json(runtime_path, runtime_metric)

        llm_summary = aggregate_llm_calls(llm_log_path, session=args.session)
        combined = combine_report(
            spec=spec,
            scaffold=scaffold_result,
            build=build_metric,
            runtime=runtime_metric,
            llm=llm_summary,
            compare=None,
            workflow_mode="single_ssg",
        )
        md = render_markdown_summary(combined)
        dump_json(report_json_path, combined)
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(md, encoding="utf-8")

        manifest = {
            "metric_type": "case_run_manifest",
            "workflow_mode": "single_ssg",
            "session": args.session,
            "selected_ssg": selected_ssg,
            "schema_validation_pass": telemetry.get("schema_validation_pass"),
            "schema_error_count": telemetry.get("schema_error_count"),
            "schema_error_types": telemetry.get("schema_error_types"),
            "paths": {
                "spec": str(spec_path),
                "telemetry": str(telemetry_path),
                "llm_log": str(llm_log_path),
                "scaffold": str(scaffold_manifest_path),
                "build": str(build_path),
                "runtime": str(runtime_path),
                "report_json": str(report_json_path),
                "report_md": str(report_md_path),
                "project_dir": str(project_dir),
            },
            "notes": [
                "Single-SSG run-case flow completed.",
                "Use compare-ssg separately for evaluation-mode multi-SSG ranking.",
            ],
        }
        dump_json(manifest_path, manifest)
        print(f"Run-case completed for session `{args.session}`.")
        print(f"Case manifest: {manifest_path}")
        print(f"Report JSON: {report_json_path}")
        print(f"Report Markdown: {report_md_path}")
        return

    if args.cmd == "scaffold":
        if args.page_multiplier < 1:
            raise SystemExit("--page-multiplier must be >= 1.")
        spec = load_spec(args.spec)
        _validate_spec_or_exit(spec)
        selected_ssg = args.ssg or str((spec.get("ssg") or {}).get("preferred", "hugo")).lower()
        result = scaffold_project(
            spec=spec,
            out_dir=args.out,
            ssg=selected_ssg,
            force=args.force,
            spec_base_dir=args.spec.parent,
            page_multiplier=args.page_multiplier,
        )
        dump_json(args.manifest, result)
        print(f"Scaffold created with {selected_ssg}: {args.out}")
        print(f"Manifest: {args.manifest}")
        return

    if args.cmd == "measure-build":
        inferred_ssg = args.ssg or _infer_ssg_from_project(args.project)
        selected_command = args.build_command
        if not selected_command and inferred_ssg:
            selected_command = DEFAULT_BUILD_COMMANDS[inferred_ssg]
        if not selected_command:
            raise SystemExit(
                "Unable to determine build command. Pass --build-command or --ssg for a known default."
            )

        metric = run_build_measurement(
            command=selected_command,
            project_dir=args.project,
            cpu_watt_assumption=args.system_watts,
            shell=args.shell,
            ssg=inferred_ssg,
            artifact_dir=args.artifact_dir,
        )
        dump_json(args.out, metric)
        print(f"Build metric written: {args.out}")
        if metric.get("warnings"):
            print("Build warnings:")
            for warning in metric["warnings"]:
                print(f"- {warning}")
        if metric["exit_code"] != 0:
            raise SystemExit(metric["exit_code"])
        return

    if args.cmd == "record-llm":
        row = record_llm_call(
            log_path=args.log,
            session=args.session,
            step=args.step,
            model=args.model,
            prompt_tokens=args.prompt_tokens,
            completion_tokens=args.completion_tokens,
            prompt_chars=args.prompt_chars,
        )
        print(f"LLM call recorded: {args.log} ({row['total_tokens']} tokens)")
        return

    if args.cmd == "measure-runtime":
        if args.dynamic_local and not args.static_dir:
            raise SystemExit("--dynamic-local requires --static-dir.")
        metric = measure_runtime_comparison(
            static_url=args.static_url,
            static_dir=args.static_dir,
            dynamic_url=args.dynamic_url,
            dynamic_local=args.dynamic_local,
            local_host=args.local_host,
            runs=args.runs,
            timeout_sec=args.timeout_sec,
        )
        dump_json(args.out, metric)
        print(f"Runtime metric written: {args.out}")
        return

    if args.cmd == "compare-ssg":
        if args.page_multiplier < 1:
            raise SystemExit("--page-multiplier must be >= 1.")
        spec = load_spec(args.spec)
        _validate_spec_or_exit(spec)
        candidates = ((spec.get("ssg") or {}).get("candidates")) or []
        if not candidates:
            raise SystemExit("Spec has no ssg.candidates list.")
        npm_diag = _resolve_npm_diagnostics()

        candidate_rows: list[dict[str, Any]] = []
        for raw in candidates:
            ssg = str(raw).lower()
            if ssg not in SUPPORTED_SSGS:
                continue

            project_dir = args.out_root / ssg
            scaffold_result = scaffold_project(
                spec=spec,
                out_dir=project_dir,
                ssg=ssg,
                force=args.force,
                spec_base_dir=args.spec.parent,
                page_multiplier=args.page_multiplier,
            )
            eleventy_install_after_scaffold: dict[str, Any] | None = None
            if ssg == "eleventy":
                eleventy_install_after_scaffold = _ensure_eleventy_dependencies(project_dir, npm_diag)
                if int(eleventy_install_after_scaffold.get("exit_code", 1)) != 0:
                    print("Eleventy diagnostics: npm install failed after scaffold.")
                    print(f"resolved_executable={eleventy_install_after_scaffold.get('resolved_executable')}")
                    print(f"PATH={eleventy_install_after_scaffold.get('path')}")
                    print(f"stderr={str(eleventy_install_after_scaffold.get('stderr', ''))[-500:]}")

            build_invocation = _resolve_compare_build_invocation(ssg, project_dir, npm_diag)
            build_command = str(build_invocation["command"])
            build_shell = bool(build_invocation["shell"])
            resolved_executable = build_invocation.get("resolved_executable")
            runs: list[dict[str, Any]] = []
            if ssg == "eleventy" and not npm_diag.get("found"):
                print("Eleventy diagnostics: npm executable not found.")
                print(f"PATH={npm_diag.get('path')}")
                print(f"Resolution attempts: {npm_diag.get('attempts')}")
            for _ in range(args.runs):
                artifact_dir = _artifact_dir_for_ssg(project_dir, ssg)
                pre_snapshot = _artifact_snapshot(artifact_dir)
                cleanup_error: str | None = None
                eleventy_setup: dict[str, Any] | None = None

                if artifact_dir.exists():
                    try:
                        shutil.rmtree(artifact_dir)
                    except OSError as exc:
                        cleanup_error = str(exc)

                deleted_before_build = not artifact_dir.exists()
                if ssg == "eleventy":
                    eleventy_setup = eleventy_install_after_scaffold

                if eleventy_setup is not None and int(eleventy_setup.get("exit_code", 1)) != 0:
                    metric = {
                        "metric_type": "build",
                        "started_at_utc": now_iso(),
                        "finished_at_utc": now_iso(),
                        "ssg": ssg,
                        "build_pid": None,
                        "observed_process_tree_pids": [],
                        "process_tree_cpu": None,
                        "command": build_command,
                        "project_dir": str(project_dir),
                        "exit_code": int(eleventy_setup["exit_code"]),
                        "wall_seconds": 0.0,
                        "cpu_seconds": None,
                        "cpu_seconds_display": None,
                        "energy_proxy": {
                            "label": "energy_seconds_x_assumed_system_watts",
                            "energy_basis": "wall_time_seconds",
                            "energy_seconds": 0.0,
                            "assumed_system_watts": args.system_watts,
                            "joules": 0.0,
                            "joules_display": _fmt_md_num(0.0),
                        },
                        "build_artifact": {
                            "path": str(artifact_dir),
                            "exists": False,
                            "size_bytes": None,
                            "file_count": 0,
                        },
                        "warnings": [],
                        "notes": ["Eleventy dependency installation failed before build."],
                        "stdout_tail": eleventy_setup.get("stdout", "")[-2500:],
                        "stderr_tail": eleventy_setup.get("stderr", "")[-2500:],
                        "stdout_full": eleventy_setup.get("stdout", ""),
                        "stderr_full": eleventy_setup.get("stderr", ""),
                    }
                else:
                    metric = run_build_measurement(
                        command=build_command,
                        project_dir=project_dir,
                        cpu_watt_assumption=args.system_watts,
                        shell=build_shell,
                        ssg=ssg,
                        capture_full_output=(ssg == "eleventy"),
                    )
                metric["resolved_executable"] = resolved_executable
                metric["path_environment"] = os.environ.get("PATH", "")
                post_snapshot = _artifact_snapshot(artifact_dir)
                artifact_exists_after_build = post_snapshot is not None
                artifact_modified_after_build = (
                    post_snapshot is not None and post_snapshot != pre_snapshot
                )

                invalid_reasons: list[str] = []
                if cleanup_error is not None:
                    invalid_reasons.append("artifact_cleanup_failed")
                if not deleted_before_build:
                    invalid_reasons.append("artifact_not_deleted_before_build")
                if not artifact_exists_after_build:
                    invalid_reasons.append("artifact_missing_after_build")
                if not artifact_modified_after_build:
                    invalid_reasons.append("artifact_unchanged_after_build")
                if eleventy_setup is not None and int(eleventy_setup.get("exit_code", 0)) != 0:
                    invalid_reasons.append("eleventy_npm_install_failed")
                if int(metric.get("exit_code", 1)) != 0:
                    invalid_reasons.append("build_failed")

                if eleventy_setup is not None:
                    metric["eleventy_setup"] = {
                        "executed": bool(eleventy_setup.get("executed")),
                        "exit_code": int(eleventy_setup.get("exit_code", 0)),
                        "resolved_executable": eleventy_setup.get("resolved_executable"),
                        "path": eleventy_setup.get("path"),
                        "stdout_tail": str(eleventy_setup.get("stdout", ""))[-2500:],
                        "stderr_tail": str(eleventy_setup.get("stderr", ""))[-2500:],
                        "stdout_full": str(eleventy_setup.get("stdout", "")),
                        "stderr_full": str(eleventy_setup.get("stderr", "")),
                    }
                metric["clean_build"] = {
                    "artifact_dir": str(artifact_dir),
                    "pre_snapshot": pre_snapshot,
                    "post_snapshot": post_snapshot,
                    "cleanup_error": cleanup_error,
                    "deleted_before_build": deleted_before_build,
                    "artifact_exists_after_build": artifact_exists_after_build,
                    "artifact_modified_after_build": artifact_modified_after_build,
                    "is_valid": len(invalid_reasons) == 0,
                    "invalid_reasons": invalid_reasons,
                }
                runs.append(metric)

            successful = [r for r in runs if int(r.get("exit_code", 1)) == 0]
            valid_runs = [r for r in runs if ((r.get("clean_build") or {}).get("is_valid") is True)]
            source_rows = valid_runs
            wall_vals = [r.get("wall_seconds") for r in source_rows]
            cpu_vals = [r.get("cpu_seconds") for r in source_rows]
            joule_vals = [((r.get("energy_proxy") or {}).get("joules")) for r in source_rows]
            artifact_vals = [((r.get("build_artifact") or {}).get("size_bytes")) for r in source_rows]

            candidate_rows.append(
                {
                    "ssg": ssg,
                    "project_dir": str(project_dir),
                    "build_command": build_command,
                    "build_shell": build_shell,
                    "resolved_executable": resolved_executable,
                    "runs_requested": int(args.runs),
                    "runs_successful": len(successful),
                    "runs_valid": len(valid_runs),
                    "scaffold_files_written": len(scaffold_result.get("files_written", [])),
                    "generated_page_count": int(scaffold_result.get("generated_page_count", 0)),
                    "medians": {
                        "wall_seconds": _compute_median(wall_vals),
                        "cpu_seconds": _compute_median(cpu_vals),
                        "joules": _compute_median(joule_vals),
                        "artifact_size_bytes": _compute_median(artifact_vals),
                    },
                    "run_metrics": runs,
                }
            )

        ranked = sorted(candidate_rows, key=_rank_key)
        compare_payload = {
            "metric_type": "ssg_compare",
            "spec_path": str(args.spec),
            "runs_per_ssg": int(args.runs),
            "page_multiplier": int(args.page_multiplier),
            "environment_diagnostics": {
                "PATH": os.environ.get("PATH", ""),
                "npm": npm_diag,
                "npx_resolved": shutil.which("npx.cmd" if os.name == "nt" else "npx"),
            },
            "ranking_policy": {
                "primary": "median_energy_proxy_joules",
                "secondary": "artifact_size_bytes",
            },
            "ranked": ranked,
        }
        dump_json(args.out, compare_payload)
        print(f"SSG comparison written: {args.out}")
        print(_render_compare_markdown(compare_payload))
        return

    if args.cmd == "report":
        spec = load_spec(args.spec) if args.spec else None
        if spec is not None:
            _validate_spec_or_exit(spec)
        scaffold = _maybe_load(args.scaffold)
        build = _maybe_load(args.build)
        runtime = _maybe_load(args.runtime)
        compare = _maybe_load(args.compare)
        llm = aggregate_llm_calls(args.llm_log, session=args.session) if args.llm_log else None

        combined = combine_report(
            spec=spec,
            scaffold=scaffold,
            build=build,
            runtime=runtime,
            llm=llm,
            compare=compare,
        )
        md = render_markdown_summary(combined)
        dump_json(args.out_json, combined)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"Combined report JSON: {args.out_json}")
        print(f"Summary Markdown: {args.out_md}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
