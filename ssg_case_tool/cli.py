from __future__ import annotations

import argparse
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
from .utils import dump_json, load_json, load_spec, now_iso

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

    p_report = sub.add_parser("report", help="Assemble final JSON+Markdown report.")
    p_report.add_argument("--spec", type=Path, help="Spec path.")
    p_report.add_argument("--scaffold", type=Path, help="Scaffold manifest JSON.")
    p_report.add_argument("--build", type=Path, help="Build measurement JSON.")
    p_report.add_argument("--runtime", type=Path, help="Runtime measurement JSON.")
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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "scaffold":
        if args.page_multiplier < 1:
            raise SystemExit("--page-multiplier must be >= 1.")
        spec = load_spec(args.spec)
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
        scaffold = _maybe_load(args.scaffold)
        build = _maybe_load(args.build)
        runtime = _maybe_load(args.runtime)
        llm = aggregate_llm_calls(args.llm_log, session=args.session) if args.llm_log else None

        combined = combine_report(spec=spec, scaffold=scaffold, build=build, runtime=runtime, llm=llm)
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
