from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .instrumentation import (
    aggregate_llm_calls,
    measure_runtime_comparison,
    record_llm_call,
    run_build_measurement,
)
from .reporting import combine_report, render_markdown_summary
from .scaffold import SUPPORTED_SSGS, scaffold_project
from .utils import dump_json, load_json, load_spec


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
    p_scaffold.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p_scaffold.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/scaffold_manifest.json"),
        help="Where to write scaffold manifest JSON.",
    )

    p_build = sub.add_parser("measure-build", help="Run build and record duration/cpu/energy proxy.")
    p_build.add_argument("--project", required=True, type=Path, help="Project directory.")
    p_build.add_argument("--build-command", required=True, help="Build command, e.g. 'hugo'.")
    p_build.add_argument("--cpu-watts", type=float, default=35.0, help="CPU watts for energy proxy.")
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
    p_runtime.add_argument("--static-url", required=True, help="Static hosted site URL.")
    p_runtime.add_argument("--dynamic-url", help="Dynamic site URL.")
    p_runtime.add_argument("--runs", type=int, default=5, help="Number of runs per URL.")
    p_runtime.add_argument("--timeout-sec", type=float, default=10.0, help="Request timeout per run.")
    p_runtime.add_argument("--out", required=True, type=Path, help="Runtime metric output JSON.")

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "scaffold":
        spec = load_spec(args.spec)
        selected_ssg = args.ssg or str((spec.get("ssg") or {}).get("preferred", "hugo")).lower()
        result = scaffold_project(
            spec=spec,
            out_dir=args.out,
            ssg=selected_ssg,
            force=args.force,
            spec_base_dir=args.spec.parent,
        )
        dump_json(args.manifest, result)
        print(f"Scaffold created with {selected_ssg}: {args.out}")
        print(f"Manifest: {args.manifest}")
        return

    if args.cmd == "measure-build":
        metric = run_build_measurement(
            command=args.build_command,
            project_dir=args.project,
            cpu_watt_assumption=args.cpu_watts,
            shell=args.shell,
        )
        dump_json(args.out, metric)
        print(f"Build metric written: {args.out}")
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
        metric = measure_runtime_comparison(
            static_url=args.static_url,
            dynamic_url=args.dynamic_url,
            runs=args.runs,
            timeout_sec=args.timeout_sec,
        )
        dump_json(args.out, metric)
        print(f"Runtime metric written: {args.out}")
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
