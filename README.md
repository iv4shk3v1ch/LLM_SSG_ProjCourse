# LLM-Assisted SSG Case-Study Prototype

Prototype CLI for a CS Project Course case study on energy-aware static site generation.

## Scope
- Generates SSG scaffolds from a site spec.
- Excludes AI content generation (uses placeholders only).
- Records build/runtime/LLM-usage metrics.
- Produces JSON + Markdown reports.

## Supported SSGs
- Eleventy
- Hugo
- Zola
- Pelican

## Install
```powershell
pip install -e .
```

## Quick start
```powershell
 --out demos/portfolio_hugo --ssg hugo
ssg-case record-llm --log reports/llm_calls.jsonl --session portfolio --step scaffold --model gpt-4.1 --prompt-tokens 1200 --completion-tokens 300
ssg-case measure-build --project demos/portfolio_hugo --build-command "hugo" --out reports/portfolio_build.json
ssg-case measure-runtime --static-url https://example-static.site --dynamic-url https://example-dynamic.site --runs 5 --out reports/portfolio_runtime.json
ssg-case report --spec scenarios/portfolio_migration.json --build reports/portfolio_build.json --runtime reports/portfolio_runtime.json --llm-log reports/llm_calls.jsonl --session portfolio --out-json reports/portfolio_report.json --out-md reports/portfolio_report.md
```

## CLI overview
- `scaffold`: generate SSG project structure/config/templates/placeholders.
- `measure-build`: run build command and log wall time, CPU time, energy proxy.
- `record-llm`: append one LLM call record (tokens/call count proxy).
- `measure-runtime`: compare static vs dynamic runtime transfer/latency proxies.
- `report`: merge all artifacts into machine/human-readable summaries.

## Notes
- Build energy uses a clearly labeled proxy: `CPU time (s) * assumed CPU watts`.
- Runtime comparison is network-level proxy (latency + transfer bytes), not full RAPL hardware power telemetry.
- For Windows/macOS/Linux energy telemetry, extend `instrumentation.py` with platform tools.
