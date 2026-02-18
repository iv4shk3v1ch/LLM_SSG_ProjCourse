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

## Prerequisites
- Windows PowerShell
- Python 3.10+

## Option A: Run Without Installing (recommended for audit)
```powershell
cd C:\Users\ivash\LLM_SSG_ProjCourse
python -m ssg_case_tool.cli -h
```

## Option B: Editable Install
```powershell
python -m pip install -e .
ssg-case -h
```

If `pip install -e .` fails with temp/permission errors on Windows, use Option A above and run via `python -m ssg_case_tool.cli ...`.

## Step-by-Step Audit (PowerShell)
1. Check CLI entrypoint:
```powershell
python -m ssg_case_tool.cli -h
```

2. Generate a scaffold (portfolio scenario):
```powershell
python -m ssg_case_tool.cli scaffold --spec scenarios/portfolio_migration.json --out demos/portfolio_hugo --ssg hugo --manifest reports/portfolio_scaffold.json --force
```

3. Confirm generated project files:
```powershell
Get-ChildItem -Recurse demos/portfolio_hugo
```

4. Log one LLM call proxy row:
```powershell
python -m ssg_case_tool.cli record-llm --log reports/llm_calls.jsonl --session portfolio --step scaffold --model gpt-4.1 --prompt-tokens 1200 --completion-tokens 300
```

5. Measure a build command:
```powershell
python -m ssg_case_tool.cli measure-build --project demos/portfolio_hugo --ssg hugo --out reports/portfolio_build.json
```

Defaults per SSG:
- `hugo`: `hugo --minify`
- `eleventy`: `npx @11ty/eleventy`
- `zola`: `zola build`
- `pelican`: `pelican content -o output -s pelicanconf.py`

These commands require the corresponding tool to be installed and available on `PATH`.

Override default command:
```powershell
python -m ssg_case_tool.cli measure-build --project demos/portfolio_hugo --build-command "hugo --minify --gc" --out reports/portfolio_build.json
```

If you pass a placeholder command like `echo`, the tool records warnings in JSON and prints them in CLI output.

6. Measure runtime proxy (example endpoint):
```powershell
python -m ssg_case_tool.cli measure-runtime --static-url https://example.com --runs 3 --out reports/portfolio_runtime.json
```

Local runtime measurement from build artifact directory:
```powershell
python -m ssg_case_tool.cli measure-runtime --static-dir demos/portfolio_hugo/public --runs 5 --out reports/portfolio_runtime.json
```

Local static vs local dynamic comparison (same HTML payload):
```powershell
python -m ssg_case_tool.cli measure-runtime --static-dir demos/portfolio_hugo/public --dynamic-local --runs 5 --out reports/portfolio_runtime_compare.json
```

The command starts temporary local servers on random free ports, runs latency/TTFB/transfer tests, and shuts servers down automatically.

7. Generate combined report:
```powershell
python -m ssg_case_tool.cli report --spec scenarios/portfolio_migration.json --scaffold reports/portfolio_scaffold.json --build reports/portfolio_build.json --runtime reports/portfolio_runtime.json --llm-log reports/llm_calls.jsonl --session portfolio --out-json reports/portfolio_report.json --out-md reports/portfolio_report.md
```

8. Review report outputs:
```powershell
Get-Content reports/portfolio_report.md
Get-Content reports/portfolio_report.json
```

## Quick command examples
```powershell
python -m ssg_case_tool.cli scaffold --spec scenarios/portfolio_migration.json --out demos/portfolio_hugo --ssg hugo
python -m ssg_case_tool.cli record-llm --log reports/llm_calls.jsonl --session portfolio --step scaffold --model gpt-4.1 --prompt-tokens 1200 --completion-tokens 300
python -m ssg_case_tool.cli measure-build --project demos/portfolio_hugo --ssg hugo --out reports/portfolio_build.json
python -m ssg_case_tool.cli measure-runtime --static-url https://example-static.site --dynamic-url https://example-dynamic.site --runs 5 --out reports/portfolio_runtime.json
python -m ssg_case_tool.cli measure-runtime --static-dir demos/portfolio_hugo/public --dynamic-local --runs 5 --out reports/portfolio_runtime_local.json
python -m ssg_case_tool.cli report --spec scenarios/portfolio_migration.json --build reports/portfolio_build.json --runtime reports/portfolio_runtime.json --llm-log reports/llm_calls.jsonl --session portfolio --out-json reports/portfolio_report.json --out-md reports/portfolio_report.md
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
