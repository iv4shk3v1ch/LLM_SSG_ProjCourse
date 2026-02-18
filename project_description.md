# Project: LLM-Assisted Static Site Generation with Energy-Aware Evaluation

This project investigates how LLM assistance can reduce adoption friction when users move from dynamic/WYSIWYG websites to static site generators (SSGs). The prototype focuses on generating project structure and configuration, not article/page prose. It outputs scaffolded static-site projects (config files, directory structures, templates, navigation partials, frontmatter stubs, and placeholder pages) from a compact site specification.

The research question is not only usability, but full-stack energy trade-offs. The evaluation pipeline records:
- SSG build cost (wall time, CPU time, and clearly labeled energy proxy),
- LLM usage proxy (prompt/completion tokens, call counts, and step-level usage),
- runtime website efficiency proxy (static vs dynamic latency and transfer-size comparison).

The tool is implemented as a CLI to support reproducible experiments. It supports multiple SSG candidates (Eleventy, Hugo, Zola, Pelican) for early-stage comparison and includes three demo scenarios:
1. portfolio migration,
2. research group website,
3. marketing landing page.

Migration assistance is included through mapping-plan generation and wrapper creation for user-provided existing text, without creating new AI-generated content.

Outputs include machine-readable JSON artifacts and a human-readable Markdown summary to support case-study analysis and reporting. The final phase will compare candidate generators on measured build characteristics and setup friction, then select one SSG for a refined end-to-end prototype and demo conversion.

