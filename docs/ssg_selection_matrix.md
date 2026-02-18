# Initial SSG Comparison Matrix

Purpose: choose one final generator after pilot measurements and pragmatic constraints.

## Criteria
- Build speed (measured wall + CPU time)
- Setup complexity for newcomers from dynamic/WYSIWYG workflows
- Template flexibility for layout/nav/frontmatter automation
- Documentation maturity and maintenance
- Deployment fit for static hosting targets

## Candidate snapshot
| SSG | Strengths | Risks |
|---|---|---|
| Hugo | Very fast builds, single binary, strong templates | Go template syntax learning curve |
| Eleventy | JS ecosystem, flexible data/layout model | Node/npm toolchain overhead |
| Zola | Good defaults, single binary, clean setup | Smaller ecosystem than Hugo/Eleventy |
| Pelican | Python ecosystem familiarity | Lower momentum in some workflows |

## Recommended initial baseline
Use Hugo as baseline for final prototype unless pilot measurements in your environment strongly contradict:
- usually lowest build-time overhead
- simple binary-based CI/deployment
- straightforward content/frontmatter model for migration wrappers

## How to finalize decision
1. Generate same scenario spec across all 4 SSGs.
2. Run `measure-build` 5x each and compare average + variance.
3. Report qualitative setup friction (install/build/debug steps).
4. Select one SSG with quantitative + qualitative justification.

