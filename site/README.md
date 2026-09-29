# site/ — HybridCUA project page

Static single-page site built from the paper in `projects/HybridCUA-arxiv`.
**No dependencies, no CDN, no build step** — three files plus figures.

```
site/
├── index.html        markup and copy
├── style.css         design tokens + layout (light/dark)
├── app.js            tables, KPI tiles and the rollout step viewer
├── leaderboard.json  ALL numbers live here — the only file to edit for results
├── .nojekyll         tells GitHub Pages to serve files as-is
└── assets/           the paper's figures, plus cases/ rollout screenshots
```

Division of labour: **figures come from the paper as-is; tables are rebuilt as
HTML.** A chart the paper already drew is never redrawn here — that would let the
two drift apart.

## Local preview

`app.js` `fetch()`es `leaderboard.json`, which `file://` blocks — serve over HTTP:

```bash
python3 -m http.server 8000 --directory site
# then open http://localhost:8000/
```

## Updating results

Edit **`leaderboard.json` only**; `index.html` hardcodes no numbers. Keys:

| Key | Drives |
|---|---|
| `headline` | the four KPI tiles in the hero |
| `main_results` | the OSWorld table (grouped; `indent`, `rule`, `highlight`, `*_delta` per row) |
| `ood_results` | the OSWorld-MCP / WindowsAgentArena table |
| `sft_ablation`, `schema_ablation` | the two ablation tables |
| `cli_exposure` | the paired accuracy chart + CLI step-share chart + its table view |
| `domain_results` | the per-domain gain/step-share chart + its table view |
| `operation_share` | the operation-category stacked bars |
| `corpus` | trajectory counts quoted in the Method prose (not currently charted) |

Best-in-column bolding, deltas, and the `→` arrows are computed at render time.
Training hyperparameters are in the `SPEC` constant in `app.js` (they are prose,
not results). `null` renders as `–`.

Figures 2 and 6 additionally carry a `<details>` table view of the plotted
numbers, so the data is reachable without reading values off an image.

## Colour

The two interface colours (`--cli` blue, `--gui` neutral) and the accents come
from a palette validated in both modes; interface identity in the UI is always
carried by a label or a border edge as well, never colour alone. If you change
`--series-*` in `style.css`, re-run:

```bash
node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --surface "#ffffff" --pairs all
node scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark  --surface "#17181b" --pairs all
```

## Assets

**Every figure on the page is the paper's own figure** — the page never redraws
a plot the paper already has. Tables are rebuilt as HTML (sortable, themed,
accessible); figures are shipped as PNG.

| `assets/` | source in `figures/` | used in |
|---|---|---|
| `teaser.png` | `intro.png` | hero (Fig. 1) |
| `problem.png` | `GUI_CLI.png` | Problem (Fig. 2) |
| `pipeline.png` | `method.png` | Method (Fig. 3) |
| `data-composition.png` | `traj-distribution.png` | Method (Fig. 4) |
| `rl-ablation.png` | `training_ablation.png` | Results (Fig. 5) |
| `domain-results.png` | `domain_results.png` | Analysis (Fig. 6) |
| `operation-share.png` | `stats.png` | Analysis (Fig. 7) |
| `case-study.png` | `case_study.png` | Analysis (Fig. 8) |
| `traj-gui-only.png` | `gui-only-case.png` | Trajectory types (Fig. 9) |
| `traj-cli-only.png` | `cli-only-case.png` | Trajectory types (Fig. 10) |
| `traj-hybrid.png` | `hybrid-case.png` | Trajectory types (Fig. 11) |
| `cases/<name>/step_N.jpg` | `case_studies/<name>/step_N.png` | Rollouts |

PNGs are downscaled to 1800px max width. The 28 rollout screenshots are JPEG
q82 (1.6 MB total instead of 7.4 MB as PNG) since they are photographic
screenshots, not line art.

## Rollout step viewer

The `cases` array in `leaderboard.json` drives the step-by-step viewer. Per case:
`dir` (the folder under `assets/cases/`), `title`, `pattern`, `flow`, `task`,
the `gui_note` / `cli_note` / `analysis` prose (inline HTML allowed), and `steps`
— one entry per **action** step, each `"GUI" | "CLI" | "Control"`.

The image count must be `steps.length + 1`: `step_0 … step_{N-1}` are the
pre-action screenshots and `step_N` is the final observation, rendered as the
`✓` button. Verify after adding a case:

```bash
python3 -c "
import json,os
for c in json.load(open('leaderboard.json'))['cases']:
    have=len([f for f in os.listdir(f'assets/cases/{c[\"dir\"]}') if f.endswith('.jpg')])
    print(c['dir'], len(c['steps'])+1 == have)"
```

## Deploying

`.github/workflows/deploy-pages.yml` publishes this directory to `gh-pages` on
every push to `main` touching `site/**`. One-time repo setup:

1. **Settings → Actions → General → Workflow permissions**: *Read and write*.
2. Push once and let the workflow create `gh-pages`.
3. **Settings → Pages → Source**: *Deploy from a branch*, `gh-pages`, `/ (root)`.

The `site/` prefix is stripped on publish, so keep every internal path
**relative** (`assets/x.png`, never `/assets/x.png`) — the site root is
`/<repo>/`.

## Before publishing

- [ ] Replace the placeholder arXiv ID `2604.00000` (hero button and footer link)
      once the paper is posted, and add the ID to the BibTeX `journal` field.
- [ ] Confirm the Hugging Face collection URL resolves publicly.
- [ ] Re-check for internal hostnames and IPs:

```bash
grep -rnIE '\b(10|172|192\.168|28)\.[0-9]+\.[0-9]+\.[0-9]+' . \
  --exclude-dir=.git --exclude-dir=__pycache__
```
