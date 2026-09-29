# site/ — HybridCUA project page

Static single-page site built from the paper in `projects/HybridCUA-arxiv`.
**No dependencies, no CDN, no build step** — three files plus figures.

```
site/
├── index.html        markup and copy
├── style.css         design tokens + layout (light/dark)
├── app.js            renders every table and chart from leaderboard.json
├── leaderboard.json  ALL numbers live here — the only file to edit for results
├── .nojekyll         tells GitHub Pages to serve files as-is
└── assets/           figures exported from the paper (see below)
```

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

## Charts

Hand-built inline SVG so marks inherit the CSS custom properties and re-render on
theme change. The palette is the validated categorical slots 1–3 plus a neutral
for the GUI series; interface identity is always carried by a legend *and* direct
labels, never colour alone, and every chart has a `<details>` table view.

Re-run the check after changing any series colour:

```bash
node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --surface "#ffffff" --pairs all
node scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark  --surface "#17181b" --pairs all
```

## Assets

Only four figures are shipped — anything the page can render live from
`leaderboard.json` is a chart, not an image, so the two never disagree:

| `assets/` | source in `figures/` | used in |
|---|---|---|
| `teaser.png` | `intro.png` | hero |
| `pipeline.png` | `method.png` | Method |
| `rl-ablation.png` | `training_ablation.png` | Results |
| `case-study.png` | `case_study.png` | Analysis |

`GUI_CLI.png`, `traj-distribution.png`, `domain_results.png` and `stats.png` are
deliberately *not* included: the page plots that data itself. To refresh a shipped
figure, re-export with PIL at 1800px max width and keep the filename.

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
