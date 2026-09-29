# site/ — HybridCUA project page

Static single-page site. No build step, no npm: Bulma, Chart.js, highlight.js,
Font Awesome and Academicons all load from CDN in `index.html`'s `<head>`.

```
site/
├── index.html        the whole page (markup + inline JS)
├── style.css         custom layer on top of Bulma
├── leaderboard.json  results table + training curve, fetched at runtime
├── .nojekyll         tells GitHub Pages to serve files as-is
└── assets/           figures (teaser.png, pipeline.png, ...)
```

## Local preview

`index.html` `fetch()`es `leaderboard.json`, which the `file://` protocol
blocks — serve over HTTP instead of double-clicking the file:

```bash
python3 -m http.server 8000 --directory site
# then open http://localhost:8000/
```

## Deploying

`.github/workflows/deploy-pages.yml` publishes this directory to the `gh-pages`
branch on every push to `main` that touches `site/**`. One-time repo setup:

1. **Settings → Actions → General → Workflow permissions**: select
   *Read and write permissions* (the workflow needs to push `gh-pages`).
2. Push once and let the workflow create the `gh-pages` branch.
3. **Settings → Pages → Source**: *Deploy from a branch*, branch `gh-pages`,
   folder `/ (root)`.

The published site lives at `https://<owner>.github.io/<repo>/`. Note the
`site/` prefix is stripped on publish, so `assets/x.png` resolves correctly —
keep every internal path **relative** (`assets/x.png`, not `/assets/x.png`),
because the site root is `/<repo>/` rather than `/`.

## Updating results

Edit `leaderboard.json` only — never hardcode numbers into `index.html`.
Columns are keyed by the `data-sort` attributes on the table headers
(`model`, `size`, `overall`, `office`, `os`, `web`); to add a metric, add the
`<th data-sort="...">` and extend `NUMERIC_COLS` in the inline script.

## Before publishing

This repo was scrubbed of internal hostnames, IPs and credentials. Re-check
before making it public:

```bash
grep -rnIE '\b(10|172|192\.168|28)\.[0-9]+\.[0-9]+\.[0-9]+' . \
  --exclude-dir=.git --exclude-dir=__pycache__
```

## TODO

- [ ] Authors, affiliations, arXiv / paper / HF links in the hero section
- [ ] Abstract and the three highlight cards
- [ ] Method section prose
- [ ] `assets/teaser.png`, `assets/pipeline.png`
- [ ] Real numbers in `leaderboard.json`
- [ ] BibTeX entry
- [ ] Repo URL (currently `TODO/HybridCUA` in several places)
