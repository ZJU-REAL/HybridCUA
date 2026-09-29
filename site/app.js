/* HybridCUA project page — chrome plus every table, rendered from leaderboard.json.
   Figures are the paper's own PNGs; only tabular data is rebuilt as HTML. */

(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const OURS = /HybridCUA/i;

    /* ---------------------------------------------------------------- chrome */

    // theme toggle: an explicit choice wins over the OS setting, and persists
    const root = document.documentElement;
    const stored = localStorage.getItem('hc-theme');
    if (stored === 'light' || stored === 'dark') root.dataset.theme = stored;

    $('theme-toggle').addEventListener('click', () => {
        const dark = root.dataset.theme
            ? root.dataset.theme === 'dark'
            : matchMedia('(prefers-color-scheme: dark)').matches;
        root.dataset.theme = dark ? 'light' : 'dark';
        localStorage.setItem('hc-theme', root.dataset.theme);
    });

    // mobile nav
    const toggle = $('nav-toggle');
    const links = $('nav-links');
    toggle.addEventListener('click', () => {
        const open = links.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
    });
    links.addEventListener('click', (e) => {
        if (e.target.tagName === 'A') {
            links.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });

    // highlight the section currently in view
    const navAnchors = [...links.querySelectorAll('a')];
    const sections = navAnchors
        .map((a) => document.querySelector(a.getAttribute('href')))
        .filter(Boolean);
    if ('IntersectionObserver' in window && sections.length) {
        const seen = new Map();
        const io = new IntersectionObserver(
            (entries) => {
                entries.forEach((en) => seen.set(en.target.id, en.intersectionRatio));
                let best = null;
                let bestRatio = 0;
                seen.forEach((ratio, id) => {
                    if (ratio > bestRatio) { bestRatio = ratio; best = id; }
                });
                navAnchors.forEach((a) =>
                    a.classList.toggle('active', best !== null && a.getAttribute('href') === `#${best}`));
            },
            { rootMargin: '-64px 0px -55% 0px', threshold: [0, 0.15, 0.4, 0.75, 1] },
        );
        sections.forEach((s) => io.observe(s));
    }

    // bibtex copy
    const copyBtn = $('copy-bib');
    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText($('bib').textContent);
            copyBtn.textContent = 'Copied';
        } catch {
            copyBtn.textContent = 'Press ⌘C';
        }
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1800);
    });

    /* ---------------------------------------------------------------- tables */

    const fmt = (v, d = 1) => (v === null || v === undefined ? '–' : v.toFixed(d));

    function table(cols, rows) {
        const t = document.createElement('table');

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        cols.forEach((c) => {
            const th = document.createElement('th');
            th.textContent = c.label;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        t.appendChild(thead);

        const tbody = document.createElement('tbody');
        rows.forEach((r) => {
            const row = document.createElement('tr');

            if (r._group) {
                const td = document.createElement('td');
                td.colSpan = cols.length;
                td.textContent = r._group;
                row.className = 'group-row';
                row.appendChild(td);
                tbody.appendChild(row);
                return;
            }

            if (r.highlight) row.classList.add('hl');
            if (r.rule) row.classList.add('rule-above');
            cols.forEach((c, i) => {
                const td = document.createElement('td');
                td.innerHTML = c.cell(r);
                if (i === 0 && r.indent) td.classList.add('indent');
                row.appendChild(td);
            });
            tbody.appendChild(row);
        });
        t.appendChild(tbody);
        return t;
    }

    const deltaSpan = (v, goodWhenNegative = false) => {
        if (v === null || v === undefined) return '';
        const good = goodWhenNegative ? v < 0 : v > 0;
        return `<span class="delta ${good ? 'up' : 'dn'}">${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</span>`;
    };

    const chip = (space) =>
        `<span class="space-chip${/CLI/.test(space) ? ' cli' : ''}">${space.replace('+', ' + ')}</span>`;

    const bestOf = (rows, key) =>
        Math.max(...rows.map((r) => (typeof r[key] === 'number' ? r[key] : -Infinity)));

    const minOf = (rows, key) =>
        Math.min(...rows.map((r) => (typeof r[key] === 'number' ? r[key] : Infinity)));

    const mark = (v, best, d = 1) =>
        v === null || v === undefined
            ? '–'
            : v === best
                ? `<span class="best">${v.toFixed(d)}</span>`
                : v.toFixed(d);

    const name = (s) => (OURS.test(s) ? `<b>${s}</b>` : s);

    /* ---------------------------------------------------------------- KPIs */

    function renderKpis(h) {
        const cards = [
            {
                label: 'OSWorld accuracy',
                value: fmt(h.osworld_acc),
                unit: '%',
                delta: `+${fmt(h.osworld_gain)} vs base`,
                sub: 'Best among comparably sized models',
            },
            {
                label: 'Average steps',
                value: fmt(h.avg_steps),
                delta: '−17.6 vs base',
                sub: '29.3% shorter than after SFT alone',
            },
            {
                label: 'CLI step share',
                value: fmt(h.cli_step_share),
                unit: '%',
                sub: 'Selective, not maximal, shell use',
            },
            {
                label: 'HybridCUA-8K',
                value: h.sft_trajectories.toLocaleString('en-US'),
                sub: `trajectories + ${h.rlvr_tasks.toLocaleString('en-US')} verified RLVR tasks`,
            },
        ];
        $('kpis').innerHTML = cards
            .map(
                (c) => `<div class="kpi">
                    <span class="kpi-label">${c.label}</span>
                    <span class="kpi-value">${c.value}${c.unit ? `<span class="kpi-unit">${c.unit}</span>` : ''}${
                    c.delta ? `<span class="kpi-delta">${c.delta}</span>` : ''}</span>
                    <span class="kpi-sub">${c.sub}</span>
                </div>`,
            )
            .join('');
    }

    /* ---------------------------------------------------------------- spec */

    const SPEC = [
        {
            title: 'Supervised fine-tuning',
            rows: [
                ['Base model', 'Qwen3.5-9B'],
                ['Samples / trajectories', '46,876 / 5,023'],
                ['Context / truncation', '12,000 tok / left'],
                ['Screenshots (cur. / hist.)', '1 / 2'],
                ['Visual tokens per frame', '2,040 / 510'],
                ['Global batch / epochs', '256 / 2'],
                ['Optimizer updates', '366'],
                ['Learning rate', '1e−5, cosine, 10% warm-up'],
                ['Cluster', '2 × 8 H20 (TP 2 / PP 1 / DP 8)'],
            ],
        },
        {
            title: 'Online RL (GRPO)',
            rows: [
                ['Tasks (sampled / verified)', '1,000 / 3,000'],
                ['Stack', 'slime + Megatron-LM + SGLang'],
                ['Rollouts per prompt', '8'],
                ['λ_CLI (trajectory level)', '0.1'],
                ['λ_exec (step level)', '0.3'],
                ['Clip ratio', '0.2 / 0.2'],
                ['KL penalty / loss', '0.001 / 0.01 (k3)'],
                ['Learning rate', '1e−6, constant'],
                ['Env steps (train / eval)', '30 / 50'],
                ['Max policy lag δ', '2 updates'],
                ['Cluster', '3 × 8 H20 (16 train / 8 inference)'],
            ],
        },
        {
            title: 'Evaluation',
            rows: [
                ['OSWorld', '361 tasks, 50-step budget'],
                ['OSWorld-MCP', '361 tasks (MCP-enabled)'],
                ['WindowsAgentArena', '154 tasks'],
                ['Runs per model', '1 (single run, no selection)'],
                ['Accuracy', 'mean evaluator score, ×100'],
                ['Avg. steps', 'mean over all tasks'],
            ],
        },
    ];

    const renderSpec = () => {
        $('spec-grid').innerHTML = SPEC.map(
            (b) => `<div class="spec-block"><h4>${b.title}</h4><dl>${b.rows
                .map(([k, v]) => `<div class="spec-row"><dt>${k}</dt><dd>${v}</dd></div>`)
                .join('')}</dl></div>`,
        ).join('');
    };

    /* ---------------------------------------------------------------- boot */

    async function main() {
        let d;
        try {
            const resp = await fetch('leaderboard.json');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            d = await resp.json();
        } catch (err) {
            console.error('failed to load leaderboard.json', err);
            document.querySelectorAll('.table-scroll').forEach((el) => {
                el.innerHTML =
                    '<p class="note" style="padding:14px 16px">Could not load <code>leaderboard.json</code> — serve this page over HTTP rather than opening the file directly.</p>';
            });
            return;
        }

        renderKpis(d.headline);
        renderSpec();

        /* ---- main results ---- */
        const grouped = d.main_results.groups.flatMap((g) => [{ _group: g.name }, ...g.rows]);
        const flatMain = d.main_results.groups.flatMap((g) => g.rows);
        const bestAcc = bestOf(flatMain, 'acc');
        const bestSteps = minOf(flatMain, 'steps');

        $('table-main').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => name(r.model) },
                    { label: 'Action space', cell: (r) => chip(r.space) },
                    { label: 'Acc. ↑', cell: (r) => mark(r.acc, bestAcc) + deltaSpan(r.acc_delta) },
                    {
                        label: 'Avg. steps ↓',
                        cell: (r) => mark(r.steps, bestSteps) + deltaSpan(r.steps_delta, true),
                    },
                ],
                grouped,
            ),
        );
        $('note-main').textContent = `${d.main_results.caption} Bold marks the best value in a column;`
            + ' deltas are relative to Qwen3.5-9B with a GUI-only action space.';

        /* ---- OOD ---- */
        const ood = d.ood_results.rows;
        $('table-ood').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => name(r.model) },
                    { label: 'Action space', cell: (r) => chip(r.space) },
                    { label: 'OSWorld-MCP ↑', cell: (r) => mark(r.mcp, bestOf(ood, 'mcp')) },
                    { label: 'WindowsAgentArena ↑', cell: (r) => mark(r.waa, bestOf(ood, 'waa')) },
                ],
                ood,
            ),
        );
        $('note-ood').textContent = d.ood_results.caption;

        /* ---- ablations ---- */
        const ablation = (hostId, rows) =>
            $(hostId).appendChild(
                table(
                    [
                        { label: 'Configuration', cell: (r) => (r.highlight ? `<b>${r.config}</b>` : r.config) },
                        { label: 'Acc. ↑', cell: (r) => mark(r.acc, bestOf(rows, 'acc')) },
                        { label: 'Avg. steps ↓', cell: (r) => mark(r.steps, minOf(rows, 'steps')) },
                    ],
                    rows,
                ),
            );
        ablation('table-sft', d.sft_ablation.rows);
        ablation('table-schema', d.schema_ablation.rows);

        /* ---- numbers behind Figure 2 ---- */
        $('table-exposure').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => name(r.model) },
                    { label: 'GUI only', cell: (r) => fmt(r.gui) },
                    { label: 'GUI + CLI', cell: (r) => fmt(r.hybrid) },
                    { label: 'Δ', cell: (r) => deltaSpan(Number((r.hybrid - r.gui).toFixed(1))) },
                    { label: 'CLI step share', cell: (r) => `${fmt(r.cli_share, r.cli_share < 1 ? 2 : 1)}%` },
                ],
                d.cli_exposure.rows,
            ),
        );

        /* ---- numbers behind Figure 6 ---- */
        $('table-domain').appendChild(
            table(
                [
                    { label: 'Domain', cell: (r) => r.domain },
                    { label: 'GUI only', cell: (r) => fmt(r.before) },
                    { label: 'HybridCUA-9B', cell: (r) => `<span class="best">${fmt(r.after)}</span>` },
                    { label: 'Δ', cell: (r) => deltaSpan(Number((r.after - r.before).toFixed(1))) },
                    { label: 'CLI steps', cell: (r) => `${r.cli}%` },
                    { label: 'GUI steps', cell: (r) => `${r.gui}%` },
                ],
                d.domain_results.rows,
            ),
        );

        renderCases(d.cases);
    }

    /* ---------------------------------------------------------------- cases */

    /* Each case ships one screenshot per action step plus a final-state frame:
       N actions → step_0 … step_N, where step_N is the outcome. */
    function renderCases(cases) {
        const host = $('cases');

        cases.forEach((c, ci) => {
            const total = c.steps.length;
            const frames = [
                ...c.steps.map((kind, i) => ({
                    src: `assets/cases/${c.dir}/step_${i}.jpg`,
                    label: String(i + 1).padStart(2, '0'),
                    kind,
                    caption: `Step ${i + 1} of ${total} · ${kind} · pre-action screenshot`,
                })),
                {
                    src: `assets/cases/${c.dir}/step_${total}.jpg`,
                    label: '✓',
                    kind: 'Result',
                    caption: 'Final observation after the last action · evaluator score 1.0',
                },
            ];

            const sec = document.createElement('section');
            sec.className = 'case';
            sec.innerHTML = `
                <header class="case-head">
                    <p class="case-kicker">Case ${ci + 1} · ${c.pattern}</p>
                    <h3>${c.title}</h3>
                    <p class="case-task"><b>Task.</b> ${c.task}</p>
                    <p class="case-flow">${c.flow}</p>
                </header>
                <div class="case-body">
                    <div class="case-viewer">
                        <div class="steps" role="tablist" aria-label="Trajectory steps"></div>
                        <figure class="case-frame">
                            <img alt="" loading="lazy" decoding="async">
                            <figcaption></figcaption>
                        </figure>
                    </div>
                    <div class="case-notes">
                        <p><span class="note-tag note-gui">GUI</span> ${c.gui_note}</p>
                        <p><span class="note-tag note-cli">CLI</span> ${c.cli_note}</p>
                        <p class="case-analysis"><b>Analysis.</b> ${c.analysis}</p>
                    </div>
                </div>`;
            host.appendChild(sec);

            const strip = sec.querySelector('.steps');
            const img = sec.querySelector('.case-frame img');
            const cap = sec.querySelector('.case-frame figcaption');
            const buttons = [];

            const show = (i) => {
                const f = frames[i];
                img.src = f.src;
                img.alt = `${c.title} — ${f.caption}`;
                cap.textContent = f.caption;
                buttons.forEach((b, j) => {
                    b.setAttribute('aria-selected', String(j === i));
                    b.tabIndex = j === i ? 0 : -1;
                });
            };

            frames.forEach((f, i) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.setAttribute('role', 'tab');
                b.className = `step step-${f.kind.toLowerCase()}`;
                b.textContent = f.label;
                b.title = f.caption;
                b.setAttribute('aria-label', f.caption);
                b.addEventListener('click', () => show(i));
                b.addEventListener('keydown', (e) => {
                    const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
                    if (!dir) return;
                    e.preventDefault();
                    const n = (i + dir + frames.length) % frames.length;
                    show(n);
                    buttons[n].focus();
                });
                strip.appendChild(b);
                buttons.push(b);
            });

            show(0);
        });
    }

    main();
})();
