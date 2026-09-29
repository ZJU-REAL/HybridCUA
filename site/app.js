/* HybridCUA project page — tables and charts, all values read from leaderboard.json.
   No dependencies; SVG is built by hand so it inherits the CSS colour tokens. */

(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const NS = 'http://www.w3.org/2000/svg';
    const OURS = /HybridCUA/i;

    /* ---------------------------------------------------------------- chrome */

    // theme toggle: explicit choice wins over the OS setting, and persists
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

    // trajectory tabs
    const tabs = [...document.querySelectorAll('#traj-tabs [role="tab"]')];
    const selectTab = (tab) => {
        tabs.forEach((t) => {
            const on = t === tab;
            t.setAttribute('aria-selected', String(on));
            t.tabIndex = on ? 0 : -1;
            $(t.getAttribute('aria-controls')).hidden = !on;
        });
    };
    tabs.forEach((t, i) => {
        t.tabIndex = i === 0 ? 0 : -1;
        t.addEventListener('click', () => selectTab(t));
        t.addEventListener('keydown', (e) => {
            const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
            if (!d) return;
            e.preventDefault();
            const next = tabs[(tabs.indexOf(t) + d + tabs.length) % tabs.length];
            selectTab(next);
            next.focus();
        });
    });

    /* ---------------------------------------------------------------- tooltip */

    const tip = $('tooltip');
    let tipOwner = null;

    function showTip(html, ev, owner) {
        tip.innerHTML = html;
        tip.classList.add('on');
        tipOwner = owner;
        const r = tip.getBoundingClientRect();
        const x = Math.min(Math.max(ev.clientX, r.width / 2 + 8), innerWidth - r.width / 2 - 8);
        const y = Math.max(ev.clientY - 12, r.height + 12);
        tip.style.left = `${x}px`;
        tip.style.top = `${y}px`;
    }

    function hideTip(owner) {
        if (owner && owner !== tipOwner) return;
        tip.classList.remove('on');
        tipOwner = null;
    }

    /* Attach hover behaviour to a mark group: dims its siblings and shows a tooltip. */
    function hoverable(target, group, html) {
        const enter = (ev) => {
            [...group.children].forEach((c) => { if (c !== target) c.classList.add('dim'); });
            showTip(html, ev, target);
        };
        const move = (ev) => showTip(html, ev, target);
        const leave = () => {
            [...group.children].forEach((c) => c.classList.remove('dim'));
            hideTip(target);
        };
        target.addEventListener('pointerenter', enter);
        target.addEventListener('pointermove', move);
        target.addEventListener('pointerleave', leave);
    }

    /* ---------------------------------------------------------------- svg helpers */

    const svgEl = (name, attrs = {}) => {
        const el = document.createElementNS(NS, name);
        for (const [k, v] of Object.entries(attrs)) {
            if (v !== null && v !== undefined) el.setAttribute(k, String(v));
        }
        return el;
    };

    const text = (x, y, str, cls, attrs = {}) => {
        const t = svgEl('text', { x, y, class: cls, ...attrs });
        t.textContent = str;
        return t;
    };

    function canvas(host, w, h) {
        host.textContent = '';
        const svg = svgEl('svg', {
            viewBox: `0 0 ${w} ${h}`,
            role: 'img',
            preserveAspectRatio: 'xMidYMid meet',
        });
        host.appendChild(svg);
        return svg;
    }

    /* A bar anchored to the baseline with only its data-end rounded (4px). */
    function barPath(x, y, w, h, r = 4, dir = 'up') {
        const rr = Math.max(0, Math.min(r, w / 2, h));
        if (h <= 0.5) return `M${x} ${y + h} h${w}`;
        if (dir === 'up') {
            return `M${x} ${y + h} V${y + rr} q0 ${-rr} ${rr} ${-rr} h${w - 2 * rr} q${rr} 0 ${rr} ${rr} V${y + h} Z`;
        }
        // 'right': grows left→right, right end rounded
        const rh = Math.max(0, Math.min(r, h / 2, w));
        return `M${x} ${y} h${w - rh} q${rh} 0 ${rh} ${rh} v${h - 2 * rh} q0 ${rh} ${-rh} ${rh} H${x} Z`;
    }

    const legend = (host, items) => {
        host.innerHTML = items
            .map((it) => `<span><i style="background:${it.color}"></i>${it.label}</span>`)
            .join('');
    };

    const fmt = (v, d = 1) => (v === null || v === undefined ? '–' : v.toFixed(d));
    const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

    /* Charts re-render on theme change so mark fills track the tokens. */
    const redraws = [];
    const register = (fn) => { redraws.push(fn); fn(); };
    const rerender = () => redraws.forEach((fn) => fn());
    new MutationObserver(rerender).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', rerender);

    let resizeTimer;
    addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(rerender, 180);
    });

    /* ---------------------------------------------------------------- tables */

    function table(cols, rows) {
        const t = document.createElement('table');
        const thead = document.createElement('thead');
        const tr = document.createElement('tr');
        cols.forEach((c) => {
            const th = document.createElement('th');
            th.textContent = c.label;
            if (c.width) th.style.width = c.width;
            tr.appendChild(th);
        });
        thead.appendChild(tr);
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
                const cell = c.cell(r);
                if (cell instanceof Node) td.appendChild(cell);
                else td.innerHTML = cell;
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
        const sign = v > 0 ? '+' : '−';
        return `<span class="delta ${good ? 'up' : 'dn'}">${sign}${Math.abs(v).toFixed(1)}</span>`;
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
                unit: '',
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
                unit: '',
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

    /* ------------------------------------------------- chart: CLI exposure */

    function chartExposure(host, rows) {
        const w = 520;
        const padT = 16;
        const padB = 54;
        const padL = 34;
        const padR = 12;
        const h = 300;
        const plotH = h - padT - padB;
        const plotW = w - padL - padR;

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Grouped bar chart of OSWorld accuracy for five agents with a GUI-only action space and with GUI plus CLI.');

        const max = 90;
        const y = (v) => padT + plotH - (v / max) * plotH;

        for (let v = 0; v <= max; v += 15) {
            svg.appendChild(svgEl('line',
                { x1: padL, x2: w - padR, y1: y(v), y2: y(v), class: 'grid-line' }));
            svg.appendChild(text(padL - 7, y(v) + 4, String(v), 'tick', { 'text-anchor': 'end' }));
        }
        svg.appendChild(text(padL - 7, padT - 4, '%', 'tick', { 'text-anchor': 'end' }));

        const band = plotW / rows.length;
        const barW = Math.min(26, (band - 14) / 2);
        const gap = 2;                       // 2px surface gap between adjacent bars
        const marks = svgEl('g');
        svg.appendChild(marks);

        rows.forEach((r, i) => {
            const cx = padL + band * (i + 0.5);
            const g = svgEl('g');
            marks.appendChild(g);

            const series = [
                { key: 'gui', label: 'GUI only', color: css('--gui'), v: r.gui },
                { key: 'hybrid', label: 'GUI + CLI', color: css('--cli'), v: r.hybrid },
            ];

            series.forEach((s, j) => {
                const x = cx - barW - gap / 2 + j * (barW + gap);
                const top = y(s.v);
                const bar = svgEl('path', {
                    d: barPath(x, top, barW, padT + plotH - top),
                    fill: s.color,
                    class: 'mark',
                });
                g.appendChild(bar);
                svg.appendChild(text(x + barW / 2, top - 6, fmt(s.v),
                    `val-label${r.highlight ? ' strong' : ''}`, { 'text-anchor': 'middle' }));
            });

            const delta = r.hybrid - r.gui;
            const hit = svgEl('rect', {
                x: cx - band / 2, y: padT, width: band, height: plotH, class: 'hit',
            });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.model}</div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${css('--gui')}"></i>GUI only</span><b>${fmt(r.gui)}%</b></div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${css('--cli')}"></i>GUI + CLI</span><b>${fmt(r.hybrid)}%</b></div>
                <div class="tt-row"><span>Δ with CLI</span><b style="color:${delta > 0 ? css('--good') : css('--bad')}">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}</b></div>
                <div class="tt-row"><span>CLI step share</span><b>${fmt(r.cli_share, 2)}%</b></div>`);

            // two-line category label
            const parts = r.model.split('-');
            const label = svgEl('text', {
                x: cx, y: h - padB + 18, class: `cat-label${r.highlight ? ' is-ours' : ''}`,
                'text-anchor': 'middle',
            });
            const l1 = svgEl('tspan', { x: cx });
            l1.textContent = parts.length > 1 ? parts.slice(0, -1).join('-') : r.model;
            const l2 = svgEl('tspan', { x: cx, dy: 14 });
            l2.textContent = parts.length > 1 ? `-${parts.at(-1)}` : '';
            label.append(l1, l2);
            svg.appendChild(label);
            if (r.highlight) {
                svg.appendChild(text(cx, h - padB + 46, '(ours)',
                    'val-label strong', { 'text-anchor': 'middle' }));
            }
        });

        svg.appendChild(svgEl('line',
            { x1: padL, x2: w - padR, y1: y(0), y2: y(0), class: 'axis-line' }));
    }

    /* ------------------------------------------------- chart: CLI step share */

    function chartShare(host, rows) {
        const w = 520;
        const rowH = 38;
        const padT = 8;
        const padB = 30;
        const padL = 122;
        const padR = 46;
        const h = padT + rows.length * rowH + padB;
        const plotW = w - padL - padR;

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Horizontal bars of the share of executable steps issued as direct shell commands, per agent.');

        const x = (v) => padL + (v / 100) * plotW;

        [0, 25, 50, 75, 100].forEach((v) => {
            svg.appendChild(svgEl('line',
                { x1: x(v), x2: x(v), y1: padT, y2: padT + rows.length * rowH, class: 'grid-line' }));
            svg.appendChild(text(x(v), h - padB + 18, `${v}%`, 'tick', { 'text-anchor': 'middle' }));
        });

        const marks = svgEl('g');
        svg.appendChild(marks);
        const barH = 15;

        rows.forEach((r, i) => {
            const cy = padT + i * rowH + rowH / 2;
            const g = svgEl('g');
            marks.appendChild(g);

            const bw = Math.max(2, (r.cli_share / 100) * plotW);
            g.appendChild(svgEl('path', {
                d: barPath(padL, cy - barH / 2, bw, barH, 4, 'right'),
                fill: r.highlight ? css('--cli') : css('--blue-light'),
                class: 'mark',
            }));

            svg.appendChild(text(padL - 10, cy + 4, r.model,
                `cat-label${r.highlight ? ' is-ours' : ''}`, { 'text-anchor': 'end' }));
            svg.appendChild(text(padL + bw + 8, cy + 4, `${fmt(r.cli_share, r.cli_share < 1 ? 2 : 1)}%`,
                `val-label${r.highlight ? ' strong' : ''}`));

            const hit = svgEl('rect',
                { x: padL, y: cy - rowH / 2, width: plotW, height: rowH, class: 'hit' });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.model}</div>
                <div class="tt-row"><span>CLI steps</span><b>${fmt(r.cli_share, 2)}%</b></div>
                <div class="tt-row"><span>GUI steps</span><b>${fmt(100 - r.cli_share, 2)}%</b></div>`);
        });

        svg.appendChild(svgEl('line', {
            x1: padL, x2: padL, y1: padT, y2: padT + rows.length * rowH, class: 'axis-line',
        }));
    }

    /* ------------------------------------------------- chart: SFT modality */

    function chartModality(host, rows) {
        const w = 520;
        const rowH = 52;
        const padT = 6;
        const padL = 118;
        const padR = 78;
        const h = padT + rows.length * rowH + 10;
        const plotW = w - padL - padR;
        const max = 3400;

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Horizontal bars of supervised trajectory counts by interaction mode.');

        const marks = svgEl('g');
        svg.appendChild(marks);
        const barH = 20;
        const shades = [css('--cli'), css('--blue-light'), css('--gui')];

        rows.forEach((r, i) => {
            const cy = padT + i * rowH + rowH / 2;
            const g = svgEl('g');
            marks.appendChild(g);

            const bw = Math.max(2, (r.count / max) * plotW);
            g.appendChild(svgEl('path', {
                d: barPath(padL, cy - barH / 2, bw, barH, 4, 'right'),
                fill: shades[i % shades.length],
                class: 'mark',
            }));

            svg.appendChild(text(padL - 10, cy + 4, r.label, 'cat-label', { 'text-anchor': 'end' }));
            svg.appendChild(text(padL + bw + 9, cy - 1,
                r.count.toLocaleString('en-US'), 'val-label strong'));
            svg.appendChild(text(padL + bw + 9, cy + 13, `${fmt(r.pct)}%`, 'val-label'));

            const hit = svgEl('rect',
                { x: padL, y: cy - rowH / 2, width: plotW + padR - 8, height: rowH, class: 'hit' });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.label}</div>
                <div class="tt-row"><span>Trajectories</span><b>${r.count.toLocaleString('en-US')}</b></div>
                <div class="tt-row"><span>Share of corpus</span><b>${fmt(r.pct)}%</b></div>`);
        });

        svg.appendChild(svgEl('line',
            { x1: padL, x2: padL, y1: padT, y2: padT + rows.length * rowH, class: 'axis-line' }));
    }

    /* ------------------------------------------------- chart: SFT domains */

    function chartDomains(host, rows) {
        const w = 520;
        const rowH = 24;
        const padT = 6;
        const padL = 148;
        const padR = 52;
        const h = padT + rows.length * rowH + 26;
        const plotW = w - padL - padR;
        const max = 900;

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Horizontal bars of supervised trajectory counts by application domain.');

        const x = (v) => padL + (v / max) * plotW;
        [0, 300, 600, 900].forEach((v) => {
            svg.appendChild(svgEl('line',
                { x1: x(v), x2: x(v), y1: padT, y2: padT + rows.length * rowH, class: 'grid-line' }));
            svg.appendChild(text(x(v), padT + rows.length * rowH + 18, String(v),
                'tick', { 'text-anchor': 'middle' }));
        });

        const marks = svgEl('g');
        svg.appendChild(marks);
        const barH = 11;

        rows.forEach((r, i) => {
            const cy = padT + i * rowH + rowH / 2;
            const g = svgEl('g');
            marks.appendChild(g);

            const bw = Math.max(2, (r.count / max) * plotW);
            g.appendChild(svgEl('path', {
                d: barPath(padL, cy - barH / 2, bw, barH, 4, 'right'),
                fill: css('--blue-dark'),
                class: 'mark',
            }));

            svg.appendChild(text(padL - 10, cy + 4, r.label, 'cat-label', { 'text-anchor': 'end' }));
            svg.appendChild(text(padL + bw + 8, cy + 4, r.count.toLocaleString('en-US'), 'val-label'));

            const hit = svgEl('rect',
                { x: padL, y: cy - rowH / 2, width: plotW + padR - 6, height: rowH, class: 'hit' });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.label}</div>
                <div class="tt-row"><span>Trajectories</span><b>${r.count.toLocaleString('en-US')}</b></div>`);
        });

        svg.appendChild(svgEl('line',
            { x1: padL, x2: padL, y1: padT, y2: padT + rows.length * rowH, class: 'axis-line' }));
    }

    /* ------------------------------------------------- chart: domain results */

    function chartDomain(host, rows) {
        const w = 1000;
        const rowH = 30;
        const padT = 26;
        const padB = 26;
        const labelW = 150;
        const gutter = 46;
        const h = padT + rows.length * rowH + padB;

        const accW = 330;
        const shareW = 330;
        const accX = labelW;
        const shareX = labelW + accW + gutter + 62;   // 62px reserved for the "a → b  Δ" text

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Per-domain accuracy of HybridCUA-9B against its GUI-only counterpart, and the GUI/CLI split of its executable steps.');

        svg.appendChild(text(accX, 12, '(a) Accuracy', 'panel-title'));
        svg.appendChild(text(shareX, 12, '(b) Step share', 'panel-title'));

        const ax = (v) => accX + (v / 100) * accW;
        const sx = (v) => shareX + (v / 100) * shareW;

        [0, 25, 50, 75, 100].forEach((v) => {
            [[ax(v), accX], [sx(v), shareX]].forEach(([px]) => {
                svg.appendChild(svgEl('line',
                    { x1: px, x2: px, y1: padT - 6, y2: padT + rows.length * rowH, class: 'grid-line' }));
            });
            svg.appendChild(text(ax(v), padT - 10, `${v}%`, 'tick', { 'text-anchor': 'middle' }));
            svg.appendChild(text(sx(v), padT - 10, `${v}%`, 'tick', { 'text-anchor': 'middle' }));
        });

        const marks = svgEl('g');
        svg.appendChild(marks);
        const barH = 13;
        const cBase = css('--blue-light');
        const cGain = css('--cli');
        const cLoss = css('--series-2');
        const cGui = css('--gui');

        rows.forEach((r, i) => {
            const cy = padT + i * rowH + rowH / 2;
            const g = svgEl('g');
            marks.appendChild(g);
            const delta = r.after - r.before;
            const lo = Math.min(r.before, r.after);
            const hi = Math.max(r.before, r.after);

            // (a) before→after: common range, then the gain or loss segment
            g.appendChild(svgEl('path', {
                d: barPath(accX, cy - barH / 2, Math.max(2, ax(lo) - accX), barH, 4, 'right'),
                fill: cBase, class: 'mark',
            }));
            if (hi > lo) {
                g.appendChild(svgEl('path', {
                    d: barPath(ax(lo) + 2, cy - barH / 2, Math.max(2, ax(hi) - ax(lo) - 2), barH, 4, 'right'),
                    fill: delta > 0 ? cGain : cLoss, class: 'mark',
                }));
            }

            svg.appendChild(text(accX - 10, cy + 4, r.domain, 'cat-label', { 'text-anchor': 'end' }));
            svg.appendChild(text(accX + accW + 12, cy + 4,
                `${fmt(r.before)} → ${fmt(r.after)}`, 'val-label'));
            const d = text(accX + accW + gutter + 46, cy + 4,
                `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}`, 'val-label strong',
                { 'text-anchor': 'end', fill: delta > 0 ? css('--good') : css('--bad') });
            svg.appendChild(d);

            // (b) GUI/CLI step share, stacked with a 2px surface gap
            const guiW = (r.gui / 100) * shareW;
            g.appendChild(svgEl('rect', {
                x: shareX, y: cy - barH / 2, width: Math.max(2, guiW - 1), height: barH,
                fill: cGui, class: 'mark', rx: 0,
            }));
            g.appendChild(svgEl('path', {
                d: barPath(shareX + guiW + 1, cy - barH / 2, Math.max(2, shareW - guiW - 1), barH, 4, 'right'),
                fill: css('--blue-light'), class: 'mark',
            }));
            if (r.gui >= 16) {
                svg.appendChild(text(shareX + guiW / 2, cy + 4, `${r.gui}%`,
                    'in-bar', { 'text-anchor': 'middle' }));
            }
            if (r.cli >= 16) {
                svg.appendChild(text(shareX + guiW + (shareW - guiW) / 2, cy + 4, `${r.cli}%`,
                    'in-bar', { 'text-anchor': 'middle' }));
            }

            const hit = svgEl('rect',
                { x: 0, y: cy - rowH / 2, width: w, height: rowH, class: 'hit' });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.domain}</div>
                <div class="tt-row"><span>GUI-only counterpart</span><b>${fmt(r.before)}%</b></div>
                <div class="tt-row"><span>HybridCUA-9B</span><b>${fmt(r.after)}%</b></div>
                <div class="tt-row"><span>Δ</span><b style="color:${delta > 0 ? css('--good') : css('--bad')}">${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(1)}</b></div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${cGui}"></i>GUI steps</span><b>${r.gui}%</b></div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${css('--blue-light')}"></i>CLI steps</span><b>${r.cli}%</b></div>`);
        });

        [[accX, accX], [shareX, shareX]].forEach(([px]) => {
            svg.appendChild(svgEl('line',
                { x1: px, x2: px, y1: padT - 6, y2: padT + rows.length * rowH, class: 'axis-line' }));
        });
    }

    /* ------------------------------------------------- chart: operation share */

    function chartOperation(host, rows) {
        const w = 520;
        const rowH = 50;
        const padT = 6;
        const padL = 150;
        const padR = 16;
        const h = padT + rows.length * rowH + 4;
        const plotW = w - padL - padR;

        const svg = canvas(host, w, h);
        svg.setAttribute('aria-label',
            'Stacked bars showing the GUI and CLI share of steps for four operation categories.');

        const marks = svgEl('g');
        svg.appendChild(marks);
        const barH = 18;
        const cGui = css('--gui');
        const cCli = css('--cli');

        rows.forEach((r, i) => {
            const cy = padT + i * rowH + rowH / 2 - 4;
            const g = svgEl('g');
            marks.appendChild(g);

            const cliW = (r.cli / 100) * plotW;
            g.appendChild(svgEl('rect', {
                x: padL, y: cy - barH / 2, width: Math.max(2, cliW - 1), height: barH,
                fill: cCli, class: 'mark',
            }));
            g.appendChild(svgEl('path', {
                d: barPath(padL + cliW + 1, cy - barH / 2, Math.max(2, plotW - cliW - 1), barH, 4, 'right'),
                fill: cGui, class: 'mark',
            }));

            svg.appendChild(text(padL - 10, cy + 4, r.operation, 'cat-label', { 'text-anchor': 'end' }));
            svg.appendChild(text(padL, cy + barH / 2 + 17, `CLI ${fmt(r.cli)}%`, 'val-label strong'));
            svg.appendChild(text(padL + plotW, cy + barH / 2 + 17, `GUI ${fmt(r.gui)}%`,
                'val-label', { 'text-anchor': 'end' }));

            const hit = svgEl('rect',
                { x: padL, y: cy - rowH / 2, width: plotW, height: rowH, class: 'hit' });
            g.appendChild(hit);
            hoverable(hit, marks, `<div class="tt-title">${r.operation}</div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${cCli}"></i>CLI</span><b>${fmt(r.cli)}%</b></div>
                <div class="tt-row"><span><i class="tt-swatch" style="background:${cGui}"></i>GUI</span><b>${fmt(r.gui)}%</b></div>`);
        });
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

    function renderSpec() {
        $('spec-grid').innerHTML = SPEC.map(
            (b) => `<div class="spec-block"><h4>${b.title}</h4><dl>${b.rows
                .map(([k, v]) => `<div class="spec-row"><dt>${k}</dt><dd>${v}</dd></div>`)
                .join('')}</dl></div>`,
        ).join('');
    }

    /* ---------------------------------------------------------------- boot */

    async function main() {
        let d;
        try {
            const resp = await fetch('leaderboard.json');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            d = await resp.json();
        } catch (err) {
            console.error('failed to load leaderboard.json', err);
            document.querySelectorAll('.table-scroll, .chart').forEach((el) => {
                el.innerHTML =
                    '<p class="note" style="padding:14px 16px">Could not load <code>leaderboard.json</code> — serve this page over HTTP rather than opening the file directly.</p>';
            });
            return;
        }

        renderKpis(d.headline);
        renderSpec();

        /* ---- main results ---- */
        const mainRows = d.main_results.groups.flatMap((g) => [{ _group: g.name }, ...g.rows]);
        const flatMain = d.main_results.groups.flatMap((g) => g.rows);
        const bestAcc = bestOf(flatMain, 'acc');
        const bestSteps = minOf(flatMain, 'steps');

        $('table-main').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => (OURS.test(r.model) ? `<b>${r.model}</b>` : r.model) },
                    { label: 'Action space', cell: (r) => chip(r.space) },
                    { label: 'Acc. ↑', cell: (r) => mark(r.acc, bestAcc) + deltaSpan(r.acc_delta) },
                    {
                        label: 'Avg. steps ↓',
                        cell: (r) => mark(r.steps, bestSteps) + deltaSpan(r.steps_delta, true),
                    },
                ],
                mainRows,
            ),
        );
        $('note-main').textContent = d.main_results.caption
            + ' Bold marks the best value in a column; deltas are relative to Qwen3.5-9B with a GUI-only action space.';

        /* ---- OOD ---- */
        const ood = d.ood_results.rows;
        const bestMcp = bestOf(ood, 'mcp');
        const bestWaa = bestOf(ood, 'waa');
        $('table-ood').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => (OURS.test(r.model) ? `<b>${r.model}</b>` : r.model) },
                    { label: 'Action space', cell: (r) => chip(r.space) },
                    { label: 'OSWorld-MCP ↑', cell: (r) => mark(r.mcp, bestMcp) },
                    { label: 'WindowsAgentArena ↑', cell: (r) => mark(r.waa, bestWaa) },
                ],
                ood,
            ),
        );
        $('note-ood').textContent = d.ood_results.caption;

        /* ---- ablations ---- */
        const sft = d.sft_ablation.rows;
        $('table-sft').appendChild(
            table(
                [
                    { label: 'Configuration', cell: (r) => (r.highlight ? `<b>${r.config}</b>` : r.config) },
                    { label: 'Acc. ↑', cell: (r) => mark(r.acc, bestOf(sft, 'acc')) },
                    { label: 'Avg. steps ↓', cell: (r) => mark(r.steps, minOf(sft, 'steps')) },
                ],
                sft,
            ),
        );

        const schema = d.schema_ablation.rows;
        $('table-schema').appendChild(
            table(
                [
                    { label: 'Configuration', cell: (r) => (r.highlight ? `<b>${r.config}</b>` : r.config) },
                    { label: 'Acc. ↑', cell: (r) => mark(r.acc, bestOf(schema, 'acc')) },
                    { label: 'Avg. steps ↓', cell: (r) => mark(r.steps, minOf(schema, 'steps')) },
                ],
                schema,
            ),
        );

        /* ---- exposure table view ---- */
        const exp = d.cli_exposure.rows;
        $('table-exposure').appendChild(
            table(
                [
                    { label: 'Model', cell: (r) => (OURS.test(r.model) ? `<b>${r.model}</b>` : r.model) },
                    { label: 'GUI only', cell: (r) => fmt(r.gui) },
                    { label: 'GUI + CLI', cell: (r) => fmt(r.hybrid) },
                    {
                        label: 'Δ',
                        cell: (r) => {
                            const v = r.hybrid - r.gui;
                            return `<span class="delta ${v > 0 ? 'up' : 'dn'}">${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</span>`;
                        },
                    },
                    { label: 'CLI step share', cell: (r) => `${fmt(r.cli_share, r.cli_share < 1 ? 2 : 1)}%` },
                ],
                exp,
            ),
        );

        /* ---- domain table view ---- */
        $('table-domain').appendChild(
            table(
                [
                    { label: 'Domain', cell: (r) => r.domain },
                    { label: 'GUI only', cell: (r) => fmt(r.before) },
                    { label: 'HybridCUA-9B', cell: (r) => `<span class="best">${fmt(r.after)}</span>` },
                    {
                        label: 'Δ',
                        cell: (r) => {
                            const v = r.after - r.before;
                            return `<span class="delta ${v > 0 ? 'up' : 'dn'}">${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</span>`;
                        },
                    },
                    { label: 'CLI steps', cell: (r) => `${r.cli}%` },
                    { label: 'GUI steps', cell: (r) => `${r.gui}%` },
                ],
                d.domain_results.rows,
            ),
        );

        /* ---- RLVR pool ---- */
        const rlvr = d.corpus.rlvr_domains;
        const maxPct = bestOf(rlvr, 'pct');
        $('table-rlvr').appendChild(
            table(
                [
                    { label: 'Domain', cell: (r) => `<code>${r.label}</code>` },
                    {
                        label: 'Share of 3,000 tasks',
                        cell: (r) => {
                            const wrapEl = document.createElement('span');
                            wrapEl.className = 'bar-cell';
                            wrapEl.innerHTML = `<span>${fmt(r.pct)}%</span>
                                <span class="bar-track"><span class="bar-fill" style="width:${(r.pct / maxPct) * 100}%"></span></span>`;
                            return wrapEl;
                        },
                    },
                ],
                rlvr,
            ),
        );

        /* ---- charts ---- */
        register(() => {
            legend($('legend-exposure'), [
                { label: 'GUI only', color: css('--gui') },
                { label: 'GUI + CLI', color: css('--cli') },
            ]);
            chartExposure($('chart-exposure'), exp);
        });
        register(() => chartShare($('chart-share'), exp));
        register(() => chartModality($('chart-modality'), d.corpus.modality));
        register(() => chartDomains($('chart-domains'), d.corpus.domains));
        register(() => {
            legend($('legend-domain'), [
                { label: 'Common range', color: css('--blue-light') },
                { label: 'Gain', color: css('--cli') },
                { label: 'Loss', color: css('--series-2') },
                { label: 'GUI steps', color: css('--gui') },
            ]);
            chartDomain($('chart-domain'), d.domain_results.rows);
        });
        register(() => {
            legend($('legend-op'), [
                { label: 'CLI', color: css('--cli') },
                { label: 'GUI', color: css('--gui') },
            ]);
            chartOperation($('chart-op'), d.operation_share.rows);
        });
    }

    main();
})();
