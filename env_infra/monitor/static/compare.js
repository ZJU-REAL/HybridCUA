// Compare mode: two experiment directories side by side.
//
// Nothing here talks to a comparison endpoint, because there isn't one. Every
// API is keyed by `?path=` and `IndexRegistry` caches per path, so the two
// experiments scan independently and this file only fetches twice and joins on
// (task_type, task_id). Adding a third path later would need no backend work.
//
// Depends on index.js for the shared shell: `pathA`/`pathB`, `buildAPIURL`,
// `renderGroups`, `normalizeTasks`, `DONE_STATUSES`, `escapeHTML`, `isCurrent`.

// A task's comparable score, or null when there is nothing to compare yet.
// Only a finished run with a parseable [0,1] result counts — an Error or an
// in-flight task is *unknown*, not a zero, and scoring it as one would invent
// regressions that never happened.
function comparableScore(status) {
    if (!status || !DONE_STATUSES.has(status.status)) return null;
    const score = parseFloat(status.result);
    return (!isNaN(score) && score >= 0 && score <= 1) ? score : null;
}

// Binary by design: '1' is a full score, '0' is everything else — a zero,
// partial credit, an error, or a run that never finished. Every task lands in
// one or the other, so the two counts always sum to the total.
function scoreBucket(status) {
    return comparableScore(status) >= 1 ? '1' : '0';
}

const BUCKET_HINT = {
    '1': 'scored 1 — solved',
    '0': 'did not score 1 — a zero, partial credit, an error, or never finished',
};

let compareData = null;               // {taskType: [row, ...]} — every row, unfiltered
// null on a side means "either value". {a:'1', b:'0'} is A-right/B-wrong, the
// selection this mode exists for.
let compareFilter = { a: null, b: null };

function fetchCompare(epoch = viewEpoch) {
    Promise.all([pathA, pathB].map(p => fetch(buildAPIURL('/api/tasks', p)).then(r => r.json())))
        .then(([a, b]) => {
            // Switching to single mode mid-flight must not let this repaint the
            // list (nor re-arm the poll below) — see `isCurrent` in index.js.
            if (!isCurrent(epoch)) return;
            compareData = joinTasks(normalizeTasks(a.tasks || {}), normalizeTasks(b.tasks || {}));
            renderCompare();
            renderScanProgress([
                { label: 'A', scan: a.scan || {} },
                { label: 'B', scan: b.scan || {} },
            ]);
            // Either side still scanning means verdicts are provisional; keep
            // polling so 'Not comparable' resolves as statuses land.
            if (!(a.scan || {}).complete || !(b.scan || {}).complete) {
                clearTimeout(pollTimer);
                pollTimer = setTimeout(() => fetchCompare(epoch), POLL_INTERVAL_MS);
            }
        })
        .catch(err => console.error('Error fetching comparison:', err));
}

// Union rather than intersection: a task only one side ran is real information
// (B was cut short, A was re-scoped), and dropping it would quietly shrink the
// denominator. It lands in 'Not comparable' with the missing side shown as '—'.
function joinTasks(groupsA, groupsB) {
    const byType = {};
    const put = (taskType, task, side) => {
        const rows = byType[taskType] || (byType[taskType] = new Map());
        const row = rows.get(task.id) || { id: task.id, instruction: null, a: null, b: null };
        row[side] = task.status;
        // Both sides derive instructions from the same examples directory, but
        // take whichever resolved — one experiment may have no args.json.
        if (row.instruction == null || row.instruction === 'Loading…') row.instruction = task.instruction;
        rows.set(task.id, row);
    };
    Object.entries(groupsA).forEach(([t, tasks]) => tasks.forEach(task => put(t, task, 'a')));
    Object.entries(groupsB).forEach(([t, tasks]) => tasks.forEach(task => put(t, task, 'b')));

    const out = {};
    Object.keys(byType).sort().forEach(taskType => {
        out[taskType] = Array.from(byType[taskType].values())
            .map(row => ({
                ...row,
                scoreA: comparableScore(row.a),
                scoreB: comparableScore(row.b),
                bucketA: scoreBucket(row.a),
                bucketB: scoreBucket(row.b),
            }))
            .sort((x, y) => String(x.id).localeCompare(String(y.id)));
    });
    return out;
}

function matchesCompareFilter(row) {
    const { a, b } = compareFilter;
    return (!a || row.bucketA === a) && (!b || row.bucketB === b);
}

function setCompareFilter(a, b) {
    compareFilter = { a: a || null, b: b || null };
    if (compareData) renderCompare();
}

function renderCompare() {
    const rows = Object.values(compareData).flat();
    renderCompareStats(rows);

    const filtered = {};
    Object.entries(compareData).forEach(([taskType, items]) => {
        const kept = items.filter(matchesCompareFilter);
        if (kept.length) filtered[taskType] = kept;
    });

    renderGroups(filtered, {
        buildCard: buildCompareCard,
        statsHTML: (taskType) => {
            // Counted over every row of the type, not just the visible ones, so a
            // section header keeps meaning the same thing under any filter.
            const rowsOfType = compareData[taskType];
            const aOnly = rowsOfType.filter(r => r.bucketA === '1' && r.bucketB === '0').length;
            const bOnly = rowsOfType.filter(r => r.bucketA === '0' && r.bucketB === '1').length;
            return `<span class="task-stat"><i class="fas fa-tasks"></i> ${rowsOfType.length} total</span>`
                + (aOnly ? `<span class="task-stat stat-a-only">A only ${aOnly}</span>` : '')
                + (bOnly ? `<span class="task-stat stat-b-only">B only ${bOnly}</span>` : '');
        },
    });
}

// Two toggle groups, one per run: pick A's outcome and B's outcome and the list
// narrows to that combination. A=1 with B=0 is the regression view; clicking a
// selected button clears it back to "either".
function renderCompareStats(rows) {
    const count = (side, bucket) => rows.filter(r => {
        const f = { ...compareFilter, [side]: bucket };
        return (!f.a || r.bucketA === f.a) && (!f.b || r.bucketB === f.b);
    }).length;

    const group = (side, label) => {
        const picked = compareFilter[side];
        const buttons = ['1', '0'].map(b => `
            <button type="button" class="score-btn score-${b}${picked === b ? ' selected' : ''}"
                    data-side="${side}" data-bucket="${b}"
                    title="${label} ${BUCKET_HINT[b]}"
                    aria-pressed="${picked === b}">
                <i class="fas ${b === '1' ? 'fa-check' : 'fa-times'}"></i>
                <span class="score-btn-value">${b}</span>
                <span class="score-btn-count">${count(side, b)}</span>
            </button>`).join('');
        return `
            <div class="score-group side-${side}">
                <span class="score-group-label">${label}</span>
                ${buttons}
            </div>`;
    };

    const shown = rows.filter(matchesCompareFilter).length;
    document.getElementById('compare-stats').innerHTML = `
        <div class="score-filters">
            ${group('a', 'A')}
            <span class="score-vs">vs</span>
            ${group('b', 'B')}
        </div>
        <div class="score-filter-state">
            <span class="score-shown"><strong>${shown}</strong> / ${rows.length}</span>
            ${compareFilter.a || compareFilter.b
                ? '<button type="button" class="score-clear"><i class="fas fa-times"></i> clear</button>'
                : '<span class="score-filter-hint">pick a result to filter</span>'}
        </div>`;

    document.querySelectorAll('#compare-stats .score-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const { side, bucket } = btn.dataset;
            // Clicking the active button clears that side rather than leaving you
            // stuck with no way back to "either".
            setCompareFilter(
                side === 'a' ? (compareFilter.a === bucket ? null : bucket) : compareFilter.a,
                side === 'b' ? (compareFilter.b === bucket ? null : bucket) : compareFilter.b);
        });
    });
    const clear = document.querySelector('#compare-stats .score-clear');
    if (clear) clear.addEventListener('click', () => setCompareFilter(null, null));

    renderCompareScore(rows);
}

// Every task counts, over the same denominator on both sides: an unfinished or
// errored run contributes 0, matching how the 0 button treats it.
function renderCompareScore(rows) {
    const el = document.getElementById('score-display');
    if (!rows.length) {
        el.innerHTML = '<span class="compare-score-empty">No tasks found in either run</span>';
        return;
    }
    const sum = pick => rows.reduce((acc, r) => acc + (pick(r) || 0), 0);
    const sumA = sum(r => r.scoreA);
    const sumB = sum(r => r.scoreB);
    const pct = total => (total / rows.length * 100).toFixed(1);
    const delta = sumB - sumA;
    const sign = delta > 0 ? '+' : delta < 0 ? '−' : '±';
    const deltaClass = delta > 0 ? 'delta-up' : delta < 0 ? 'delta-down' : 'delta-flat';

    el.innerHTML = `
        <span class="compare-side side-a">A ${sumA.toFixed(2)}/${rows.length}
            <span class="accuracy-percentage">${pct(sumA)}%</span></span>
        <span class="compare-vs">vs</span>
        <span class="compare-side side-b">B ${sumB.toFixed(2)}/${rows.length}
            <span class="accuracy-percentage">${pct(sumB)}%</span></span>
        <span class="compare-delta ${deltaClass}">${sign}${Math.abs(delta).toFixed(2)}
            (${sign}${Math.abs(delta / rows.length * 100).toFixed(1)}%)</span>`;
}

function buildCompareCard(taskType, row) {
    const card = document.createElement('div');
    const differ = row.bucketA !== row.bucketB;
    card.className = `task-card compare-card ${differ ? 'cmp-differ' : 'cmp-same'}`;
    card.setAttribute('data-task-id', row.id);
    card.setAttribute('data-task-type', taskType);

    // The two scores are the verdict — no label needed to say what "1 → 0" means.
    const pair = `<span class="compare-score-pair">${fmtScore(row.scoreA)} → ${fmtScore(row.scoreB)}</span>`;

    card.innerHTML = `
        <div class="task-header">
            <div class="task-title"><i class="fas fa-tasks"></i> Task ID: ${escapeHTML(row.id)}</div>
            <div class="compare-verdict cmp-badge-${differ ? 'differ' : 'same'}">${pair}</div>
        </div>
        <div class="task-instruction"><strong><i class="fas fa-info-circle"></i> Instruction:</strong> ${escapeHTML(row.instruction == null ? 'Loading…' : row.instruction)}</div>
        <div class="compare-sides">
            ${sidePanelHTML('A', row.a, row.scoreA, taskType, row.id, pathA, pathB)}
            ${sidePanelHTML('B', row.b, row.scoreB, taskType, row.id, pathB, pathA)}
        </div>`;
    return card;
}

function fmtScore(score) {
    return score === null ? '—' : String(Math.round(score * 100) / 100);
}

// Each side is a link to that run's trajectory, so the screenshots and actions
// are one click away — the usual next question when a task passes on one side
// and fails on the other. The whole panel is the target
// rather than a small icon, matching single mode's whole-card click.
function sidePanelHTML(label, status, score, taskType, taskId, path, otherPath) {
    if (!status) {
        return `<div class="compare-side-panel side-${label.toLowerCase()} missing">
            <div class="compare-side-head"><span class="compare-side-tag">${label}</span> not in this run</div>
        </div>`;
    }
    const [statusClass, statusIcon] = statusBadge(status.status);
    // 'Loading' means the status hasn't been read yet, so the directory can't be
    // assumed to exist — matching buildTaskCard's rule for the same reason.
    const hasDir = status.status !== 'Not Started' && status.status !== 'Loading';
    const progress = status.progress > 0 && status.max_steps > 0
        ? `<div class="compare-side-meta"><i class="fas fa-chart-line"></i> ${status.progress}/${status.max_steps} steps</div>`
        : '';
    const body = `
            <div class="compare-side-head">
                <span class="compare-side-tag">${label}</span>
                <span class="task-status ${statusClass}"><i class="fas ${statusIcon}"></i> ${escapeHTML(status.status)}</span>
                <span class="compare-side-score">score ${fmtScore(score)}</span>
                ${hasDir ? '<span class="compare-side-link"><i class="fas fa-images"></i> View steps</span>' : ''}
            </div>
            ${progress}
            ${status.last_update ? `<div class="compare-side-meta"><i class="far fa-clock"></i> ${escapeHTML(status.last_update)}</div>` : ''}`;

    const classes = `compare-side-panel side-${label.toLowerCase()}`;
    const href = taskDetailURL(taskType, taskId, path,
        { other: otherPath, side: label.toLowerCase() });
    return hasDir
        ? `<a class="${classes} clickable" href="${href}"
              title="Open ${label}'s screenshots and actions">${body}</a>`
        : `<div class="${classes}">${body}</div>`;
}
