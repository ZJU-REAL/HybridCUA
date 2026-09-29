// The experiment path is the only setting. Everything the dashboard needs —
// task config, examples directory, step ceiling — is derived from `args.json`
// inside it and reported by /api/experiment, along with the provenance of each
// value and warnings for anything that could not be worked out.
//
// Compare mode adds a second path and shows the two side by side. Every API is
// already keyed by `?path=`, and the status cache is per-path, so comparing is
// just fetching twice and joining on (task_type, task_id) — see compare.js.
const PATH_KEY = 'path';
const PATH_B_KEY = 'path_b';
const MODE_KEY = 'mode';
// Pre-auto-discovery name for the same value; still read so old links work.
const LEGACY_PATH_KEY = 'results_base_path';
const STORAGE_KEY = 'osworld.monitor.settings';
const SIDEBAR_STORAGE_KEY = 'osworld.monitor.sidebar-collapsed';

// How often to re-poll while the background scan is still filling in statuses.
const POLL_INTERVAL_MS = 2000;

let pathA = '';
let pathB = '';
let viewMode = 'single';     // 'single' | 'compare'
let allTaskData = null;
let currentFilter = 'all';
let categoryStats = {};
let pollTimer = null;

// Bumped every time the view changes. An in-flight request captures the epoch
// and drops its response if it is no longer current, because `clearTimeout`
// only cancels a *scheduled* poll — a request already on the wire still
// resolves. Without this, switching mode mid-flight lets the old mode repaint
// the list and re-arm its own poll loop, leaving the page wedged showing one
// mode's content under the other mode's chrome.
let viewEpoch = 0;

document.addEventListener('DOMContentLoaded', () => {
    bindStatCards();
    bootstrap();
});

function bindStatCards() {
    document.getElementById('total-tasks').parentElement.addEventListener('click', () => setTaskFilter('all'));
    document.getElementById('active-tasks').parentElement.addEventListener('click', () => setTaskFilter('active'));
    document.getElementById('completed-tasks').parentElement.addEventListener('click', () => setTaskFilter('completed'));
    document.getElementById('error-tasks').parentElement.addEventListener('click', () => setTaskFilter('error'));
}

function bootstrap() {
    fetch('/api/settings')
        .then(r => r.json())
        .then(defaults => resolveState(defaults))
        .catch(() => resolveState(null))
        .then(() => {
            document.getElementById('field-path').value = pathA;
            document.getElementById('field-path-b').value = pathB;
            applyModeUI();
            if (pathA) {
                updateURL();
                loadDashboard();
            } else {
                showEmptyState();
            }
        });
}

// ---------- the two settings ----------

function resolveState(defaults) {
    const params = new URLSearchParams(window.location.search);
    const stored = readStored();
    const seed = defaults || {};
    pathA = params.get(PATH_KEY) || params.get(LEGACY_PATH_KEY)
        || stored[PATH_KEY] || stored[LEGACY_PATH_KEY] || seed.path || '';
    pathB = params.get(PATH_B_KEY) || stored[PATH_B_KEY] || seed.path_b || '';
    viewMode = (params.get(MODE_KEY) || stored[MODE_KEY]) === 'compare' ? 'compare' : 'single';
}

function readStored() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') || {};
    } catch (e) {
        return {};
    }
}

function writeStored() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            [PATH_KEY]: pathA, [PATH_B_KEY]: pathB, [MODE_KEY]: viewMode,
        }));
    } catch (e) { /* ignore */ }
}

function updateURL() {
    const url = new URL(window.location);
    // Drop the six settings the form used to carry, so a stale value in an old
    // bookmark can't linger in the address bar looking authoritative.
    ['task_config_path', 'examples_base_path', 'action_space',
        'observation_type', 'model_name', 'max_steps', LEGACY_PATH_KEY]
        .forEach(k => url.searchParams.delete(k));
    if (pathA) url.searchParams.set(PATH_KEY, pathA);
    else url.searchParams.delete(PATH_KEY);
    // Only compare views carry the second path and the mode, so a single-mode
    // link stays as short as it ever was and always opens in single mode.
    if (viewMode === 'compare') {
        url.searchParams.set(MODE_KEY, 'compare');
        if (pathB) url.searchParams.set(PATH_B_KEY, pathB);
        else url.searchParams.delete(PATH_B_KEY);
    } else {
        url.searchParams.delete(MODE_KEY);
        url.searchParams.delete(PATH_B_KEY);
    }
    window.history.replaceState({}, '', url);
}

// ---------- form actions ----------

function applySettings() {
    const nextA = document.getElementById('field-path').value.trim();
    const nextB = document.getElementById('field-path-b').value.trim();
    if (!nextA) {
        alert('Please provide an Experiment Path.');
        document.getElementById('field-path').focus();
        return;
    }
    if (viewMode === 'compare' && !nextB) {
        alert('Compare mode needs a second Experiment Path.');
        document.getElementById('field-path-b').focus();
        return;
    }
    pathA = nextA;
    pathB = nextB;
    writeStored();
    updateURL();
    window.location.reload();
}

function resetSettings() {
    document.getElementById('field-path').value = '';
    document.getElementById('field-path-b').value = '';
}

function setViewMode(mode) {
    if (mode === viewMode) return;
    viewMode = mode;
    // Pick up a second path that was typed but never applied, so switching into
    // compare mode doesn't silently discard it.
    pathB = document.getElementById('field-path-b').value.trim();
    writeStored();
    updateURL();
    applyModeUI();
    clearRenderedView();
    if (pathA) loadDashboard();
    else viewEpoch++;   // nothing to load, but any in-flight response is now stale
}

// A single class on <html> drives every mode-dependent bit of chrome, so the
// show/hide rules live in CSS instead of being toggled element by element.
function applyModeUI() {
    document.documentElement.classList.toggle('compare-mode', viewMode === 'compare');
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === viewMode);
    });
}

// The two modes render mutually incompatible cards, so the old ones have to go
// the moment the mode changes. Otherwise the previous view's list stays on
// screen under the new view's chrome until the fetch returns — seconds on a warm
// path, minutes on a cold one.
function clearRenderedView() {
    document.getElementById('task-container').innerHTML =
        '<div class="loading-spinner"><div class="spinner"></div><div>Loading…</div></div>';
    document.getElementById('score-display').innerHTML = '—';
    ['total-tasks', 'active-tasks', 'completed-tasks', 'error-tasks'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    document.getElementById('compare-stats').innerHTML = '';
    // Both caches describe the mode being left, so neither may be re-rendered.
    allTaskData = null;
    compareData = null;
}

function toggleSidebar() {
    const html = document.documentElement;
    const collapsed = html.classList.toggle('sidebar-collapsed');
    try {
        localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? 'true' : 'false');
    } catch (e) { /* ignore */ }
}

function clearCacheAndRefresh() {
    if (!pathA) {
        alert('Set an Experiment Path before clearing the cache.');
        return;
    }
    if (!confirm('Re-read everything from disk? The first load afterwards is slow again.')) {
        return;
    }
    const btn = document.getElementById('clear-cache-btn');
    const original = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
    btn.disabled = true;

    Promise.all(activePaths().map(p => fetch(buildAPIURL('/api/clear-cache', p), { method: 'POST' })
        .then(r => r.json())))
        .then(results => {
            results.forEach(data => console.log('Cache cleared:', data.message));
            // Re-fetch in place rather than reloading: the background rescan has
            // already started, and the poll loop shows it progressing.
            btn.innerHTML = original;
            btn.disabled = false;
            loadDashboard();
        })
        .catch(err => {
            console.error('Failed to clear cache:', err);
            alert('Failed to clear cache. Please try again.');
            btn.innerHTML = original;
            btn.disabled = false;
        });
}

// ---------- dashboard ----------

// The paths this view is reading — one in single mode, both in compare mode.
function activePaths() {
    return (viewMode === 'compare' ? [pathA, pathB] : [pathA]).filter(Boolean);
}

function buildAPIURL(endpoint, path) {
    const target = path === undefined ? pathA : path;
    if (!target) return endpoint;
    return `${endpoint}?${new URLSearchParams({ [PATH_KEY]: target })}`;
}

// Task types may contain "/" (the route uses a `path:` converter), so each
// segment is escaped individually and the separators are left intact.
//
// `compare` is {other, side} when the link comes from a comparison: it lets the
// detail page's Back link restore the original pair instead of comparing the
// clicked run against itself.
function taskDetailURL(taskType, taskId, path, compare) {
    const type = String(taskType).split('/').map(encodeURIComponent).join('/');
    const params = path ? { [PATH_KEY]: path } : {};
    if (compare && compare.other) {
        Object.assign(params, { mode: 'compare', compare_with: compare.other, side: compare.side });
    }
    const query = new URLSearchParams(params).toString();
    return `/task/${type}/${encodeURIComponent(taskId)}${query ? `?${query}` : ''}`;
}

function loadDashboard() {
    clearTimeout(pollTimer);
    // Invalidate anything already on the wire before starting the new view.
    const epoch = ++viewEpoch;
    hideEmptyState();
    fetchExperiment(epoch);
    if (viewMode === 'compare' && !pathB) {
        showEmptyState('Enter a second <strong>Experiment Path</strong> in the sidebar and press Load.',
            'Compare mode needs two experiments');
        return;
    }
    (viewMode === 'compare' ? fetchCompare : fetchTasks)(epoch);
}

// Whether a response from `epoch` may still touch the DOM. Anything older was
// requested for a view the user has since left.
function isCurrent(epoch) {
    return epoch === viewEpoch;
}

function fetchTasks(epoch = viewEpoch) {
    fetch(buildAPIURL('/api/tasks'))
        .then(r => r.json())
        .then(data => {
            if (!isCurrent(epoch)) return;
            allTaskData = normalizeTasks(data.tasks || {});
            categoryStats = calculateCategoryStats(allTaskData);
            renderTasks(allTaskData);
            updateStatistics(allTaskData);
            renderScanProgress([{ label: '', scan: data.scan || {} }]);
            // The scan runs on a background thread; keep polling until it settles.
            if (data.scan && !data.scan.complete) {
                clearTimeout(pollTimer);
                pollTimer = setTimeout(() => fetchTasks(epoch), POLL_INTERVAL_MS);
            }
        })
        .catch(err => console.error('Error fetching tasks:', err));
}

// A task discovered but not yet read comes back with a null status (and a null
// instruction until the examples directory finishes loading). Fill in a
// placeholder here so the render path can assume both are present.
function normalizeTasks(grouped) {
    const out = {};
    Object.entries(grouped).forEach(([taskType, tasks]) => {
        out[taskType] = tasks.map(task => ({
            id: task.id,
            instruction: task.instruction == null ? 'Loading…' : task.instruction,
            status: task.status || {
                status: 'Loading', progress: 0, max_steps: 0,
                last_update: null, result: null,
            },
        }));
    });
    return out;
}

// `entries` is [{label, scan}] — one per experiment being read. Compare mode
// passes both, so a slow second path is visible rather than looking like a hang;
// single mode passes one with no label, since there is nothing to disambiguate.
function renderScanProgress(entries) {
    const box = document.getElementById('scan-progress');
    const text = document.getElementById('scan-progress-text');
    if (!box) return;
    if (entries.every(e => e.scan.complete)) { box.style.display = 'none'; return; }
    box.style.display = '';
    text.textContent = entries.map(({ label, scan }) => {
        const prefix = label ? `${label}: ` : '';
        return scan.scanned
            ? `${prefix}reading task status ${scan.done} / ${scan.total}…`
            : `${prefix}scanning the experiment directory…`;
    }).join('  ·  ');
}

// Labels for where each derived value came from, shown as a badge so a guessed
// value is never mistaken for one the runner actually recorded.
const SOURCE_LABELS = {
    'args.json': 'from args.json',
    derived: 'derived',
    scan: 'guessed by scanning',
    default: 'fallback default',
    unavailable: 'not found',
};

const DERIVED_FIELDS = [
    ['max_steps', 'Max Steps'],
    ['model', 'Model'],
    ['task_config_path', 'Task Config'],
    ['examples_dir', 'Examples Dir'],
    ['repo_root', 'Repo Root'],
];

function fetchExperiment(epoch = viewEpoch) {
    const paths = activePaths();
    // Label the blocks only when there are two — a single experiment has nothing
    // to disambiguate, so single mode's sidebar is unchanged.
    const labels = paths.length > 1 ? ['A', 'B'] : [''];
    Promise.all(paths.map(p => fetch(buildAPIURL('/api/experiment', p)).then(r => r.json())))
        .then(configs => {
            if (!isCurrent(epoch)) return;
            const blocks = configs.map((config, i) => ({ label: labels[i], config }));
            renderDerived(blocks);
            renderWarnings(blocks);
            renderModelArgs(blocks);
        })
        .catch(err => console.error('Error fetching experiment config:', err));
}

function derivedRowsHTML(config) {
    const sources = config.sources || {};
    return DERIVED_FIELDS.map(([key, label]) => {
        const value = config[key];
        const source = sources[key] || 'unavailable';
        const shown = (value === null || value === '' || value === undefined) ? '—' : String(value);
        return `
            <div class="settings-summary-item derived-item">
                <span class="settings-summary-label">${label}
                    <span class="derived-source derived-source-${source.replace('.', '-')}">${SOURCE_LABELS[source] || source}</span>
                </span>
                <span class="settings-summary-value" title="${escapeHTML(shown)}">${escapeHTML(shown)}</span>
            </div>`;
    }).join('');
}

function renderDerived(blocks) {
    const container = document.getElementById('derived-fields');
    if (!container) return;
    container.innerHTML = blocks.map(({ label, config }) => {
        const head = label
            ? `<div class="derived-block-head side-${label.toLowerCase()}">${label} · ${escapeHTML(config.model || 'unknown model')}</div>`
            : '';
        return `<div class="derived-block">${head}${derivedRowsHTML(config)}</div>`;
    }).join('');
}

function renderWarnings(blocks) {
    const box = document.getElementById('derived-warnings');
    if (!box) return;
    // Attribute each warning to its side, so "could not be derived" in compare
    // mode never leaves you guessing which experiment it is about.
    const items = blocks.flatMap(({ label, config }) =>
        (config.warnings || []).map(w => (label ? `${label}: ${w}` : w)));
    if (!items.length) { box.style.display = 'none'; box.innerHTML = ''; return; }
    box.innerHTML = `
        <div class="derived-warnings-head">
            <i class="fas fa-exclamation-triangle"></i>
            ${items.length} thing${items.length > 1 ? 's' : ''} could not be determined
        </div>
        <ul>${items.map(w => `<li>${escapeHTML(w)}</li>`).join('')}</ul>`;
    box.style.display = 'block';
}

const MODEL_ARG_SKIP = new Set(['action_space', 'observation_type', 'model', 'max_steps']);

function renderModelArgs(blocks) {
    const container = document.getElementById('model-args');
    const sections = blocks.map(({ label, config }) => {
        const extras = Object.entries(config.model_args || {}).filter(([k]) => !MODEL_ARG_SKIP.has(k));
        return { label, extras };
    }).filter(s => s.extras.length);
    if (!sections.length) { container.style.display = 'none'; return; }

    container.innerHTML = sections.map(({ label, extras }, i) => {
        const id = `config-args-${i}`;
        const title = label ? `${label} · args.json (${extras.length})` : `args.json (${extras.length})`;
        const rows = extras.map(([k, v]) =>
            `<div class="config-item"><span class="config-label">${escapeHTML(k)}</span><span class="config-value">${escapeHTML(JSON.stringify(v))}</span></div>`).join('');
        return `
            <div class="config-collapsible">
                <div class="config-collapsible-header" onclick="toggleConfigArgs('${id}')">
                    <i class="fas fa-chevron-right" id="${id}-chevron"></i>
                    <span>${title}</span>
                </div>
                <div class="config-collapsible-content" id="${id}-content" style="display:none;">${rows}</div>
            </div>`;
    }).join('');
    container.style.display = 'block';
}

function toggleConfigArgs(id) {
    const content = document.getElementById(`${id}-content`);
    const chev = document.getElementById(`${id}-chevron`);
    const hidden = content.style.display === 'none' || !content.style.display;
    content.style.display = hidden ? 'block' : 'none';
    chev.classList.toggle('fa-chevron-right', !hidden);
    chev.classList.toggle('fa-chevron-down', hidden);
}

function setTaskFilter(filter) {
    currentFilter = filter;
    if (!allTaskData) return;
    renderTasks(allTaskData);
    highlightSelectedStatCard();
}

function highlightSelectedStatCard() {
    document.querySelectorAll('.stat-card').forEach(card => card.classList.remove('selected'));
    const id = {
        all: 'total-tasks', active: 'active-tasks',
        completed: 'completed-tasks', error: 'error-tasks',
    }[currentFilter];
    if (id) document.getElementById(id).parentElement.classList.add('selected');
}

const DONE_STATUSES = new Set(['Done', 'Done (Message Exit)', 'Done (Max Steps)', 'Done (Thought Exit)']);
const ACTIVE_STATUSES = new Set(['Running', 'Preparing', 'Initializing']);

function updateStatistics(data) {
    let total = 0, active = 0, completed = 0, errored = 0, totalScore = 0;
    Object.values(data).forEach(tasks => {
        total += tasks.length;
        tasks.forEach(task => {
            const s = task.status.status;
            if (ACTIVE_STATUSES.has(s)) active++;
            else if (DONE_STATUSES.has(s)) {
                completed++;
                const score = parseFloat(task.status.result);
                if (!isNaN(score) && score >= 0 && score <= 1) totalScore += score;
            } else if (s === 'Error') errored++;
        });
    });
    document.getElementById('total-tasks').textContent = total;
    document.getElementById('active-tasks').textContent = active;
    document.getElementById('completed-tasks').textContent = completed;
    document.getElementById('error-tasks').textContent = errored;

    const scoreDisplay = document.getElementById('score-display');
    if (completed > 0) {
        const accuracy = (totalScore / completed * 100).toFixed(1);
        scoreDisplay.innerHTML = `<span>${totalScore.toFixed(2)}</span> / <span>${completed}</span> <span class="accuracy-percentage">(${accuracy}%)</span>`;
    } else {
        scoreDisplay.innerHTML = '<span>0.00</span> / <span>0</span> <span class="accuracy-percentage">(0.0%)</span>';
    }
    highlightSelectedStatCard();
}

function calculateCategoryStats(data) {
    const stats = {};
    Object.entries(data).forEach(([taskType, tasks]) => {
        let completed = 0, totalScore = 0, totalSteps = 0, completedWithSteps = 0;
        tasks.forEach(task => {
            const s = task.status.status;
            if (DONE_STATUSES.has(s)) {
                completed++;
                const score = parseFloat(task.status.result);
                if (!isNaN(score) && score >= 0 && score <= 1) totalScore += score;
                if (task.status.progress > 0) { totalSteps += task.status.progress; completedWithSteps++; }
            }
        });
        stats[taskType] = {
            total_score: Math.round(totalScore * 100) / 100,
            avg_steps: completedWithSteps ? Math.round((totalSteps / completedWithSteps) * 10) / 10 : 0,
        };
    });
    return stats;
}

function renderTasks(data) {
    const filtered = {};
    Object.entries(data).forEach(([taskType, tasks]) => {
        const t = currentFilter === 'all' ? tasks : tasks.filter(x => {
            if (currentFilter === 'active') return ACTIVE_STATUSES.has(x.status.status);
            if (currentFilter === 'completed') return DONE_STATUSES.has(x.status.status);
            if (currentFilter === 'error') return x.status.status === 'Error';
            return true;
        });
        if (t.length) filtered[taskType] = t;
    });

    renderGroups(filtered, {
        buildCard: (taskType, task) => buildTaskCard(taskType, task),
        statsHTML: (taskType, tasks) => {
            const counts = { running: 0, completed: 0, error: 0 };
            tasks.forEach(task => {
                const s = task.status.status;
                if (ACTIVE_STATUSES.has(s)) counts.running++;
                else if (DONE_STATUSES.has(s)) counts.completed++;
                else if (s === 'Error') counts.error++;
            });
            const s = categoryStats[taskType] || {};
            return `
                ${counts.error ? `<span class="task-stat error"><i class="fas fa-exclamation-circle"></i> ${counts.error} error</span>` : ''}
                <span class="task-stat"><i class="fas fa-tasks"></i> ${tasks.length} total</span>
                <span class="task-stat running"><i class="fas fa-running"></i> ${counts.running} active</span>
                <span class="task-stat completed"><i class="fas fa-check-circle"></i> ${counts.completed} completed</span>
                ${s.total_score ? `<span class="task-stat score"><i class="fas fa-star"></i> ${s.total_score} total score</span>` : ''}
                ${s.avg_steps ? `<span class="task-stat steps"><i class="fas fa-chart-line"></i> ${s.avg_steps} avg steps</span>` : ''}`;
        },
    });
}

// The collapsible per-task-type shell, shared by both modes: each supplies the
// badges for a section header and how one row is drawn, and gets the expand
// state, scroll cap, and empty state for free.
function renderGroups(groups, { buildCard, statsHTML }) {
    const container = document.getElementById('task-container');
    container.innerHTML = '';

    if (!Object.keys(groups).length) {
        container.innerHTML = '<div class="no-tasks"><i class="fas fa-info-circle"></i> No tasks at the moment</div>';
        return;
    }

    const expanded = JSON.parse(sessionStorage.getItem('expandedTaskTypes') || '[]');

    Object.entries(groups).forEach(([taskType, items]) => {
        const section = document.createElement('div');
        section.className = 'task-type collapsed';

        const header = document.createElement('div');
        header.className = 'task-type-header';
        header.innerHTML = `
            <span class="task-type-name"><i class="fas fa-layer-group"></i> ${escapeHTML(taskType)}</span>
            <div class="task-type-stats">${statsHTML(taskType, items)}</div>
        `;
        section.appendChild(header);

        const tasksBox = document.createElement('div');
        tasksBox.className = 'tasks-container';
        tasksBox.setAttribute('aria-hidden', 'true');
        if (items.length > 10) { tasksBox.style.maxHeight = '600px'; tasksBox.style.overflowY = 'auto'; }

        items.forEach(item => tasksBox.appendChild(buildCard(taskType, item)));
        section.appendChild(tasksBox);

        header.addEventListener('click', (e) => {
            if (e.target.closest('.task-card')) return;
            section.classList.toggle('collapsed');
            tasksBox.setAttribute('aria-hidden', section.classList.contains('collapsed'));
            const open = [];
            document.querySelectorAll('.task-type').forEach(el => {
                if (!el.classList.contains('collapsed')) {
                    open.push(el.querySelector('.task-type-name').textContent.trim());
                }
            });
            sessionStorage.setItem('expandedTaskTypes', JSON.stringify(open));
        });

        if (expanded.includes(taskType)) {
            section.classList.remove('collapsed');
            tasksBox.setAttribute('aria-hidden', 'false');
        }
        container.appendChild(section);
    });
}

function statusBadge(status) {
    if (status === 'Not Started') return ['status-not-started', 'fa-hourglass-start'];
    if (status === 'Loading') return ['status-not-started', 'fa-spinner fa-pulse'];
    if (status === 'Preparing' || status === 'Initializing') return ['status-preparing', 'fa-spinner fa-pulse'];
    if (status === 'Running') return ['status-running', 'fa-running'];
    if (DONE_STATUSES.has(status)) return ['status-completed', 'fa-check-circle'];
    if (status === 'Error') return ['status-error', 'fa-exclamation-circle'];
    return ['status-unknown', 'fa-question-circle'];
}

function buildTaskCard(taskType, task) {
    const card = document.createElement('div');
    card.className = 'task-card';
    card.setAttribute('data-task-id', task.id);
    card.setAttribute('data-task-type', taskType);

    const [statusClass, statusIcon] = statusBadge(task.status.status);

    const progressHTML = task.status.progress > 0 ? `
        <div><i class="fas fa-chart-line"></i> Progress: ${task.status.progress}/${task.status.max_steps} step(s)</div>
        <div class="progress-bar"><div class="progress-fill" style="width:${(task.status.progress / task.status.max_steps) * 100}%"></div></div>
        <div class="progress-percentage">${Math.round((task.status.progress / task.status.max_steps) * 100)}%</div>
    ` : '';

    // 'Loading' means the status hasn't been read yet — treat it like a task
    // with no directory so the card isn't clickable or resettable prematurely.
    const hasDir = task.status.status !== 'Not Started' && task.status.status !== 'Loading';
    const resetBtn = hasDir
        ? `<button type="button" class="task-reset-btn" title="Delete this task's result directory">
               <i class="fas fa-redo"></i> Reset
           </button>`
        : '';

    card.innerHTML = `
        <div class="task-header">
            <div class="task-title"><i class="fas fa-tasks"></i> Task ID: ${escapeHTML(task.id)}</div>
            <div class="task-status ${statusClass}"><i class="fas ${statusIcon}"></i> ${escapeHTML(task.status.status)}</div>
        </div>
        <div class="task-instruction"><strong><i class="fas fa-info-circle"></i> Instruction:</strong> ${escapeHTML(task.instruction)}</div>
        <div class="task-details">
            ${progressHTML}
            ${task.status.last_update ? `<div class="timestamp"><i class="far fa-clock"></i> Last Update: ${escapeHTML(task.status.last_update)}</div>` : ''}
            ${task.status.result ? `<div class="task-result"><strong><i class="fas fa-flag-checkered"></i> Result:</strong> ${escapeHTML(task.status.result)}</div>` : ''}
        </div>
        ${resetBtn ? `<div class="task-card-actions">${resetBtn}</div>` : ''}
    `;

    if (hasDir) {
        card.style.cursor = 'pointer';
        card.addEventListener('click', (e) => {
            if (e.target.closest('.task-reset-btn')) return;
            window.location.href = taskDetailURL(taskType, task.id, pathA);
        });

        const btn = card.querySelector('.task-reset-btn');
        if (btn) btn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetTask(taskType, task.id, btn);
        });
    }
    return card;
}

// ---------- empty state ----------

function showEmptyState(message, title) {
    const container = document.getElementById('task-container');
    container.innerHTML = `
        <div class="empty-state">
            <i class="fas fa-folder-open"></i>
            <div class="empty-state-title">${title || 'No experiment selected'}</div>
            <div class="empty-state-hint">${message || 'Open the configuration panel on the left and set an <strong>Experiment Path</strong> to load data.'}</div>
        </div>
    `;
    ['total-tasks', 'active-tasks', 'completed-tasks', 'error-tasks'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    document.getElementById('score-display').innerHTML = '—';
}

function hideEmptyState() {
    const el = document.getElementById('empty-state');
    if (el) el.remove();
}

// ---------- destructive: cleanup unfinished / reset single task ----------

function openCleanupModal() {
    if (!pathA) {
        alert('Please set the Experiment Path before cleaning up.');
        return;
    }
    const modal = document.getElementById('cleanup-modal');
    const body = document.getElementById('cleanup-body');
    const confirm = document.getElementById('cleanup-confirm-btn');
    confirm.disabled = true;
    body.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><div>Scanning…</div></div>';
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';

    fetch(buildAPIURL('/api/cleanup/preview'))
        .then(r => r.json())
        .then(data => renderCleanupPreview(data.items || []))
        .catch(err => {
            body.innerHTML = `<div class="cleanup-error"><i class="fas fa-exclamation-circle"></i> ${String(err)}</div>`;
        });
}

function closeCleanupModal() {
    const modal = document.getElementById('cleanup-modal');
    if (modal) modal.classList.remove('open');
    document.body.style.overflow = '';
}

function renderCleanupPreview(items) {
    const body = document.getElementById('cleanup-body');
    const confirmBtn = document.getElementById('cleanup-confirm-btn');

    if (!items.length) {
        body.innerHTML = '<div class="cleanup-empty"><i class="fas fa-check-circle"></i> No unfinished directories found — nothing to delete.</div>';
        confirmBtn.disabled = true;
        return;
    }

    body.__items = items;
    const rowsHTML = items.map((it, i) => `
        <label class="cleanup-row">
            <input type="checkbox" class="cleanup-check" data-idx="${i}" checked>
            <div class="cleanup-row-body">
                <div class="cleanup-row-head">
                    <span class="cleanup-row-id"><strong>${escapeHTML(it.task_type)}</strong> / ${escapeHTML(it.task_id)}</span>
                    <span class="cleanup-row-status">${escapeHTML(it.status)}</span>
                </div>
                <div class="cleanup-row-meta">
                    ${it.progress ? `<span><i class="fas fa-chart-line"></i> ${it.progress}/${it.max_steps} steps</span>` : ''}
                    ${it.last_update && it.last_update !== 'None' ? `<span><i class="far fa-clock"></i> ${escapeHTML(it.last_update)}</span>` : ''}
                </div>
                <div class="cleanup-row-path" title="${escapeHTML(it.path)}">${escapeHTML(it.path)}</div>
            </div>
        </label>
    `).join('');

    body.innerHTML = `
        <div class="cleanup-summary">
            Found <strong>${items.length}</strong> unfinished director${items.length > 1 ? 'ies' : 'y'}.
            Uncheck any you want to keep, then confirm.
            <button type="button" class="cleanup-select-all">Toggle all</button>
        </div>
        <div class="cleanup-list">${rowsHTML}</div>
    `;
    confirmBtn.disabled = false;
    updateConfirmLabel();

    body.querySelectorAll('.cleanup-check').forEach(cb => cb.addEventListener('change', updateConfirmLabel));
    const toggle = body.querySelector('.cleanup-select-all');
    if (toggle) toggle.addEventListener('click', () => {
        const boxes = Array.from(body.querySelectorAll('.cleanup-check'));
        const allOn = boxes.every(cb => cb.checked);
        boxes.forEach(cb => { cb.checked = !allOn; });
        updateConfirmLabel();
    });
}

function updateConfirmLabel() {
    const body = document.getElementById('cleanup-body');
    const confirmBtn = document.getElementById('cleanup-confirm-btn');
    const n = body.querySelectorAll('.cleanup-check:checked').length;
    confirmBtn.disabled = n === 0;
    confirmBtn.innerHTML = `<i class="fas fa-trash-alt"></i> Delete ${n} selected`;
}

function executeCleanup() {
    const body = document.getElementById('cleanup-body');
    const items = body.__items || [];
    const selected = Array.from(body.querySelectorAll('.cleanup-check:checked'))
        .map(cb => items[+cb.dataset.idx])
        .filter(Boolean)
        .map(it => ({ task_type: it.task_type, task_id: it.task_id }));

    if (!selected.length) return;
    if (!confirm(`Permanently delete ${selected.length} director${selected.length > 1 ? 'ies' : 'y'}? This cannot be undone.`)) return;

    const btn = document.getElementById('cleanup-confirm-btn');
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting…';

    fetch(buildAPIURL('/api/cleanup/execute'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: selected }),
    })
        .then(r => r.json())
        .then(data => {
            const ok = (data.deleted || []).length;
            const errs = (data.errors || []).length;
            alert(`Deleted ${ok} director${ok === 1 ? 'y' : 'ies'}${errs ? `; ${errs} error(s):\n` + (data.errors || []).join('\n') : '.'}`);
            closeCleanupModal();
            window.location.reload();
        })
        .catch(err => {
            alert('Failed: ' + String(err));
            btn.disabled = false;
            btn.innerHTML = original;
        });
}

function resetTask(taskType, taskId, button) {
    if (!pathA) {
        alert('Please set the Experiment Path first.');
        return;
    }
    if (!confirm(`Reset ${taskType} / ${taskId}?\nThis will permanently delete its result directory.`)) return;

    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resetting…';

    fetch(buildAPIURL(`/api/task/${encodeURIComponent(taskType)}/${encodeURIComponent(taskId)}/reset`), { method: 'POST' })
        .then(r => r.json().then(d => ({ ok: r.ok, data: d })))
        .then(({ ok, data }) => {
            if (!ok || data.error) {
                alert('Failed: ' + (data.error || 'unknown error'));
                button.disabled = false;
                button.innerHTML = original;
                return;
            }
            fetchTasks(); // refresh counts + cards
        })
        .catch(err => {
            alert('Failed: ' + String(err));
            button.disabled = false;
            button.innerHTML = original;
        });
}

function escapeHTML(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
        c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
