(function () {
    'use strict';

    const COLS_KEY = 'osworld.monitor.steps-per-row';
    const DEFAULT_COLS = 2;
    const MIN_COLS = 1;
    const MAX_COLS = 10;

    const KIND_COLORS = {
        click:    '#ef4444',
        dblclick: '#dc2626',
        rclick:   '#2563eb',
        move:     '#f59e0b',
        drag:     '#8b5cf6',
        scroll:   '#0ea5e9',
        type:     '#111827',
        press:    '#111827',
        hotkey:   '#111827',
        done:     '#22c55e',
        fail:     '#ef4444',
        wait:     '#f59e0b',
        other:    '#111827',
    };

    document.addEventListener('DOMContentLoaded', () => {
        initColsControl();
        initStepOverlays();
        initZoom();
    });

    // ---------- per-row control ----------

    function initColsControl() {
        const input = document.getElementById('steps-per-row');
        const grids = document.querySelectorAll('.steps-grid');
        if (!input || !grids.length) return;
        const apply = v => grids.forEach(g => g.style.setProperty('--steps-cols', v));

        // Comparing already splits the width in two, so the stored preference —
        // set while viewing a single run — would overflow each column. The
        // template's own default wins there.
        const comparing = document.querySelector('.run-steps.comparing');
        const stored = comparing ? NaN : parseInt(safeGet(COLS_KEY), 10);
        const initial = clamp(stored || parseInt(input.value, 10) || DEFAULT_COLS, MIN_COLS, MAX_COLS);
        input.value = initial;
        apply(initial);

        input.addEventListener('input', () => {
            const v = clamp(parseInt(input.value, 10) || DEFAULT_COLS, MIN_COLS, MAX_COLS);
            input.value = v;
            apply(v);
            if (!comparing) safeSet(COLS_KEY, String(v));
        });
    }

    // ---------- action parsing ----------

    function parseAction(raw) {
        if (typeof raw !== 'string') return null;
        const s = raw.trim();
        if (!s) return null;

        let m = s.match(/pyautogui\.(click|doubleClick|rightClick|moveTo|dragTo|mouseDown|mouseUp)\s*\(\s*(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s*,\s*(?:y\s*=\s*)?(-?\d+(?:\.\d+)?)/);
        if (m) {
            const fn = m[1];
            const kind = fn === 'moveTo'      ? 'move'
                       : fn === 'dragTo'      ? 'drag'
                       : fn === 'doubleClick' ? 'dblclick'
                       : fn === 'rightClick'  ? 'rclick'
                       : 'click';
            return { kind, x: +m[2], y: +m[3], label: fn };
        }

        // scroll(amount) or scroll(amount, x=.., y=..)
        m = s.match(/pyautogui\.scroll\s*\(\s*(-?\d+)(?:[^)]*?x\s*=\s*(-?\d+)[^)]*?y\s*=\s*(-?\d+))?/);
        if (m) {
            const amount = +m[1];
            const info = { kind: 'scroll', amount, label: `scroll ${amount}` };
            if (m[2] && m[3]) { info.x = +m[2]; info.y = +m[3]; }
            return info;
        }

        m = s.match(/pyautogui\.(?:typewrite|write)\s*\(\s*(['"])([\s\S]*?)\1/);
        if (m) return { kind: 'type', text: m[2], label: `type: ${truncate(m[2], 80)}` };

        m = s.match(/pyautogui\.press\s*\(\s*(['"])([\s\S]*?)\1/);
        if (m) return { kind: 'press', keys: [m[2]], label: `press ${m[2]}` };

        m = s.match(/pyautogui\.hotkey\s*\(\s*([\s\S]+?)\s*\)/);
        if (m) {
            const keys = m[1].split(',').map(k => k.trim().replace(/^['"]|['"]$/g, ''));
            return { kind: 'hotkey', keys, label: keys.join(' + ') };
        }

        m = s.match(/^(DONE|FAIL|WAIT)\b/i);
        if (m) return { kind: m[1].toLowerCase(), label: m[1].toUpperCase() };

        return { kind: 'other', label: truncate(s, 120) };
    }

    // ---------- overlay rendering ----------

    function initStepOverlays() {
        // Each run carries its own action list on its grid, so a comparison
        // draws A's overlays from A's actions and B's from B's.
        document.querySelectorAll('.steps-grid').forEach(grid => initGridOverlays(grid));
    }

    function initGridOverlays(grid) {
        let actions = [];
        try {
            const parsed = JSON.parse(grid.dataset.actions || '[]');
            if (Array.isArray(parsed)) actions = parsed;
        } catch (e) { /* leave empty: overlays are an enhancement, not the content */ }
        grid.querySelectorAll('.step-card').forEach(card => {
            const idx = parseInt(card.dataset.stepIndex, 10);
            const wrap = card.querySelector('.step-image-wrap');
            if (!wrap) return;
            const img = wrap.querySelector('img');
            const svg = wrap.querySelector('svg');
            const action = parseAction(actions[idx]);
            if (!img || !svg || !action) return;

            // For `dragTo` the start is implicit — it's the cursor position
            // established by the previous action. In practice that previous
            // action is `moveTo`; anything else is ignored (fall back to the
            // default offset arrow).
            if (action.kind === 'drag' && idx > 0) {
                const prev = parseAction(actions[idx - 1]);
                if (prev && prev.kind === 'move') {
                    action.startX = prev.x;
                    action.startY = prev.y;
                }
            }

            const apply = () => {
                if (!img.naturalWidth || !img.naturalHeight) return;
                drawOverlay(svg, action, img.naturalWidth, img.naturalHeight);
            };
            if (img.complete) apply();
            else img.addEventListener('load', apply, { once: true });
        });
    }

    function drawOverlay(svg, action, w, h) {
        svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
        const r = Math.max(14, Math.min(w, h) * 0.018);
        const stroke = Math.max(3, r * 0.22);
        const fontSize = Math.max(22, Math.min(w, h) * 0.024);
        const bigFontSize = Math.max(44, Math.min(w, h) * 0.06);
        const color = KIND_COLORS[action.kind] || KIND_COLORS.other;

        switch (action.kind) {
            case 'click': case 'dblclick': case 'rclick':
                svg.innerHTML = targetIcon(action.x, action.y, color, r, stroke) +
                                coordLabel(action.label, action.x, action.y, color, fontSize, stroke, r);
                return;

            case 'move':
                svg.innerHTML = arrowIcon(action.x, action.y, color, r, stroke) +
                                coordLabel(action.label, action.x, action.y, color, fontSize, stroke, r);
                return;

            case 'drag': {
                const hasStart = Number.isFinite(action.startX) && Number.isFinite(action.startY);
                const path = hasStart
                    ? arrowShape(action.startX, action.startY, action.x, action.y, color, stroke, r * 1.6) +
                      `<circle cx="${action.startX}" cy="${action.startY}" r="${r * 0.55}"
                               fill="${color}" stroke="white" stroke-width="${stroke * 0.45}"/>`
                    : arrowIcon(action.x, action.y, color, r, stroke);
                svg.innerHTML = path +
                                `<circle cx="${action.x}" cy="${action.y}" r="${r * 0.55}"
                                         fill="white" stroke="${color}" stroke-width="${stroke * 0.9}"/>` +
                                coordLabel(action.label, action.x, action.y, color, fontSize, stroke, r);
                return;
            }

            case 'scroll': {
                if (Number.isFinite(action.x) && Number.isFinite(action.y)) {
                    svg.innerHTML = scrollIcon(action.x, action.y, color, r, stroke, action.amount) +
                                    coordLabel(action.label, action.x, action.y, color, fontSize, stroke, r);
                } else {
                    svg.innerHTML = pillBadge(action.label, w, h, fontSize, color, '#fff', '🖱', 'bottom');
                }
                return;
            }

            case 'type':
                svg.innerHTML = pillBadge(action.label, w, h, fontSize, 'rgba(17,24,39,0.92)', '#fff', '⌨', 'bottom');
                return;

            case 'press': case 'hotkey':
                svg.innerHTML = keyBadge(action.label, w, h, fontSize);
                return;

            case 'done':
                svg.innerHTML = statusBanner(action.label, w, h, bigFontSize, color, '✓');
                return;

            case 'fail':
                svg.innerHTML = statusBanner(action.label, w, h, bigFontSize, color, '✕');
                return;

            case 'wait':
                svg.innerHTML = statusBanner(action.label, w, h, bigFontSize, color, '⏳');
                return;

            default:
                svg.innerHTML = pillBadge(action.label, w, h, fontSize, 'rgba(17,24,39,0.92)', '#fff', '•', 'top');
        }
    }

    // ---------- svg primitives ----------

    function targetIcon(x, y, color, r, stroke) {
        return `
            <circle cx="${x}" cy="${y}" r="${r * 2.2}" fill="${color}" opacity="0.15"/>
            <circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}"/>
            <circle cx="${x}" cy="${y}" r="${r * 0.3}" fill="${color}"/>`;
    }

    function arrowIcon(x, y, color, r, stroke) {
        const tail = r * 3.5;
        return arrowShape(x - tail, y - tail, x, y, color, stroke, r * 1.6) +
               `<circle cx="${x}" cy="${y}" r="${r * 0.4}" fill="${color}"/>`;
    }

    function scrollIcon(x, y, color, r, stroke, amount) {
        // pyautogui: negative amount = scroll down.
        const down = amount < 0;
        const arrowLen = r * 2.4;
        const y1 = down ? y - arrowLen : y + arrowLen;
        const y2 = down ? y + arrowLen : y - arrowLen;
        return `
            <circle cx="${x}" cy="${y}" r="${r * 2.4}" fill="${color}" opacity="0.18"/>
            <circle cx="${x}" cy="${y}" r="${r * 1.5}" fill="none" stroke="${color}" stroke-width="${stroke}"/>` +
               arrowShape(x, y1, x, y2, color, stroke, r);
    }

    function arrowShape(x1, y1, x2, y2, color, stroke, head) {
        const dx = x2 - x1, dy = y2 - y1;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const px = -uy, py = ux;
        const hx1 = x2 - ux * head + px * head * 0.55;
        const hy1 = y2 - uy * head + py * head * 0.55;
        const hx2 = x2 - ux * head - px * head * 0.55;
        const hy2 = y2 - uy * head - py * head * 0.55;
        return `
            <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
                  stroke="${color}" stroke-width="${stroke}" stroke-linecap="round"/>
            <polygon points="${x2},${y2} ${hx1},${hy1} ${hx2},${hy2}" fill="${color}"/>`;
    }

    function coordLabel(text, x, y, color, fontSize, stroke, r) {
        return `
            <text x="${x + r + 6}" y="${y - r - 4}" font-size="${fontSize}" font-weight="700"
                  fill="${color}" stroke="white" stroke-width="${stroke * 0.5}" paint-order="stroke"
                  font-family="Segoe UI, Arial, sans-serif">${escapeXML(text)}</text>`;
    }

    function pillBadge(text, w, h, fontSize, bg, fg, glyph, position) {
        const padX = fontSize * 0.8;
        const padY = fontSize * 0.4;
        const glyphW = glyph ? fontSize * 1.4 : 0;
        const textW = Math.min(w * 0.9, text.length * fontSize * 0.55);
        const boxW = textW + padX * 2 + glyphW;
        const boxH = fontSize + padY * 2;
        const x = (w - boxW) / 2;
        const y = position === 'top' ? h * 0.04 : h - boxH - h * 0.04;
        const textX = x + padX + glyphW;
        const textY = y + boxH / 2 + fontSize * 0.34;
        return `
            <rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="${boxH/2}" ry="${boxH/2}"
                  fill="${bg}" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>
            ${glyph ? `<text x="${x + padX + glyphW * 0.45}" y="${textY}"
                            font-size="${fontSize * 1.1}" text-anchor="middle" fill="${fg}"
                            font-family="Segoe UI Emoji, Apple Color Emoji, Segoe UI Symbol, sans-serif">${escapeXML(glyph)}</text>` : ''}
            <text x="${textX}" y="${textY}" font-size="${fontSize}" font-weight="700" fill="${fg}"
                  font-family="Segoe UI, Arial, sans-serif">${escapeXML(text)}</text>`;
    }

    function keyBadge(text, w, h, fontSize) {
        // Key-cap styled badge for `press` / `hotkey`.
        const padX = fontSize * 0.9, padY = fontSize * 0.55;
        const textW = Math.min(w * 0.85, text.length * fontSize * 0.62);
        const boxW = textW + padX * 2;
        const boxH = fontSize + padY * 2;
        const x = (w - boxW) / 2;
        const y = h - boxH - h * 0.06;
        return `
            <rect x="${x}" y="${y + 4}" width="${boxW}" height="${boxH}" rx="10" ry="10"
                  fill="rgba(15,23,42,0.55)"/>
            <rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="10" ry="10"
                  fill="#f8fafc" stroke="#111827" stroke-width="2.5"/>
            <text x="${w/2}" y="${y + boxH/2 + fontSize * 0.34}" text-anchor="middle"
                  font-size="${fontSize}" font-weight="700" fill="#111827"
                  font-family="Consolas, Menlo, monospace">${escapeXML(text)}</text>`;
    }

    function statusBanner(text, w, h, fontSize, color, glyph) {
        const padX = fontSize * 0.9, padY = fontSize * 0.45;
        const glyphW = fontSize * 1.4;
        const textW = Math.min(w * 0.7, text.length * fontSize * 0.7);
        const boxW = textW + padX * 2 + glyphW;
        const boxH = fontSize + padY * 2;
        const x = (w - boxW) / 2;
        const y = (h - boxH) / 2;
        return `
            <rect x="0" y="0" width="${w}" height="${h}" fill="${color}" opacity="0.14"/>
            <rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="14" ry="14"
                  fill="${color}" opacity="0.94"/>
            <text x="${x + padX + glyphW * 0.45}" y="${y + boxH/2 + fontSize*0.34}"
                  text-anchor="middle" font-size="${fontSize * 1.1}" fill="#fff"
                  font-family="Segoe UI Symbol, Apple Color Emoji, sans-serif">${escapeXML(glyph)}</text>
            <text x="${x + padX + glyphW}" y="${y + boxH/2 + fontSize*0.34}"
                  font-size="${fontSize}" font-weight="800" fill="#fff" letter-spacing="2"
                  font-family="Segoe UI, Arial, sans-serif">${escapeXML(text)}</text>`;
    }

    // ---------- zoom modal ----------

    function initZoom() {
        const modal = document.getElementById('zoom-modal');
        const zoomImg = document.getElementById('zoom-image');
        const zoomSvg = document.getElementById('zoom-overlay');
        const closeBtn = modal && modal.querySelector('.zoom-close');
        if (!modal || !zoomImg || !zoomSvg) return;

        document.querySelectorAll('.step-image-wrap').forEach(wrap => {
            wrap.addEventListener('click', () => {
                const img = wrap.querySelector('img');
                const svg = wrap.querySelector('svg');
                if (!img) return;
                zoomImg.src = img.src;
                zoomImg.alt = img.alt;
                if (svg) {
                    zoomSvg.innerHTML = svg.innerHTML;
                    const vb = svg.getAttribute('viewBox');
                    if (vb) zoomSvg.setAttribute('viewBox', vb);
                } else {
                    zoomSvg.innerHTML = '';
                }
                modal.classList.add('open');
                document.body.style.overflow = 'hidden';
            });
        });

        const close = () => {
            modal.classList.remove('open');
            document.body.style.overflow = '';
            zoomImg.removeAttribute('src');
            zoomSvg.innerHTML = '';
        };

        modal.addEventListener('click', (e) => {
            // Close unless user clicked the image itself (keeps the image interactive).
            if (e.target === zoomImg) return;
            close();
        });
        if (closeBtn) closeBtn.addEventListener('click', (e) => { e.stopPropagation(); close(); });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('open')) close();
        });
    }

    // ---------- utilities ----------

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }
    function escapeXML(s) {
        return String(s).replace(/[<>&'"]/g, c =>
            ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' }[c])
        );
    }
    function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
    function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } }
})();
