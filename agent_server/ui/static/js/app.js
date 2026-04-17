/* ══════════════════════════════════════════════
   CoMarketer — Chat Application
   ══════════════════════════════════════════════ */

// ── Highcharts dark theme ──
if (window.Highcharts) {
    Highcharts.setOptions({
        chart: { backgroundColor: '#1e2030', style: { fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' } },
        title: { style: { color: '#e0e0e0' } },
        subtitle: { style: { color: '#9ca3af' } },
        xAxis: { labels: { style: { color: '#9ca3af' } }, gridLineColor: '#2a2d37', lineColor: '#2a2d37' },
        yAxis: { labels: { style: { color: '#9ca3af' } }, gridLineColor: '#2a2d37', title: { style: { color: '#9ca3af' } } },
        legend: { itemStyle: { color: '#e0e0e0' }, itemHoverStyle: { color: '#6366f1' } },
        tooltip: { backgroundColor: '#1a1d27', style: { color: '#e0e0e0' }, borderColor: '#2a2d37' },
        plotOptions: { series: { dataLabels: { style: { color: '#e0e0e0' } } } },
        credits: { enabled: false },
    });
}

// ═══════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════
const chat = document.getElementById('chat');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const convList = document.getElementById('convList');
const convTitleHeader = document.getElementById('convTitleHeader');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');

// ═══════════════════════════════════════════════
// Conversation Storage (localStorage)
// ═══════════════════════════════════════════════
const STORAGE_KEY = 'comarketer_conversations';
const ACTIVE_KEY = 'comarketer_active_conv';
const MAX_CONVERSATIONS = 50;

let conversations = [];
let activeConvId = null;

function loadConversationsFromStorage() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        conversations = raw ? JSON.parse(raw) : [];
    } catch {
        conversations = [];
    }
    activeConvId = localStorage.getItem(ACTIVE_KEY) || null;
}

function saveConversationsToStorage() {
    try {
        // Prune if over limit
        if (conversations.length > MAX_CONVERSATIONS) {
            conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
            conversations = conversations.slice(0, MAX_CONVERSATIONS);
            showToast('Oldest conversation removed to free space', 'info');
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
        if (activeConvId) {
            localStorage.setItem(ACTIVE_KEY, activeConvId);
        }
    } catch (e) {
        if (e.name === 'QuotaExceededError') {
            // Remove oldest conversation and retry
            conversations.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
            conversations.pop();
            showToast('Storage full — oldest conversation removed', 'warning');
            try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); } catch { /* give up */ }
        }
    }
}

function getConversation(id) {
    return conversations.find(c => c.id === id);
}

function generateConvId() {
    return 'conv-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function autoTitle(query) {
    const clean = query.trim().replace(/\n/g, ' ');
    return clean.length > 50 ? clean.slice(0, 50) + '...' : clean;
}

// ═══════════════════════════════════════════════
// Sidebar Rendering
// ═══════════════════════════════════════════════
function renderSidebar() {
    if (!convList) return;

    const now = new Date();
    const todayStr = now.toDateString();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toDateString();
    const weekAgo = new Date(now);
    weekAgo.setDate(weekAgo.getDate() - 7);

    // Sort newest first
    const sorted = [...conversations].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

    const groups = { today: [], yesterday: [], week: [], older: [] };
    sorted.forEach(c => {
        const d = new Date(c.updatedAt);
        const ds = d.toDateString();
        if (ds === todayStr) groups.today.push(c);
        else if (ds === yesterdayStr) groups.yesterday.push(c);
        else if (d > weekAgo) groups.week.push(c);
        else groups.older.push(c);
    });

    let html = '';
    const renderGroup = (label, items) => {
        if (items.length === 0) return '';
        let h = `<div class="conv-group-label">${label}</div>`;
        items.forEach(c => {
            const isActive = c.id === activeConvId ? ' active' : '';
            h += `<div class="conv-item${isActive}" onclick="switchConversation('${c.id}')" title="${escapeHtml(c.title)}">
                <svg class="conv-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span class="conv-title">${escapeHtml(c.title)}</span>
                <button class="conv-delete" onclick="event.stopPropagation(); deleteConversation('${c.id}')" title="Delete">&times;</button>
            </div>`;
        });
        return h;
    };

    html += renderGroup('Today', groups.today);
    html += renderGroup('Yesterday', groups.yesterday);
    html += renderGroup('Last 7 Days', groups.week);
    html += renderGroup('Older', groups.older);

    if (html === '') {
        html = '<div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 13px;">No conversations yet</div>';
    }

    convList.innerHTML = html;
}

function newConversation() {
    activeConvId = null;
    localStorage.removeItem(ACTIVE_KEY);
    renderSidebar();
    showWelcomeScreen();
    updateHeaderTitle('New Conversation');
    userInput.focus();
    closeSidebarMobile();
}

function switchConversation(convId) {
    const conv = getConversation(convId);
    if (!conv) return;

    activeConvId = convId;
    localStorage.setItem(ACTIVE_KEY, convId);
    renderSidebar();

    // Restore messages
    chat.innerHTML = '';
    conv.messages.forEach(msg => {
        if (msg.role === 'user') {
            appendMessageRow('user', escapeHtml(msg.content));
        } else {
            appendMessageRow('assistant', msg.html);
        }
    });
    chat.scrollTop = chat.scrollHeight;
    updateHeaderTitle(conv.title);
    closeSidebarMobile();
}

function deleteConversation(convId) {
    conversations = conversations.filter(c => c.id !== convId);
    saveConversationsToStorage();

    if (activeConvId === convId) {
        newConversation();
    } else {
        renderSidebar();
    }
}

function updateHeaderTitle(title) {
    if (convTitleHeader) {
        convTitleHeader.textContent = title;
    }
}

// ═══════════════════════════════════════════════
// Sidebar Toggle
// ═══════════════════════════════════════════════
function toggleSidebar() {
    if (!sidebar) return;
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
        sidebar.classList.toggle('open');
    } else {
        sidebar.classList.toggle('collapsed');
    }
}

function closeSidebarMobile() {
    if (window.innerWidth <= 768 && sidebar) {
        sidebar.classList.remove('open');
    }
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebarMobile);
}

// ═══════════════════════════════════════════════
// Welcome Screen
// ═══════════════════════════════════════════════
function showWelcomeScreen() {
    chat.innerHTML = `
        <div class="welcome-screen" id="welcome">
            <div class="welcome-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 3v18h18"/><path d="M7 16l4-8 4 5 5-9"/>
                </svg>
            </div>
            <div class="welcome-title">Campaign Analytics Assistant</div>
            <div class="welcome-subtitle">Ask questions about your campaign performance across Email, SMS, WhatsApp, and more.</div>
            <div class="example-grid">
                <div class="example-card" onclick="askExample(this)">
                    <span class="card-icon">&#128200;</span>
                    <span class="card-text">Show me top 5 email campaigns by open rate last month</span>
                </div>
                <div class="example-card" onclick="askExample(this)">
                    <span class="card-icon">&#128202;</span>
                    <span class="card-text">Compare Email and WhatsApp campaigns last quarter</span>
                </div>
                <div class="example-card" onclick="askExample(this)">
                    <span class="card-icon">&#128241;</span>
                    <span class="card-text">What was my SMS performance in January 2025?</span>
                </div>
            </div>
        </div>
    `;
}

function askExample(el) {
    const text = el.querySelector('.card-text').textContent;
    userInput.value = text;
    autoResize(userInput);
    sendMessage();
}

// ═══════════════════════════════════════════════
// Message Rendering
// ═══════════════════════════════════════════════
function appendMessageRow(role, contentHtml) {
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.remove();

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    if (role === 'user') {
        row.innerHTML = `
            <div class="avatar user-avatar">U</div>
            <div class="message-bubble user-bubble">${contentHtml}</div>
        `;
    } else {
        row.innerHTML = `
            <div class="avatar bot-avatar">CM</div>
            <div class="message-bubble bot-bubble">${contentHtml}</div>
        `;
    }

    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return row;
}

function showTypingIndicator() {
    const row = appendMessageRow('assistant',
        `<div class="typing-indicator">
            <span class="typing-label">Analyzing your data</span>
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>`
    );
    row.id = 'typing-row';
    return row;
}

// ═══════════════════════════════════════════════
// Input Area
// ═══════════════════════════════════════════════
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// ═══════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ═══════════════════════════════════════════════
// Markdown / Table / Insights Parsing
// ═══════════════════════════════════════════════
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function parseTable(md) {
    const lines = md.trim().split('\n').filter(l => l.trim());
    if (lines.length < 2) return `<pre>${escapeHtml(md)}</pre>`;

    const parseRow = line => line.split('|').map(c => c.trim()).filter(c => c !== '');
    const headers = parseRow(lines[0]);

    let dataStart = 1;
    if (lines[1] && /^[\s|:\-]+$/.test(lines[1])) dataStart = 2;

    let html = '<div class="table-wrapper"><table><thead><tr>';
    const skipIdx = headers[0] === '' || /^\d+$/.test(headers[0].trim());
    const startCol = skipIdx ? 1 : 0;

    for (let i = startCol; i < headers.length; i++) {
        html += `<th>${escapeHtml(headers[i])}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (let r = dataStart; r < lines.length; r++) {
        const cols = parseRow(lines[r]);
        html += '<tr>';
        for (let i = startCol; i < cols.length; i++) {
            html += `<td>${escapeHtml(cols[i] || '')}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table></div>';
    return html;
}

function parseMd(text) {
    text = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    // Markdown → HTML conversion
    text = text
        .replace(/^#{4,5}\s+(.+)$/gm, '<h4>$1</h4>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[(.+?)\]/g, '<strong>$1</strong>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>')
        .replace(/(<li>.*<\/li>\n?)+/gs, '<ul>$&</ul>');
    // Newlines → <br>, then strip redundant <br> around block elements.
    // Headings (<h2>-<h4>) and lists (<ul>) have CSS margins already —
    // extra <br> tags between them cause ugly double-spacing.
    text = text
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/(<\/h[1-5]>)(<br\s*\/?>)+/gi, '$1')
        .replace(/(<br\s*\/?>)+(<h[1-5]>)/gi, '$2')
        .replace(/(<\/ul>)(<br\s*\/?>)+/gi, '$1')
        .replace(/(<br\s*\/?>)+(<ul>)/gi, '$2')
        .replace(/(<br\s*\/?>){3,}/gi, '<br><br>');
    return text;
}

function parseInsights(raw) {
    try {
        const data = JSON.parse(raw);
        if (data.rows && data.rows.length > 0) {
            const insIdx = data.columns.indexOf('insights');
            if (insIdx >= 0) {
                let insightsText = data.rows[0][insIdx];
                const items = insightsText.replace(/^\[/, '').replace(/\]$/, '').split(/\.,\s*/);
                let html = '';
                items.forEach(item => {
                    item = item.trim().replace(/\.$/, '');
                    if (item) {
                        html += `<div class="insight-item">${escapeHtml(item)}.</div>`;
                    }
                });
                return `<div class="insights-block">${html}</div>`;
            }
        }
        return `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    } catch {
        return `<pre>${escapeHtml(raw)}</pre>`;
    }
}

// ═══════════════════════════════════════════════
// Response Rendering
// ═══════════════════════════════════════════════
function formatCellValue(cell) {
    if (typeof cell === 'number') {
        if (cell % 1 === 0 && cell > 999) return cell.toLocaleString();
        if (cell % 1 !== 0) return cell.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return cell;
}

function buildTableHtml(val) {
    if (typeof val === 'object' && val.tableHeaders) {
        let tbl = '<div class="table-wrapper"><table><thead><tr>';
        val.tableHeaders.forEach(h => { tbl += `<th>${escapeHtml(String(h))}</th>`; });
        tbl += '</tr></thead><tbody>';
        (val.data || []).forEach(row => {
            tbl += '<tr>';
            row.forEach(cell => { tbl += `<td>${escapeHtml(String(formatCellValue(cell)))}</td>`; });
            tbl += '</tr>';
        });
        tbl += '</tbody></table></div>';
        return tbl;
    }
    return parseTable(String(val));
}

function buildItemsHtml(items) {
    let html = '';
    items.forEach(item => {
        html += '<div class="response-section">';
        const val = item.value;
        switch (item.type) {
            case 'text':
                html += `<div class="section-label">Analysis</div>`;
                html += `<div class="text-content">${parseMd(String(val))}</div>`;
                break;
            case 'table':
                html += `<div class="section-label">Data</div>`;
                html += buildTableHtml(val);
                break;
            case 'chart':
                const chartId = `chart-${item.id || 'c' + Date.now() + Math.random().toString(36).slice(2, 6)}`;
                html += `<div class="section-label">Chart</div>`;
                html += `<div id="${chartId}" style="width:100%;min-height:350px;background:#1e2030;border-radius:8px;padding:8px;"></div>`;
                try {
                    const chartCfg = typeof val === 'string' ? JSON.parse(val) : val;
                    if (window.Highcharts) {
                        setTimeout(() => {
                            try {
                                Highcharts.chart(chartId, chartCfg);
                            } catch (renderErr) {
                                const el = document.getElementById(chartId);
                                if (el) el.innerHTML = `<div style="color:#f87171;padding:16px;">Chart render failed: ${renderErr.message}</div>`;
                            }
                        }, 150);
                    }
                } catch (parseErr) {
                    console.error('Chart JSON parse error:', parseErr);
                }
                break;
            case 'sql':
                html += `<div class="collapsible-header" onclick="toggleCollapse(this)">`;
                html += `<span class="arrow">&#9654;</span>`;
                html += `<div class="section-label" style="margin:0">SQL Query</div></div>`;
                html += `<div class="collapsible-body"><div class="sql-block">${escapeHtml(String(val))}</div></div>`;
                break;
            case 'insights':
                html += `<div class="collapsible-header" onclick="toggleCollapse(this)">`;
                html += `<span class="arrow">&#9654;</span>`;
                html += `<div class="section-label" style="margin:0">Insights &amp; Recommendations</div></div>`;
                html += `<div class="collapsible-body">${parseInsights(val)}</div>`;
                break;
            case 'collapsedText':
                html += `<div class="collapsible-header" onclick="toggleCollapse(this)">`;
                html += `<span class="arrow">&#9654;</span>`;
                html += `<div class="section-label" style="margin:0">${escapeHtml(item.name || item.type)}</div></div>`;
                html += `<div class="collapsible-body"><div class="sql-block">${escapeHtml(String(val))}</div></div>`;
                break;
            default:
                html += `<div class="section-label">${escapeHtml(item.type)}</div>`;
                html += `<div class="text-content">${parseMd(String(val))}</div>`;
        }
        html += '</div>';
    });
    return html;
}

function toggleCollapse(el) {
    el.classList.toggle('open');
    const body = el.nextElementSibling;
    body.classList.toggle('open');
}

// ═══════════════════════════════════════════════
// Feedback System
// ═══════════════════════════════════════════════
function addFeedbackBar(containerEl, traceId) {
    const bar = document.createElement('div');
    bar.className = 'feedback-bar';
    const safeTraceId = escapeHtml(traceId);
    bar.innerHTML = `
        <span class="feedback-label">Was this response helpful?</span>
        <button class="fb-btn fb-up" title="Helpful" aria-label="Helpful" onclick="handleFeedback(this, true, '${safeTraceId}')">&#128077;</button>
        <button class="fb-btn fb-down" title="Not helpful" aria-label="Not helpful" onclick="handleFeedback(this, false, '${safeTraceId}')">&#128078;</button>
        <div class="fb-comment" style="display:none">
            <span class="fb-comment-label">Tell us more (optional):</span>
            <textarea class="fb-textarea" placeholder="What could be improved?" maxlength="500" oninput="this.parentElement.querySelector('.fb-char-count').textContent=this.value.length+'/500'"></textarea>
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="fb-char-count">0/500</span>
                <button class="fb-submit" onclick="submitFeedback(this, '${safeTraceId}')">Submit feedback</button>
            </div>
        </div>
        <span class="fb-confirmation" style="display:none">Feedback recorded</span>
    `;
    containerEl.appendChild(bar);
}

function handleFeedback(btn, isHelpful, traceId) {
    const bar = btn.closest('.feedback-bar');
    if (!bar) return;

    const upBtn = bar.querySelector('.fb-up');
    const downBtn = bar.querySelector('.fb-down');
    const commentDiv = bar.querySelector('.fb-comment');
    if (!commentDiv) return;

    if (upBtn) upBtn.classList.remove('active-up', 'active-down');
    if (downBtn) downBtn.classList.remove('active-up', 'active-down');
    btn.classList.add(isHelpful ? 'active-up' : 'active-down');
    bar.dataset.isHelpful = isHelpful ? 'true' : 'false';

    if (isHelpful) {
        commentDiv.style.display = 'none';
        submitFeedbackData(bar, traceId, true, null);
    } else {
        commentDiv.style.display = 'flex';
        const textarea = commentDiv.querySelector('.fb-textarea');
        if (textarea) textarea.focus();
    }
}

function submitFeedback(submitBtn, traceId) {
    const bar = submitBtn.closest('.feedback-bar');
    const comment = bar.querySelector('.fb-textarea').value.trim() || null;
    const isHelpful = bar.dataset.isHelpful !== 'false';
    submitFeedbackData(bar, traceId, isHelpful, comment);
}

async function submitFeedbackData(bar, traceId, isHelpful, comment) {
    const upBtn = bar.querySelector('.fb-up');
    const downBtn = bar.querySelector('.fb-down');
    const submitBtn = bar.querySelector('.fb-submit');
    const confirmation = bar.querySelector('.fb-confirmation');

    if (upBtn) upBtn.disabled = true;
    if (downBtn) downBtn.disabled = true;
    if (submitBtn) submitBtn.disabled = true;

    try {
        const resp = await fetch('/ui/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trace_id: traceId,
                is_helpful: isHelpful,
                comment: comment,
                user_id: 'ui-user',
            }),
        });
        const result = await resp.json();
        if (resp.ok && result.status === 'recorded') {
            if (confirmation) confirmation.style.display = 'inline';
            showToast('Feedback recorded', 'success');
            setTimeout(() => { if (confirmation) confirmation.style.display = 'none'; }, 3000);
        } else {
            showToast('Failed to submit feedback', 'error');
            if (upBtn) upBtn.disabled = false;
            if (downBtn) downBtn.disabled = false;
            if (submitBtn) submitBtn.disabled = false;
        }
    } catch {
        showToast('Network error — feedback not sent', 'error');
        if (upBtn) upBtn.disabled = false;
        if (downBtn) downBtn.disabled = false;
        if (submitBtn) submitBtn.disabled = false;
    }
}

// ═══════════════════════════════════════════════
// Send Message (SSE Streaming)
// ═══════════════════════════════════════════════
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    const spId = document.getElementById('client').value;
    const clientName = document.getElementById('client').selectedOptions[0]?.text || 'Unknown';

    // Create or get conversation
    let conv;
    if (activeConvId) {
        conv = getConversation(activeConvId);
    }
    if (!conv) {
        const convId = generateConvId();
        conv = {
            id: convId,
            title: autoTitle(query),
            spId: spId,
            clientName: clientName,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            messages: [],
        };
        conversations.push(conv);
        activeConvId = convId;
    }

    // Save user message
    conv.messages.push({ role: 'user', content: query });
    conv.updatedAt = new Date().toISOString();
    saveConversationsToStorage();
    renderSidebar();
    updateHeaderTitle(conv.title);

    // Render user message
    appendMessageRow('user', escapeHtml(query));
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    const typingRow = showTypingIndicator();
    const t0 = performance.now();
    let eventCount = 0;
    let currentTraceId = '';
    let bubbleEl = null;

    function elapsed() {
        return ((performance.now() - t0) / 1000).toFixed(1);
    }

    function ensureBubble() {
        if (!bubbleEl) {
            if (typingRow.parentNode) typingRow.remove();
            const row = appendMessageRow('assistant', '');
            bubbleEl = row.querySelector('.bot-bubble');
        }
        return bubbleEl;
    }

    function appendEvent(stageClass, stageLabel, itemsHtml) {
        const el = ensureBubble();
        if (eventCount > 0) {
            el.insertAdjacentHTML('beforeend', '<div class="stream-separator"></div>');
        }
        const header = `<div class="stream-event-header">` +
            `<span class="stream-stage ${stageClass}">${stageLabel}</span>` +
            `<span class="stream-timing">${elapsed()}s</span>` +
            `</div>`;
        el.insertAdjacentHTML('beforeend', header + itemsHtml);
        chat.scrollTop = chat.scrollHeight;
        eventCount++;
    }

    try {
        const resp = await fetch('/invocations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: [{ role: 'user', content: query }],
                custom_inputs: {
                    sp_id: spId,
                    user_name: 'UI User',
                    user_id: 'ui-user',
                    conversation_id: conv.id,
                    task_type: 'analytics',
                },
                stream: true,
            }),
        });

        if (!resp.ok) {
            if (typingRow.parentNode) typingRow.remove();
            const errText = await resp.text();
            appendMessageRow('assistant', `<div class="error-block">HTTP ${resp.status}: ${escapeHtml(errText)}</div>`);
            showToast(`Request failed: HTTP ${resp.status}`, 'error');
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let obsCount = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6).trim();
                if (!payload || payload === '[DONE]') continue;

                let event;
                try { event = JSON.parse(payload); } catch { continue; }

                if (event.error) {
                    ensureBubble();
                    bubbleEl.insertAdjacentHTML('beforeend',
                        `<div class="response-section"><div class="error-block">${escapeHtml(String(event.error))}</div></div>`);
                    continue;
                }

                const co = event.custom_outputs || {};

                if (co.type === 'TRACE_DONE') {
                    currentTraceId = co.mlflow_trace_id || '';
                    if (bubbleEl && currentTraceId) {
                        addFeedbackBar(bubbleEl, currentTraceId);
                    }
                    continue;
                }

                // SSE keepalive — ignore heartbeat events
                if (co.type === 'HEARTBEAT') continue;

                const item = event.item || {};
                const text = (item.content || [{}])[0]?.text || '';
                if (!text) continue;

                let items = [];
                try {
                    const parsed = JSON.parse(text);
                    const raw = parsed.items || [];
                    items = Array.isArray(raw[0]) ? raw[0] : raw;
                } catch {
                    items = [{ type: 'text', value: text }];
                }

                const html = buildItemsHtml(items);

                if (co.type === 'RATIONALE') {
                    const el = ensureBubble();
                    const thoughtText = items[0]?.value || 'Analyzing your query...';
                    // Find or create the collapsible thought-process container
                    let thoughtBox = el.querySelector('.thought-process');
                    if (!thoughtBox) {
                        el.insertAdjacentHTML('beforeend',
                            `<details class="thought-process" open>` +
                            `<summary class="thought-process-toggle">Thought Process</summary>` +
                            `<div class="thought-process-body"></div>` +
                            `</details>`
                        );
                        thoughtBox = el.querySelector('.thought-process');
                    }
                    const body = thoughtBox.querySelector('.thought-process-body');
                    body.insertAdjacentHTML('beforeend',
                        `<div class="thought-step"><span class="thought-dot"></span>${escapeHtml(thoughtText)}</div>`
                    );
                    chat.scrollTop = chat.scrollHeight;
                } else if (co.type === 'observation') {
                    obsCount++;
                    const types = items.map(i => i.type);
                    let label, cls;
                    if (types.includes('chart')) {
                        label = 'Analysis & Chart';
                        cls = 'analysis';
                    } else if (types.includes('table')) {
                        label = 'Data';
                        cls = 'data';
                    } else if (obsCount === 1) {
                        label = 'Data';
                        cls = 'data';
                    } else {
                        label = 'Analysis';
                        cls = 'analysis';
                    }
                    appendEvent(cls, label, html);
                } else {
                    appendEvent('data', co.type || 'Event', html);
                }
            }
        }

        if (typingRow.parentNode) typingRow.remove();

        if (bubbleEl) {
            bubbleEl.insertAdjacentHTML('beforeend',
                `<div style="text-align:right;margin-top:8px;">` +
                `<span class="stream-timing">Total: ${elapsed()}s | ${eventCount} events</span></div>`
            );

            // Save assistant response to conversation
            conv.messages.push({ role: 'assistant', html: bubbleEl.innerHTML, traceId: currentTraceId });
            conv.updatedAt = new Date().toISOString();
            saveConversationsToStorage();
            renderSidebar();
        }

    } catch (err) {
        if (typingRow.parentNode) typingRow.remove();
        appendMessageRow('assistant', `<div class="error-block">Error: ${escapeHtml(err.message)}</div>`);
        showToast('Connection error', 'error');
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// ═══════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    loadConversationsFromStorage();
    renderSidebar();

    if (activeConvId && getConversation(activeConvId)) {
        switchConversation(activeConvId);
    } else {
        showWelcomeScreen();
    }

    userInput.focus();
});
