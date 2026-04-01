/**
 * dashboard.js - Analytics Dashboard
 * Auto-refreshing stats, Chart.js visualisations, conversation table
 */

const REFRESH_INTERVAL = 30_000; // 30 s

let charts = {};
let refreshTimer = null;

// ─── Init ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  refreshTimer = setInterval(loadDashboard, REFRESH_INTERVAL);
  document.getElementById('refresh-btn')?.addEventListener('click', () => {
    clearInterval(refreshTimer);
    loadDashboard();
    refreshTimer = setInterval(loadDashboard, REFRESH_INTERVAL);
    showToast('Dashboard refreshed ✓', 'success');
  });
});

async function loadDashboard() {
  try {
    const [stats, intentData, convData, modelInfo, unknownData] = await Promise.all([
      fetchJSON('/api/analytics/dashboard'),
      fetchJSON('/api/analytics/intents'),
      fetchJSON('/api/analytics/conversations?limit=10'),
      fetchJSON('/api/analytics/model-info'),
      fetchJSON('/api/analytics/unknown-queries')
    ]);

    renderOverview(stats.overview);
    renderIntentChart(stats.intent_distribution);
    renderSentimentChart(stats.sentiment_distribution);
    renderDailyChart(stats.daily_messages);
    renderIntentTable(intentData.intent_performance);
    renderConversationsTable(convData.conversations);
    renderModelInfo(modelInfo);
    renderUnknownQueries(unknownData.unknown_queries);

  } catch (err) {
    console.error('Dashboard load error:', err);
    showToast('Failed to load analytics data.', 'error');
  }
}

// ─── Overview Stats ───────────────────────────────────
function renderOverview(o) {
  if (!o) return;
  setEl('stat-sessions',       fmtNum(o.total_sessions));
  setEl('stat-messages',       fmtNum(o.total_messages));
  setEl('stat-today-sessions', fmtNum(o.today_sessions));
  setEl('stat-today-msgs',     fmtNum(o.today_messages));
  setEl('stat-confidence',     (o.avg_confidence || 0).toFixed(1) + '%');
  setEl('stat-rating',         (o.avg_rating || 0).toFixed(2) + ' ⭐');
  setEl('stat-feedback',       fmtNum(o.total_feedback));
  setEl('stat-escalations',    fmtNum(o.total_escalations));
  setEl('stat-urgent',         fmtNum(o.urgent_messages));
  setEl('stat-fallback',       (o.fallback_rate || 0).toFixed(1) + '%');
  setEl('stat-response-time',  (o.avg_response_time_ms || 0).toFixed(0) + 'ms');
  setEl('stat-low-conf',       (o.low_confidence_rate || 0).toFixed(1) + '%');
}

// ─── Intent Distribution Bar Chart ───────────────────
function renderIntentChart(data) {
  if (!data || data.length === 0) return;
  const ctx = document.getElementById('intent-chart')?.getContext('2d');
  if (!ctx) return;

  const labels = data.map(d => d.intent.replace(/_/g, ' '));
  const values = data.map(d => d.count);
  const palette = generatePalette(data.length);

  if (charts.intent) charts.intent.destroy();

  charts.intent = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Message Count',
        data: values,
        backgroundColor: palette.map(c => c + 'cc'),
        borderColor: palette,
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y} messages`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#8b949e', font: { size: 10 }, maxRotation: 45 },
          grid: { color: 'rgba(48,54,61,0.5)' }
        },
        y: {
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: 'rgba(48,54,61,0.5)' },
          beginAtZero: true
        }
      }
    }
  });
}

// ─── Sentiment Doughnut Chart ─────────────────────────
function renderSentimentChart(data) {
  if (!data || data.length === 0) return;
  const ctx = document.getElementById('sentiment-chart')?.getContext('2d');
  if (!ctx) return;

  const colourMap = { positive: '#3fb950', neutral: '#8b949e', negative: '#f85149' };
  const labels = data.map(d => d.sentiment || 'neutral');
  const values = data.map(d => d.count);
  const colours = labels.map(l => colourMap[l] || '#58a6ff');

  if (charts.sentiment) charts.sentiment.destroy();

  charts.sentiment = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colours.map(c => c + 'bb'),
        borderColor: colours,
        borderWidth: 2,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#8b949e', font: { size: 11 }, padding: 14, boxWidth: 12 }
        }
      }
    }
  });
}

// ─── Daily Messages Line Chart ────────────────────────
function renderDailyChart(data) {
  if (!data || data.length === 0) return;
  const ctx = document.getElementById('daily-chart')?.getContext('2d');
  if (!ctx) return;

  const labels = data.map(d => {
    const dt = new Date(d.date);
    return dt.toLocaleDateString([], { month: 'short', day: 'numeric' });
  });
  const values = data.map(d => d.count);

  if (charts.daily) charts.daily.destroy();

  charts.daily = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Messages/Day',
        data: values,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        borderWidth: 2,
        pointBackgroundColor: '#58a6ff',
        pointRadius: 4,
        tension: 0.35,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: 'rgba(48,54,61,0.5)' }
        },
        y: {
          ticks: { color: '#8b949e', font: { size: 10 } },
          grid: { color: 'rgba(48,54,61,0.5)' },
          beginAtZero: true
        }
      }
    }
  });
}

// ─── Intent Performance Table ─────────────────────────
function renderIntentTable(data) {
  const tbody = document.getElementById('intent-table-body');
  if (!tbody || !data) return;

  const maxHits = Math.max(...data.map(d => d.total_hits), 1);

  tbody.innerHTML = data.slice(0, 15).map(d => `
    <tr>
      <td>
        <span class="badge badge-blue">${(d.intent || '—').replace(/_/g, ' ')}</span>
      </td>
      <td>
        ${fmtNum(d.total_hits)}
        <div class="progress-bar" style="min-width:80px">
          <div class="progress-fill" style="width:${(d.total_hits / maxHits * 100).toFixed(1)}%"></div>
        </div>
      </td>
      <td>
        <span style="color:${confColour(d.avg_confidence)}">${d.avg_confidence.toFixed(1)}%</span>
      </td>
      <td style="color:var(--text-muted);font-size:0.75rem">${d.last_seen || '—'}</td>
    </tr>`).join('');
}

// ─── Recent Conversations ─────────────────────────────
function renderConversationsTable(data) {
  const tbody = document.getElementById('conv-table-body');
  if (!tbody || !data) return;

  tbody.innerHTML = data.map(c => `
    <tr>
      <td style="font-family:monospace;font-size:0.72rem;color:var(--text-muted)">${c.id.slice(0, 8)}…</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
        ${escapeHtml(c.first_message || '—')}
      </td>
      <td>${c.total_messages}</td>
      <td>${statusBadge(c.resolution_status)}</td>
      <td>${c.avg_rating ? (c.avg_rating.toFixed(1) + ' ⭐') : '—'}</td>
      <td style="font-size:0.72rem;color:var(--text-muted)">${fmtDate(c.created_at)}</td>
    </tr>`).join('');
}

// ─── Model Info ───────────────────────────────────────
function renderModelInfo(info) {
  setEl('model-loaded',   info?.is_loaded  ? '✅ Loaded' : '❌ Not loaded');
  setEl('model-intents',  info?.num_intents ?? '—');
  setEl('model-words',    info?.num_words   ?? '—');

  const meta = info?.metadata || {};
  setEl('model-trained',  meta.trained_at  ? fmtDate(meta.trained_at) : '—');
  setEl('model-accuracy', meta.final_accuracy
    ? (meta.final_accuracy * 100).toFixed(2) + '%' : '—');
  setEl('model-type',     meta.model_type  || 'neural_network');
}

// ─── Unknown Queries ──────────────────────────────────
function renderUnknownQueries(data) {
  const tbody = document.getElementById('unknown-table-body');
  if (!tbody || !data) return;

  tbody.innerHTML = data.slice(0, 10).map(q => `
    <tr>
      <td style="font-style:italic;color:var(--text-secondary)">"${escapeHtml(q.query)}"</td>
      <td>${q.frequency}</td>
      <td>${q.best_guess_intent || '—'}</td>
      <td style="font-size:0.72rem;color:var(--text-muted)">${fmtDate(q.last_seen)}</td>
    </tr>`).join('');
}

// ─── Helpers ──────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? '—';
}

function fmtNum(n) {
  if (n == null) return '0';
  return Number(n).toLocaleString();
}

function fmtDate(str) {
  if (!str) return '—';
  const d = new Date(str);
  return isNaN(d) ? str : d.toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function confColour(val) {
  if (val >= 80) return '#3fb950';
  if (val >= 60) return '#d29922';
  return '#f85149';
}

function statusBadge(status) {
  const map = {
    open:      ['badge-blue',  'Open'],
    resolved:  ['badge-green', 'Resolved'],
    escalated: ['badge-red',   'Escalated']
  };
  const [cls, label] = map[status] || ['badge-grey', status || 'Unknown'];
  return `<span class="badge ${cls}">${label}</span>`;
}

function generatePalette(n) {
  const base = ['#58a6ff','#bc8cff','#3fb950','#f0a500','#f85149','#79c0ff','#d2a8ff','#56d364'];
  const palette = [];
  for (let i = 0; i < n; i++) palette.push(base[i % base.length]);
  return palette;
}

function showToast(message, type = 'success') {
  const c = document.getElementById('toast-container') || (() => {
    const el = document.createElement('div');
    el.id = 'toast-container';
    el.className = 'toast-container';
    document.body.appendChild(el);
    return el;
  })();
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = message;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}
