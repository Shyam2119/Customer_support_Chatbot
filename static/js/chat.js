/**
 * chat.js - Customer Support Chatbot UI
 * Handles session management, messaging, typing indicators, and feedback
 */

const API_BASE = '/api/chat';

const state = {
  sessionId: null,
  rating: 0,
  feedbackSubmitted: false,
  messageCount: 0
};

// ─── DOM Refs ─────────────────────────────────────────
const messagesEl     = document.getElementById('messages');
const inputEl        = document.getElementById('message-input');
const sendBtn        = document.getElementById('send-btn');
const sessionSpan    = document.getElementById('session-id-display');
const msgCountSpan   = document.getElementById('msg-count');
const scrollBtn      = document.getElementById('scroll-bottom-btn');
const feedbackPanel  = document.getElementById('feedback-panel');
const feedbackThanks = document.getElementById('feedback-thanks');
const starBtns       = document.querySelectorAll('.star-btn');

// ─── Init ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSession();
  setupInput();
  setupFeedback();
  setupScrollWatcher();
});

async function initSession() {
  try {
    const res = await fetch(`${API_BASE}/session`, { method: 'POST' });
    const data = await res.json();
    state.sessionId = data.session_id;
    if (sessionSpan) sessionSpan.textContent = state.sessionId.slice(0, 8) + '…';
    appendSystemMessage('New session started');
    appendBotMessage(
      "👋 Hello! I'm your virtual support assistant. I can help with orders, returns, payments, account issues, and more. How can I assist you today?",
      { intent: 'greeting', confidence: 99 }
    );
  } catch (e) {
    appendSystemMessage('⚠️ Could not connect to server. Please refresh.');
    console.error('Session init error:', e);
  }
}

// ─── Input Handling ───────────────────────────────────
function setupInput() {
  inputEl?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputEl?.addEventListener('input', () => {
    // Auto-resize textarea
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });

  sendBtn?.addEventListener('click', sendMessage);
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !state.sessionId) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  appendUserMessage(text);
  state.messageCount++;
  updateMsgCount();

  // Hide quick replies after first message
  document.getElementById('quick-replies')?.remove();

  const typingId = showTyping();

  try {
    const res = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, message: text })
    });

    removeTyping(typingId);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    appendBotMessage(data.response, {
      intent:     data.intent,
      confidence: data.confidence,
      sentiment:  data.sentiment,
      entities:   data.entities,
      responseTime: data.response_time_ms
    });

    state.messageCount++;
    updateMsgCount();

    // Show feedback panel after 3+ bot messages
    if (state.messageCount >= 4 && !state.feedbackSubmitted && feedbackPanel) {
      feedbackPanel.style.display = 'flex';
    }

  } catch (err) {
    removeTyping(typingId);
    appendBotMessage('⚠️ I\'m having trouble connecting. Please try again in a moment.', { intent: 'error' });
    console.error('Send message error:', err);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ─── Quick Replies ────────────────────────────────────
function handleQuickReply(text) {
  if (!inputEl) return;
  inputEl.value = text;
  sendMessage();
}

// ─── Message Rendering ────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="avatar-sm user">😊</div>
    <div class="message-group">
      <div class="bubble user">${escapeHtml(text)}</div>
      <div class="message-meta">${formatTime(new Date())}</div>
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function appendBotMessage(text, meta = {}) {
  const { intent, confidence, sentiment, entities, responseTime } = meta;
  const sentClass = sentiment?.sentiment || '';
  const sentEmoji = { positive: '😊', negative: '😔', neutral: '😐' }[sentClass] || '';

  const intentHtml = intent && intent !== 'error'
    ? `<span class="intent-badge">🏷️ ${intent.replace(/_/g, ' ')}</span>`
    : '';

  const sentHtml = sentiment && sentClass
    ? `<span class="sentiment-tag ${sentClass}">${sentEmoji} ${sentClass}</span>`
    : '';

  const confVal = typeof confidence === 'number' ? confidence : 0;
  const confHtml = confVal > 0
    ? `<div class="confidence-bar"><div class="confidence-fill" style="width:${confVal}%"></div></div>`
    : '';

  const entityHtml = buildEntityPills(entities);
  const rtHtml = responseTime ? `<span style="color:var(--text-muted);font-size:0.65rem">${responseTime}ms</span>` : '';

  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.innerHTML = `
    <div class="avatar-sm bot">🤖</div>
    <div class="message-group">
      <div class="bubble bot">${escapeHtml(text)}${entityHtml}</div>
      <div class="message-meta">
        ${intentHtml}${sentHtml}${rtHtml}
        <span>${formatTime(new Date())}</span>
      </div>
      ${confHtml}
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function appendSystemMessage(text) {
  const el = document.createElement('div');
  el.className = 'system-message';
  el.textContent = text;
  messagesEl.appendChild(el);
}

function buildEntityPills(entities) {
  if (!entities || Object.keys(entities).length === 0) return '';
  const pills = Object.entries(entities)
    .map(([k, v]) => `<span class="entity-pill">📌 ${k}: ${escapeHtml(String(v))}</span>`)
    .join('');
  return `<div class="entity-pills">${pills}</div>`;
}

// ─── Typing Indicator ─────────────────────────────────
let typingCounter = 0;

function showTyping() {
  const id = `typing-${++typingCounter}`;
  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.id = id;
  row.innerHTML = `
    <div class="avatar-sm bot">🤖</div>
    <div class="typing-bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  document.getElementById(id)?.remove();
}

// ─── Feedback ─────────────────────────────────────────
function setupFeedback() {
  starBtns.forEach(btn => {
    btn.addEventListener('mouseenter', () => highlightStars(+btn.dataset.value));
    btn.addEventListener('mouseleave', () => highlightStars(state.rating));
    btn.addEventListener('click', () => {
      state.rating = +btn.dataset.value;
      highlightStars(state.rating);
    });
  });

  document.getElementById('submit-feedback-btn')?.addEventListener('click', submitFeedback);
}

function highlightStars(val) {
  starBtns.forEach(b => b.classList.toggle('active', +b.dataset.value <= val));
}

async function submitFeedback() {
  if (!state.sessionId) return;
  const payload = {
    session_id: state.sessionId,
    rating: state.rating || null,
    helpful: state.rating >= 4,
    feedback_text: document.getElementById('feedback-text')?.value || ''
  };

  try {
    await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (feedbackThanks) feedbackThanks.style.display = 'inline';
    document.getElementById('feedback-controls')?.remove();
    state.feedbackSubmitted = true;
    showToast('Thank you for your feedback! 🙏', 'success');
  } catch (e) {
    showToast('Could not submit feedback.', 'error');
  }
}

// ─── Scroll Watcher ───────────────────────────────────
function setupScrollWatcher() {
  messagesEl?.addEventListener('scroll', () => {
    if (!scrollBtn) return;
    const dist = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
    scrollBtn.classList.toggle('visible', dist > 120);
  });

  scrollBtn?.addEventListener('click', () => scrollToBottom(true));
}

function scrollToBottom(smooth = false) {
  if (!messagesEl) return;
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
}

// ─── Toast Notifications ──────────────────────────────
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container') || (() => {
    const c = document.createElement('div');
    c.id = 'toast-container';
    c.className = 'toast-container';
    document.body.appendChild(c);
    return c;
  })();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ─── Helpers ──────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function updateMsgCount() {
  if (msgCountSpan) msgCountSpan.textContent = Math.floor(state.messageCount / 2);
}

// Expose for inline onclick handlers
window.handleQuickReply = handleQuickReply;
window.sendMessage      = sendMessage;
