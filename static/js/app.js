/**
 * HealthBot frontend — no framework, no build step.
 *
 * Talks to the Flask API in api/routes.py:
 *   POST /api/chat/start    -> { session_id, reply }
 *   POST /api/chat/message  -> { session_id, reply, meta }
 *   POST /api/chat/reset    -> { session_id, reply }
 *   GET  /api/meta          -> model metadata for the top-bar badge
 *
 * State is intentionally tiny and explicit: one object, one render pass
 * per panel. No virtual DOM — the panels are small enough that direct
 * innerHTML rebuilds stay simple and fast.
 */

(() => {
  "use strict";

  const SESSION_STORAGE_KEY = "healthbot_session_id";

  const state = {
    sessionId: localStorage.getItem(SESSION_STORAGE_KEY) || null,
    sending: false,
  };

  // ── DOM refs ────────────────────────────────────────────────
  const el = {
    chatScroll: document.getElementById("chat-scroll"),
    composer: document.getElementById("composer"),
    input: document.getElementById("composer-input"),
    sendBtn: document.querySelector(".send-btn"),
    examples: document.getElementById("examples"),
    modelBadge: document.getElementById("model-badge"),
    insightCol: document.getElementById("insight-col"),
    insightToggle: document.getElementById("insight-toggle"),
    safetyBanner: document.getElementById("safety-banner"),
    safetyText: document.getElementById("safety-text"),
    kvLanguage: document.getElementById("kv-language"),
    evidenceTags: document.getElementById("evidence-tags"),
    predList: document.getElementById("pred-list"),
    severityText: document.getElementById("severity-text"),
    explainToggle: document.getElementById("explain-toggle"),
    explainBody: document.getElementById("explain-body"),
    explainNote: document.getElementById("explain-note"),
    explainList: document.getElementById("explain-list"),
  };

  // ── Tiny markdown-lite renderer (matches chatbot/bot.py's output) ──
  // Supports **bold**, `code`, and blank-line-separated paragraphs only —
  // that is the full vocabulary chatbot/bot.py actually emits.
  function renderMarkdownLite(text) {
    const escapeHtml = (s) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    return text
      .split(/\n\n+/)
      .map((para) => {
        let html = escapeHtml(para);
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/`(.+?)`/g, "<code>$1</code>");
        html = html.replace(/\n/g, "<br>");
        return `<p>${html}</p>`;
      })
      .join("");
  }

  // ── Chat rendering ──────────────────────────────────────────
  function appendMessage(role, text, { urgency } = {}) {
    const wrap = document.createElement("div");
    wrap.className = `msg msg--${role}`;
    if (role === "bot" && urgency === "emergency") wrap.classList.add("msg--emergency");
    if (role === "bot" && urgency === "urgent") wrap.classList.add("msg--urgent");

    const roleLabel = document.createElement("div");
    roleLabel.className = "msg__role";
    roleLabel.innerHTML = `<span>${role === "user" ? "you" : "healthbot"}</span><span>${timeNow()}</span>`;

    const body = document.createElement("div");
    body.className = "msg__body";
    body.innerHTML = renderMarkdownLite(text);

    wrap.appendChild(roleLabel);
    wrap.appendChild(body);
    el.chatScroll.appendChild(wrap);
    el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
    return wrap;
  }

  function appendTyping() {
    const wrap = document.createElement("div");
    wrap.className = "msg msg--bot msg--typing";
    wrap.id = "typing-indicator";
    wrap.innerHTML =
      '<div class="msg__role"><span>healthbot</span></div>' +
      '<div class="msg__body"><span class="typing-dots"><span></span><span></span><span></span></span></div>';
    el.chatScroll.appendChild(wrap);
    el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
    return wrap;
  }

  function removeTyping() {
    const node = document.getElementById("typing-indicator");
    if (node) node.remove();
  }

  function timeNow() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // ── Insight panel rendering ─────────────────────────────────
  function renderInsight(meta) {
    if (!meta) return;

    // Safety banner
    const redFlag = meta.red_flag || {};
    const urgency = redFlag.urgency_level || "routine";
    el.safetyBanner.className = `safety-banner safety-banner--${urgency}`;
    el.safetyText.textContent =
      redFlag.safety_message ||
      (urgency === "routine" ? "No emergency warning sign detected." : "");

    // Language + evidence understood
    const lang = meta.language || {};
    el.kvLanguage.textContent = lang.language ? `${lang.language} (${Math.round((lang.confidence || 0) * 100)}%)` : "—";

    el.evidenceTags.innerHTML = "";
    el.evidenceTags.className = "tag-list";
    const understood = meta.evidence_understood && meta.evidence_understood.length
      ? meta.evidence_understood
      : (meta.normalized_symptoms || []);
    understood.slice(0, 10).forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      el.evidenceTags.appendChild(li);
    });
    if (!understood.length) {
      const li = document.createElement("li");
      li.textContent = "none yet";
      el.evidenceTags.appendChild(li);
    }

    // Predictions
    const preds = meta.predictions || [];
    el.predList.innerHTML = "";
    if (!preds.length) {
      const li = document.createElement("li");
      li.className = "pred-list__empty";
      li.textContent = "Predictions appear once symptoms are described.";
      el.predList.appendChild(li);
    } else {
      preds.forEach((p) => {
        const li = document.createElement("li");
        li.className = "pred-item";
        const pct = Math.max(2, Math.round((p.confidence || 0) * 100));
        li.innerHTML = `
          <div class="pred-item__row">
            <span class="pred-item__name">${p.disease || "Unknown"}</span>
            <span class="pred-item__pct">${p.confidence_pct || "n/a"}</span>
          </div>
          <div class="pred-item__bar-track"><div class="pred-item__bar-fill" style="width:${pct}%"></div></div>
        `;
        el.predList.appendChild(li);
      });
    }

    // Severity triage
    const severity = meta.severity_triage || {};
    const scope = meta.scope_warning || {};
    el.severityText.textContent =
      scope.message || severity.message || "No severity data for the current candidates.";

    // Explanation (collapsed by default — populate but keep hidden state as-is)
    const explanation = meta.explanation || { contributions: [], note: "" };
    el.explainList.innerHTML = "";
    if (explanation.contributions && explanation.contributions.length) {
      el.explainNote.textContent = explanation.pathology
        ? `Top contributing evidence for ${explanation.pathology}:`
        : "";
      explanation.contributions.forEach((c) => {
        const li = document.createElement("li");
        li.innerHTML = `<span class="ev-meaning">${c.meaning}</span><span class="ev-weight">+${c.weight}</span>`;
        el.explainList.appendChild(li);
      });
    } else {
      el.explainNote.textContent = explanation.note || "No explanation available for this message yet.";
    }
  }

  // ── API calls ───────────────────────────────────────────────
  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Request failed" }));
      throw new Error(err.error || `Request failed (${res.status})`);
    }
    return res.json();
  }

  async function startSession() {
    try {
      const data = await apiPost("/api/chat/start", {});
      state.sessionId = data.session_id;
      localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);
      appendMessage("bot", data.reply);
    } catch (err) {
      appendMessage("bot", "Could not reach the HealthBot server. Is `python3 server.py` running?");
    }
  }

  async function loadMeta() {
    try {
      const res = await fetch("/api/meta");
      const data = await res.json();
      el.modelBadge.innerHTML =
        '<span class="dot dot--ok"></span>' +
        `<span class="topbar__meta-text">${data.selected_model || "model"} · ${data.pathology_count || "?"} conditions</span>`;
    } catch (err) {
      el.modelBadge.innerHTML =
        '<span class="dot dot--down"></span><span class="topbar__meta-text">offline</span>';
    }
  }

  async function sendMessage(text) {
    if (state.sending || !text.trim()) return;
    state.sending = true;
    el.sendBtn.disabled = true;

    appendMessage("user", text);
    el.input.value = "";
    autosize();
    const typingNode = appendTyping();

    try {
      const data = await apiPost("/api/chat/message", {
        session_id: state.sessionId,
        message: text,
      });
      state.sessionId = data.session_id;
      localStorage.setItem(SESSION_STORAGE_KEY, state.sessionId);

      removeTyping();
      const urgency = data.meta && data.meta.red_flag ? data.meta.red_flag.urgency_level : null;
      appendMessage("bot", data.reply, { urgency });
      if (data.meta) renderInsight(data.meta);
    } catch (err) {
      removeTyping();
      appendMessage("bot", "Something went wrong reaching the server. Please try again.");
      console.error(err);
    } finally {
      state.sending = false;
      el.sendBtn.disabled = false;
      typingNode.remove?.();
    }
  }

  // ── Composer behavior ───────────────────────────────────────
  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 140) + "px";
  }

  el.input.addEventListener("input", autosize);
  el.input.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      el.composer.requestSubmit();
    }
  });

  el.composer.addEventListener("submit", (evt) => {
    evt.preventDefault();
    sendMessage(el.input.value);
  });

  el.examples.addEventListener("click", (evt) => {
    const btn = evt.target.closest(".chip");
    if (!btn) return;
    sendMessage(btn.dataset.example);
  });

  el.insightToggle.addEventListener("click", () => {
    const open = el.insightCol.classList.toggle("is-open");
    el.insightToggle.setAttribute("aria-expanded", String(open));
  });

  el.explainToggle.addEventListener("click", () => {
    const hidden = el.explainBody.hasAttribute("hidden");
    if (hidden) {
      el.explainBody.removeAttribute("hidden");
      el.explainToggle.textContent = "hide evidence weights";
    } else {
      el.explainBody.setAttribute("hidden", "");
      el.explainToggle.textContent = "show evidence weights";
    }
    el.explainToggle.setAttribute("aria-expanded", String(hidden));
  });

  // ── Boot ────────────────────────────────────────────────────
  loadMeta();
  startSession();
  el.input.focus();
})();
