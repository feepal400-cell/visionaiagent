"""Gradio UI for the Vision Agent AI — Modern dark redesign."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from gradio.themes import Soft

from agent import AgentError, DEFAULT_PROMPT, analyze_image

HISTORY_FILE = Path(__file__).parent / "analysis_history.json"

APP_THEME = Soft(
    primary_hue="teal",
    neutral_hue="slate",
).set(
    body_background_fill="#0a0e14",
    body_background_fill_dark="#0a0e14",
    body_text_color="#eaf2f2",
    body_text_color_dark="#eaf2f2",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    block_background_fill="#121826",
    block_background_fill_dark="#121826",
    block_label_background_fill="#16202f",
    block_label_background_fill_dark="#16202f",
    block_label_text_color="#5eead4",
    block_label_text_color_dark="#5eead4",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    block_border_color="rgba(94, 234, 212, 0.14)",
    block_border_color_dark="rgba(94, 234, 212, 0.14)",
    input_background_fill="#0d131d",
    input_background_fill_dark="#0d131d",
    input_border_color="rgba(94, 234, 212, 0.18)",
    input_border_color_dark="rgba(94, 234, 212, 0.18)",
    slider_color="#2dd4bf",
    slider_color_dark="#2dd4bf",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    color-scheme: dark !important;
    --bg: #0a0e14;
    --bg-2: #0e131b;
    --surface: #121826;
    --surface-2: #161d2c;
    --line: rgba(94, 234, 212, 0.14);
    --line-strong: rgba(94, 234, 212, 0.32);
    --ink: #eaf2f2;
    --muted: #94a3b8;
    --accent: #2dd4bf;
    --accent-2: #38bdf8;
    --accent-3: #a78bfa;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
}

* { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
code, pre, .json-holder, .cm-editor * {
    font-family: 'JetBrains Mono', monospace !important;
}

html, body, .gradio-container {
    color-scheme: dark !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 10% 0%, rgba(45, 212, 191, 0.10), transparent 32%),
        radial-gradient(circle at 90% 15%, rgba(56, 189, 248, 0.08), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(167, 139, 250, 0.06), transparent 40%),
        var(--bg) !important;
    min-height: 100vh;
    padding: 24px 20px !important;
}
.main { background: transparent !important; }

/* ── Header ─────────────────────────────────────────────────────────── */
#header-block {
    background: linear-gradient(135deg, #101826 0%, #0d1420 55%, #111b2c 100%) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 32px 40px !important;
    margin-bottom: 28px !important;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.4),
        0 0 80px rgba(45, 212, 191, 0.06),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
#header-block::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 25% 40%, rgba(45, 212, 191, 0.10) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(56, 189, 248, 0.08) 0%, transparent 50%);
    pointer-events: none;
}
#header-block h1 {
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    letter-spacing: 4px !important;
    background: linear-gradient(135deg, #ecfeff 0%, #99f6e4 30%, #2dd4bf 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0 !important;
    position: relative;
    z-index: 1;
}
#header-block p {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    letter-spacing: 1.5px !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}

/* ── Badges ─────────────────────────────────────────────────────────── */
.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    border: 1px solid transparent;
}
.badge-tag:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
}
.badge-agent {
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.18), rgba(45, 212, 191, 0.06));
    color: #5eead4;
    border-color: rgba(45, 212, 191, 0.28);
}
.badge-yolo {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(56, 189, 248, 0.06));
    color: #7dd3fc;
    border-color: rgba(56, 189, 248, 0.28);
}
.badge-groq {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.18), rgba(167, 139, 250, 0.06));
    color: #c4b5fd;
    border-color: rgba(167, 139, 250, 0.28);
}

/* ── Panels & Layout Spacing ─────────────────────────────────────────── */
.panel-card {
    background: linear-gradient(160deg, var(--surface-2) 0%, var(--surface) 100%) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    display: flex !important;
    flex-direction: column !important;
    gap: 18px !important; /* Generous breathing room */
}
.panel-card > div {
    background: transparent !important;
}
.panel-card:hover {
    border-color: var(--line-strong) !important;
    box-shadow: 0 8px 40px rgba(45, 212, 191, 0.08) !important;
}

.section-label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #cbd5e1 !important;
    margin-bottom: 12px !important;
}

/* ── Button ─────────────────────────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 55%, #2563eb 100%) !important;
    background-size: 200% 200% !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.8px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    margin-top: 6px !important;
    cursor: pointer;
    transition: box-shadow 0.25s ease, background-position 0.25s ease !important;
    box-shadow: 0 5px 22px rgba(13, 148, 136, 0.4) !important;
    position: relative;
    overflow: hidden;
    min-height: 54px !important;
}
#analyze-btn::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    transition: left 0.6s ease;
}
#analyze-btn:hover {
    background-position: 100% 0 !important;
    filter: brightness(1.12) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(13, 148, 136, 0.55) !important;
}
#analyze-btn:hover::before { left: 100%; }
#analyze-btn:active {
    filter: brightness(0.96) !important;
    transform: translateY(0);
    box-shadow: 0 2px 12px rgba(13, 148, 136, 0.35) !important;
}
#analyze-btn:focus-visible {
    outline: 3px solid rgba(94, 234, 212, 0.8) !important;
    outline-offset: 3px;
}
#analyze-btn:disabled {
    opacity: 0.55 !important;
    filter: grayscale(0.25) !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────── */
.tabs {
    background: transparent !important;
    border: none !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}
.tab-nav, [role="tablist"] {
    background: rgba(14, 20, 30, 0.85) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 6px !important;
    margin-bottom: 18px !important;
}
.tab-nav button,
.tab-nav button *,
[role="tablist"] button,
[role="tablist"] button *,
button[role="tab"],
button[role="tab"] * {
    background: transparent !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    transition: all 0.25s ease !important;
}
.tab-nav button:hover,
.tab-nav button:hover *,
[role="tablist"] button:hover,
[role="tablist"] button:hover *,
button[role="tab"]:hover,
button[role="tab"]:hover * {
    background: rgba(45, 212, 191, 0.12) !important;
    color: #5eead4 !important;
}
.tab-nav button.selected,
.tab-nav button.selected *,
[role="tablist"] button.selected,
[role="tablist"] button.selected *,
button[role="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] *,
button[role="tab"].selected,
button[role="tab"].selected * {
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.22), rgba(56, 189, 248, 0.18)) !important;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 12px rgba(45, 212, 191, 0.2) !important;
}
.tabitem {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 0 0 !important;
}

.tab-nav .overflow-menu > button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 36px !important;
    min-width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    background: #17202f !important;
    color: #eaf2f2 !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3) !important;
}
.tab-nav .overflow-menu > button svg { color: #eaf2f2 !important; fill: currentColor !important; }
.tab-nav .overflow-menu > button:hover {
    background: #1c2637 !important;
    color: #5eead4 !important;
    border-color: #5eead4 !important;
    transform: translateY(-1px);
}
.tab-nav .overflow-dropdown {
    background: #101826 !important;
    border-color: var(--line-strong) !important;
}
.tab-nav .overflow-dropdown button, .tab-nav .overflow-dropdown button span {
    color: #eaf2f2 !important;
    background: transparent !important;
}
.tab-nav .overflow-dropdown button:hover {
    background: rgba(45, 212, 191, 0.14) !important;
    color: #99f6e4 !important;
}

/* ── Markdown output ───────────────────────────────────────────────── */
#ai-reasoning {
    background: linear-gradient(160deg, var(--surface-2) 0%, var(--surface) 100%) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 28px !important;
    min-height: 340px !important;
    color: var(--ink) !important;
    line-height: 1.8 !important;
    font-size: 0.95rem !important;
}
#ai-reasoning > div { background: transparent !important; }
#ai-reasoning .prose { background: transparent !important; }
#ai-reasoning h1, #ai-reasoning h2, #ai-reasoning h3 { color: #7dd3fc !important; font-weight: 700 !important; }
#ai-reasoning strong { color: #5eead4 !important; }
#ai-reasoning em { color: var(--muted) !important; }
#ai-reasoning p, #ai-reasoning li, #ai-reasoning blockquote { color: #e2e8f0 !important; }

/* ── Images ─────────────────────────────────────────────────────────── */
.image-container, .image-frame, .upload-container { border-radius: 14px !important; overflow: hidden !important; }
.output-image { min-height: 440px !important; height: 440px !important; }
.output-image .image-container, .output-image .image-frame { min-height: 440px !important; height: 440px !important; }
.image-frame img { border-radius: 12px !important; object-fit: contain !important; }

#input-image {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
}
#input-image label, #input-image .label-wrap {
    background: var(--surface-2) !important;
    color: #99f6e4 !important;
}
#input-image .image-container, #input-image .image-frame,
#input-image .upload-container, #input-image .wrap, #input-image .empty {
    background: linear-gradient(145deg, #16202f, #101725) !important;
    color: #e2e8f0 !important;
}
#input-image .upload-text, #input-image .upload-text *, #input-image .or,
#input-image button, #input-image svg { color: #99f6e4 !important; }
#input-image button:hover { background: rgba(45, 212, 191, 0.14) !important; }

/* ── Form Controls & General Block Theme Fixes ───────────────────────── */
.block {
    background: #121826 !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
}

/* Consistent block labels / badges */
.block-label,
[data-testid="block-label"],
.label-wrap,
span.label {
    background: rgba(22, 32, 47, 0.9) !important;
    color: #5eead4 !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
}

/* ── Confidence Slider (Dark Theme & Spacing) ────────────────────────── */
#confidence-slider,
.gradio-slider {
    background: #121826 !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin: 4px 0 !important;
}

#confidence-slider .label-wrap,
.gradio-slider .label-wrap {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 12px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}

#confidence-slider .label-wrap span,
.gradio-slider .label-wrap span,
#confidence-slider label span,
.gradio-slider label span {
    background: transparent !important;
    color: #5eead4 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    padding: 0 !important;
}

#confidence-slider input[type="number"],
.gradio-slider input[type="number"] {
    background: #0d131d !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 6px 12px !important;
    min-width: 68px !important;
    text-align: center !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
}

#confidence-slider input[type="range"],
.gradio-slider input[type="range"] {
    accent-color: #2dd4bf !important;
    margin-top: 8px !important;
    cursor: pointer !important;
}

#confidence-slider span,
.gradio-slider span,
#confidence-slider .min-max,
.gradio-slider .min-max {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

#confidence-slider button,
.gradio-slider button {
    color: #94a3b8 !important;
    background: transparent !important;
}
#confidence-slider button:hover,
.gradio-slider button:hover {
    color: #5eead4 !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: #0d131d !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--ink) !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 14px 16px !important;
    line-height: 1.45 !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(45, 212, 191, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.12) !important;
    outline: none !important;
}

#prompt-input { width: 100% !important; }
#prompt-input,
#prompt-input > div {
    background: var(--surface) !important;
}
#prompt-input label span {
    color: #99f6e4 !important;
}
#prompt-input textarea {
    min-height: 96px !important;
    height: auto !important;
    max-height: 180px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}

/* ── Chatbot ────────────────────────────────────────────────────────── */
#agent-chat {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
}
#agent-chat .message, #agent-chat .message * {
    color: #eaf2f2 !important;
}
#agent-chat .bot {
    background: #16202f !important;
    color: #eaf2f2 !important;
    border: 1px solid var(--line) !important;
}
#agent-chat .user {
    background: rgba(45, 212, 191, 0.18) !important;
    color: #ffffff !important;
    border: 1px solid rgba(45, 212, 191, 0.3) !important;
}

/* ── Status bar ────────────────────────────────────────────────────── */
#status-bar {
    background: linear-gradient(135deg, rgba(10, 14, 20, 0.85) 0%, rgba(16, 24, 38, 0.7) 100%) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    margin-top: 24px !important;
}
#status-bar p, #status-bar span { font-size: 0.85rem !important; color: #94a3b8 !important; letter-spacing: 0.5px !important; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; animation: pulse 2s ease-in-out infinite; }
.status-dot.idle { background: #64748b; animation: none; }
.status-dot.running { background: var(--accent); }
.status-dot.done { background: var(--success); animation: none; }
.status-dot.error { background: var(--danger); animation: none; }
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(45, 212, 191, 0); }
}

/* ── JSON viewer ───────────────────────────────────────────────────── */
.json-holder {
    background: rgba(10, 14, 20, 0.8) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    min-height: 340px !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1219; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(45, 212, 191, 0.35); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(45, 212, 191, 0.6); }

.built-with, footer, .gradio-container > footer { display: none !important; }
[data-testid="footer"], [class*="built-with"], [class*="footer"] { display: none !important; }

/* ── Toasts / Errors ───────────────────────────────────────────────── */
.toast-wrap, div[class*="toast"] {
    display: none !important;
}

/* ── Responsive ────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    #header-block h1 { font-size: 1.6rem !important; letter-spacing: 2px !important; }
    .badge-row { gap: 6px; }
    .badge-tag { font-size: 0.65rem; padding: 4px 10px; }
}
"""

HEADER_HTML = """
<div>
    <h1>VISION AGENT AI</h1>
    <p>Self-Analyzing AI Vision System</p>
    <div class="badge-row">
        <span class="badge-tag badge-agent">🤖 Autonomous Agent</span>
        <span class="badge-tag badge-yolo">📦 YOLOv8 Perception</span>
        <span class="badge-tag badge-groq">⚡ Groq Powered</span>
    </div>
</div>
"""

IDLE_STATUS = (
    '<span class="status-dot idle"></span> '
    '<strong style="color:#94a3b8;">IDLE</strong> '
    '<span style="margin-left:12px;color:#cbd5e1;">— Upload an image and click Analyze</span>'
)

PLACEHOLDER_MD = """
<div style="text-align:center; padding:60px 20px; opacity:0.75;">
    <p style="font-size:2.5rem; margin-bottom:8px;">🧠</p>
    <p style="font-size:0.95rem; color:#94a3b8; font-weight:500;">
        AI scene analysis will appear here after you run the agent.
    </p>
</div>
"""

DARK_MODE_JS = """
function() {
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    const container = document.querySelector('.gradio-container');
    if (container) {
        container.classList.add('dark');
    }
}
"""


def _status_html(state: str, model: str = "", elapsed: float = 0.0) -> str:
    """Build the status bar HTML."""
    if state == "running":
        return (
            '<span class="status-dot running"></span> '
            '<strong style="color:#5eead4;">ANALYZING</strong> '
            '<span style="margin-left:12px;color:#2dd4bf;">— Agent is processing…</span>'
        )
    if state == "done":
        model_tag = (
            f' <span style="margin-left:16px;color:#64748b;">|</span> '
            f'<span style="margin-left:8px;color:#34d399;">Model: {model}</span>'
            if model else ""
        )
        return (
            f'<span class="status-dot done"></span> '
            f'<strong style="color:#34d399;">COMPLETE</strong> '
            f'<span style="margin-left:12px;color:#10b981;">— Finished in {elapsed}s</span>'
            f'{model_tag}'
        )
    if state == "error":
        return (
            '<span class="status-dot error"></span> '
            '<strong style="color:#fca5a5;">ERROR</strong> '
            '<span style="margin-left:12px;color:#ef4444;">— See message above</span>'
        )
    return IDLE_STATUS


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_history(history: list[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_history_html() -> str:
    history = _load_history()
    if not history:
        return (
            '<div style="text-align:center; padding:60px 20px; opacity:0.6;">'
            '<p style="font-size:2rem; margin-bottom:8px;">📂</p>'
            '<p style="color:#94a3b8; font-size:0.9rem;">No analysis history yet. '
            'Run your first analysis to see it here.</p></div>'
        )

    rows = []
    for i, entry in enumerate(reversed(history)):
        idx = len(history) - 1 - i
        name = entry.get("filename", "Unknown")
        ts = entry.get("timestamp", "")
        detected = entry.get("objects_detected", "None")
        model = entry.get("model_used", "")
        elapsed = entry.get("elapsed", 0)
        was_detected = entry.get("was_detected", False)

        status_dot = "#34d399" if was_detected else "#f87171"
        status_label = "Detected" if was_detected else "No Detection"

        rows.append(
            f'<div style="display:flex; align-items:center; gap:14px; padding:14px 18px; '
            f'background:#0e131b; border:1px solid rgba(94,234,212,0.12); border-radius:12px; '
            f'margin-bottom:10px; transition:border-color 0.2s;" '
            f'onmouseover="this.style.borderColor=\'rgba(94,234,212,0.35)\'" '
            f'onmouseout="this.style.borderColor=\'rgba(94,234,212,0.12)\'">' 
            f'  <div style="flex-shrink:0; width:10px; height:10px; border-radius:50%; background:{status_dot};"></div>'
            f'  <div style="flex:1; min-width:0;">'
            f'    <p style="margin:0; font-size:0.88rem; font-weight:600; color:#e2e8f0; '
            f'       overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name}</p>'
            f'    <p style="margin:2px 0 0; font-size:0.75rem; color:#64748b;">'
            f'      {ts} &nbsp;·&nbsp; {status_label} &nbsp;·&nbsp; {detected} &nbsp;·&nbsp; {elapsed}s</p>'
            f'  </div>'
            f'  <button onclick="document.getElementById(\'view-idx\').value=\'{idx}\'; '
            f'    document.getElementById(\'view-idx\').dispatchEvent(new Event(\'input\'));" '
            f'    style="padding:6px 14px; font-size:0.75rem; font-weight:600; border-radius:8px; '
            f'    border:1px solid rgba(94,234,212,0.3); background:rgba(94,234,212,0.1); '
            f'    color:#5eead4; cursor:pointer; transition:all 0.2s;" '
            f'    onmouseover="this.style.background=\'rgba(94,234,212,0.2)\'" '
            f'    onmouseout="this.style.background=\'rgba(94,234,212,0.1)\'">👁 View</button>'
            f'  <button onclick="document.getElementById(\'del-idx\').value=\'{idx}\'; '
            f'    document.getElementById(\'del-idx\').dispatchEvent(new Event(\'input\'));" '
            f'    style="padding:6px 14px; font-size:0.75rem; font-weight:600; border-radius:8px; '
            f'    border:1px solid rgba(248,113,113,0.3); background:rgba(248,113,113,0.1); '
            f'    color:#fca5a5; cursor:pointer; transition:all 0.2s;" '
            f'    onmouseover="this.style.background=\'rgba(248,113,113,0.2)\'" '
            f'    onmouseout="this.style.background=\'rgba(248,113,113,0.1)\'">🗑 Delete</button>'
            f'</div>'
        )

    return (
        f'<div style="max-height:420px; overflow-y:auto; padding:4px;">'
        f'{chr(10).join(rows)}'
        f'</div>'
        f'<p style="text-align:right; margin:10px 4px 0; font-size:0.75rem; color:#64748b;">'
        f'{len(history)} analysis record{"s" if len(history) != 1 else ""}</p>'
    )


def _delete_history_entry(idx_str: str) -> str:
    try:
        idx = int(idx_str)
    except (ValueError, TypeError):
        return _render_history_html()
    history = _load_history()
    if 0 <= idx < len(history):
        history.pop(idx)
        _save_history(history)
    return _render_history_html()


def _view_history_entry(idx_str: str) -> tuple[str, str, str | None]:
    try:
        idx = int(idx_str)
    except (ValueError, TypeError):
        return "", "", None
    history = _load_history()
    if 0 <= idx < len(history):
        entry = history[idx]
        reasoning = (
            f"### 🔍 Scene Analysis (from history)\n\n"
            f"> **Verified detection:** {entry.get('objects_detected', 'None')}\n\n"
            f"{entry.get('description', 'No description saved.')}"
        )
        model = entry.get("model_used", "")
        elapsed = entry.get("elapsed", 0)
        status = _status_html("done", model=model, elapsed=elapsed)
        annotated = entry.get("annotated_path")
        return reasoning, status, annotated
    return "", "", None


def run_analysis(
    image_path: str | None,
    prompt: str,
    conf_threshold: float,
) -> tuple[str, str, str]:
    """Run the agent and return (reasoning_md, status_html, annotated_image_path, history_html)."""
    if not image_path:
        raise gr.Error("⚠️ Please upload an image before running the agent.")

    prompt = prompt.strip() if prompt else DEFAULT_PROMPT

    try:
        result = analyze_image(image_path, prompt, conf_threshold)
    except (AgentError, ValueError, FileNotFoundError, OSError) as error:
        # Save failed attempt to history
        history = _load_history()
        history.append({
            "filename": Path(image_path).name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "objects_detected": "Error",
            "was_detected": False,
            "model_used": "",
            "elapsed": 0,
            "description": str(error),
            "annotated_path": None,
        })
        _save_history(history)
        raise gr.Error(f"❌ {error}") from error

    text = result["text"]
    raw = result.get("raw_metadata", {})
    counts = raw.get("counts", {}) if isinstance(raw, dict) else {}
    verified_counts = ", ".join(
        f"{count} {name}" for name, count in counts.items()
    ) or "No objects detected"
    reasoning_md = (
        "### 🔍 Scene Analysis\n\n"
        f"> **Verified detection:** {verified_counts}\n\n"
        f"{text}"
    )

    model_used = result.get("model_used", "unknown")
    elapsed = result.get("elapsed_seconds", 0.0)
    status = _status_html("done", model=model_used, elapsed=elapsed)
    annotated_image_path = result.get("annotated_image_path")
    if not isinstance(annotated_image_path, str) or not annotated_image_path:
        raise gr.Error("❌ Detection did not produce an annotated image.")

    # Save to history
    history = _load_history()
    history.append({
        "filename": Path(image_path).name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "objects_detected": verified_counts,
        "was_detected": bool(counts),
        "model_used": model_used,
        "elapsed": elapsed,
        "description": text,
        "annotated_path": annotated_image_path,
    })
    _save_history(history)

    return reasoning_md, status, annotated_image_path, _render_history_html()


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""
    with gr.Blocks(
        title="Vision Agent AI — Autonomous Scene Perception",
    ) as app:

        gr.HTML(HEADER_HTML, elem_id="header-block")

        with gr.Row(equal_height=False):

            with gr.Column(scale=4, min_width=360):
                gr.HTML('<p class="section-label">📤 INPUT</p>')
                with gr.Column(elem_classes=["panel-card"]):
                    image_input = gr.Image(
                        label="Input Image",
                        type="filepath",
                        height=320,
                        sources=["upload", "clipboard"],
                        elem_id="input-image",
                    )
                    gr.HTML(
                        '<p style="margin:4px 0 0 2px; font-size:0.78rem; color:#64748b; letter-spacing:0.3px;">'
                        '📎 Supported formats: JPG, JPEG, PNG</p>'
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Detection Confidence Threshold",
                        elem_id="confidence-slider",
                    )
                    prompt_input = gr.Textbox(
                        label="Analysis Prompt (optional)",
                        placeholder="Describe the scene, spatial layout, and notable objects…",
                        value=DEFAULT_PROMPT,
                        lines=4,
                        max_lines=6,
                        elem_id="prompt-input",
                    )
                    analyze_btn = gr.Button(
                        "Analyze Scene",
                        variant="primary",
                        elem_id="analyze-btn",
                    )

            with gr.Column(scale=6, min_width=440):
                gr.HTML('<p class="section-label">📊 AGENT WORKSPACE</p>')
                with gr.Column(elem_classes=["panel-card"]):
                    with gr.Tabs():
                        with gr.Tab("Annotated Image & Spatial Grid"):
                            annotated_output = gr.Image(
                                label="Detections",
                                type="filepath",
                                height=440,
                                elem_id="annotated-image",
                                show_label=False,
                            )
                        with gr.Tab("Initial Scene Description"):
                            reasoning_output = gr.Markdown(
                                value=PLACEHOLDER_MD,
                                elem_id="ai-reasoning",
                            )
                        with gr.Tab("📂 Analysis History"):
                            history_display = gr.HTML(
                                value=_render_history_html(),
                                elem_id="history-panel",
                            )
                            # Hidden textboxes to relay button clicks from HTML
                            del_idx = gr.Textbox(visible=False, elem_id="del-idx")
                            view_idx = gr.Textbox(visible=False, elem_id="view-idx")

        status_bar = gr.HTML(
            value=IDLE_STATUS,
            elem_id="status-bar",
        )

        analyze_btn.click(
            fn=lambda: _status_html("running"),
            inputs=None,
            outputs=status_bar,
            queue=False,
        ).then(
            fn=run_analysis,
            inputs=[image_input, prompt_input, confidence_slider],
            outputs=[reasoning_output, status_bar, annotated_output, history_display],
        )

        del_idx.input(
            fn=_delete_history_entry,
            inputs=del_idx,
            outputs=history_display,
        )

        view_idx.input(
            fn=_view_history_entry,
            inputs=view_idx,
            outputs=[reasoning_output, status_bar, annotated_output],
        )

    return app


demo = build_app()

import os
import socket
import streamlit as st

def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    # Prevent duplicate server initialization in Streamlit reruns
    if "gradio_launched" not in st.session_state:
        port = get_open_port()
        demo.launch(
            theme=APP_THEME,
            css=CUSTOM_CSS,
            js=DARK_MODE_JS,
            server_name="0.0.0.0",
            server_port=port,
            prevent_thread_lock=True,
            inline=False,
            quiet=True
        )
        st.session_state["gradio_launched"] = True
        st.session_state["gradio_port"] = port

    # Streamlit page view
    st.set_page_config(page_title="Vision AI Agent", layout="wide")
    st.components.v1.html(
        f'<iframe src="http://localhost:{st.session_state["gradio_port"]}" width="100%" height="800" frameborder="0"></iframe>',
        height=800
    )