"""
Global loading utilities for Agent RICH.

Usage
-----
1.  Call `inject_global_loader()` once per page render (already wired into
    `apply_professional_theme()` in sidebar.py — no manual call needed).

    The top-bar loader is pure-CSS and activates automatically whenever
    *any* ``st.spinner()`` is active — no manual wiring required.

2.  For step-based operations use `loading_bar`:

        with loading_bar("Fetching market data", total_steps=3) as pb:
            data = fetch_data()
            pb.advance("Running analysis…")
            result = analyse(data)
            pb.advance("Rendering charts…")

Implementation note
-------------------
``st.markdown(unsafe_allow_html=True)`` uses React's dangerouslySetInnerHTML
which **never executes <script> tags**.  All JavaScript that needs to run must
be delivered via ``st.components.v1.html()`` (real iframe → scripts execute).
The full-screen dashboard overlay therefore uses a hybrid approach:
  - CSS + HTML  →  st.markdown()   (renders fine)
  - Dismiss JS  →  components.html(height=0) using window.parent.document
"""

import time
import streamlit as st
import streamlit.components.v1 as components
from contextlib import contextmanager


# ─── Global Top-Bar Loader (pure CSS — zero JS required) ─────────────────────
#
# Uses the CSS :has() selector to activate whenever Streamlit's own spinner
# element ([data-testid="stSpinner"]) is present in the DOM.  Because the
# <style> block is injected into the global Streamlit document (not an iframe),
# the selector targets real Streamlit DOM elements directly.

_LOADER_HTML = """
<style>
/* ── Top progress bar ───────────────────────────────────────────────────── */
#ar-topbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 0%;
    height: 3px;
    background: linear-gradient(
        90deg,
        #10b981 0%,
        #3b82f6 30%,
        #8b5cf6 60%,
        #10b981 100%
    );
    background-size: 200% 100%;
    z-index: 999999;
    opacity: 0;
    border-radius: 0 2px 2px 0;
    pointer-events: none;
    /* fade-out transition when spinner disappears */
    transition: opacity 0.45s ease 0.15s, width 0.2s ease;
}

/* Activate when ANY Streamlit spinner is present */
body:has([data-testid="stSpinner"]) #ar-topbar {
    opacity: 1;
    width: 85%;
    animation: ar-topbar-slide 1.8s linear infinite;
    /* fast fade-in */
    transition: opacity 0.1s ease;
}

@keyframes ar-topbar-slide {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Floating "Processing" badge ────────────────────────────────────────── */
#ar-badge {
    position: fixed;
    top: 10px;
    right: 18px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.45);
    border-radius: 20px;
    padding: 5px 14px 5px 10px;
    font-size: 11.5px;
    font-weight: 600;
    color: #10b981;
    z-index: 999998;
    display: none;
    align-items: center;
    gap: 7px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.18);
    letter-spacing: 0.02em;
    font-family: 'Inter', -apple-system, sans-serif;
    pointer-events: none;
}

body:has([data-testid="stSpinner"]) #ar-badge {
    display: flex !important;
    animation: ar-badge-fadein 0.2s ease;
}

@keyframes ar-badge-fadein {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.ar-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
    animation: ar-dot-pulse 1s ease-in-out infinite;
}

@keyframes ar-dot-pulse {
    0%, 100% { transform: scale(1);   opacity: 1; }
    50%       { transform: scale(1.6); opacity: 0.55; }
}
</style>

<div id="ar-topbar"></div>
<div id="ar-badge"><span class="ar-dot"></span>Processing&hellip;</div>
"""


def inject_global_loader() -> None:
    """
    Inject the global top-progress-bar and processing badge into the page.

    This only needs to be called ONCE per render cycle.  ``apply_professional_theme()``
    in sidebar.py already calls it, so you do NOT need to call it manually.

    The loader activates automatically via CSS ``:has([data-testid="stSpinner"])``
    whenever any ``st.spinner()`` is running — no JavaScript or manual wiring needed.
    """
    st.markdown(_LOADER_HTML, unsafe_allow_html=True)


# ─── Full-Screen Dashboard Transition Loader ──────────────────────────────────
#
# CSS + HTML → st.markdown()  (renders the overlay visually)
# Dismiss JS → components.html(height=0)  (runs in real iframe, controls parent)

_DASHBOARD_LOADER_HTML = """
<style>
#ar-dl-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    z-index: 2147483647;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    opacity: 1;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    transition: opacity 0.7s ease;
}
#ar-dl-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(16,185,129,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 50%, rgba(59,130,246,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.ar-dl-logo-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    position: relative;
}
.ar-dl-emoji {
    font-size: 3.8rem;
    animation: ar-dl-float 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 24px rgba(16,185,129,0.45));
}
.ar-dl-app-name {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    background: linear-gradient(90deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ar-dl-tagline {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.ar-dl-spinner-wrap {
    position: relative;
    width: 64px;
    height: 64px;
}
.ar-dl-ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
}
.ar-dl-ring-1 {
    border: 3px solid transparent;
    border-top-color: #10b981;
    border-right-color: #10b981;
    animation: ar-dl-spin 0.85s linear infinite;
}
.ar-dl-ring-2 {
    inset: 9px;
    border: 2px solid transparent;
    border-top-color: #3b82f6;
    animation: ar-dl-spin 1.35s linear infinite reverse;
}
.ar-dl-ring-3 {
    inset: 18px;
    border: 2px solid transparent;
    border-top-color: #8b5cf6;
    animation: ar-dl-spin 1.9s linear infinite;
}
.ar-dl-center-dot {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ar-dl-center-dot::after {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: ar-dl-pulse-dot 1s ease-in-out infinite;
}
.ar-dl-progress-track {
    width: 220px;
    height: 2px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    overflow: hidden;
}
.ar-dl-progress-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
    animation: ar-dl-progress 4s ease-in-out forwards;
}
.ar-dl-status {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.03em;
    min-height: 1.3em;
    transition: opacity 0.3s ease;
}
@keyframes ar-dl-float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
@keyframes ar-dl-spin {
    to { transform: rotate(360deg); }
}
@keyframes ar-dl-pulse-dot {
    0%, 100% { transform: scale(1);   opacity: 0.8; }
    50%       { transform: scale(1.6); opacity: 1;   }
}
@keyframes ar-dl-progress {
    0%   { width: 0%;  }
    20%  { width: 20%; }
    50%  { width: 55%; }
    80%  { width: 82%; }
    100% { width: 95%; }
}
</style>

<div id="ar-dl-overlay">
    <div class="ar-dl-logo-wrap">
        <span class="ar-dl-emoji">&#x1F916;</span>
        <div class="ar-dl-app-name">Agent RICH</div>
        <div class="ar-dl-tagline">Real-time Investment Capital Hub</div>
    </div>
    <div class="ar-dl-spinner-wrap">
        <div class="ar-dl-ring ar-dl-ring-1"></div>
        <div class="ar-dl-ring ar-dl-ring-2"></div>
        <div class="ar-dl-ring ar-dl-ring-3"></div>
        <div class="ar-dl-center-dot"></div>
    </div>
    <div class="ar-dl-progress-track">
        <div class="ar-dl-progress-fill"></div>
    </div>
    <div class="ar-dl-status" id="ar-dl-status-msg">Initializing dashboard&hellip;</div>
</div>
"""

# JavaScript delivered via components.html() so it actually executes.
# Uses window.parent.document to reach the Streamlit frame where the overlay lives.
_DASHBOARD_LOADER_JS = """
<script>
(function () {
    var messages = [
        'Initializing dashboard\u2026',
        'Loading AI agents\u2026',
        'Fetching market data\u2026',
        'Configuring analytics\u2026',
        'Preparing workspace\u2026',
        'Almost ready\u2026'
    ];

    function init() {
        var pdoc     = window.parent.document;
        var overlay  = pdoc.getElementById('ar-dl-overlay');
        var statusEl = pdoc.getElementById('ar-dl-status-msg');

        /* Retry until overlay is present in the parent frame */
        if (!overlay) {
            setTimeout(init, 120);
            return;
        }

        /* Prevent double-init if iframe somehow re-runs */
        if (overlay.dataset.arInit) return;
        overlay.dataset.arInit = '1';

        /* ── Rotating status messages ── */
        var msgIdx   = 0;
        var msgTimer = setInterval(function () {
            msgIdx = Math.min(msgIdx + 1, messages.length - 1);
            if (statusEl) {
                statusEl.style.opacity = '0';
                setTimeout(function () {
                    if (statusEl) {
                        statusEl.textContent = messages[msgIdx];
                        statusEl.style.opacity = '1';
                    }
                }, 150);
            }
            if (msgIdx === messages.length - 1) clearInterval(msgTimer);
        }, 650);

        /* ── Dismiss logic ── */
        function dismiss() {
            clearInterval(msgTimer);
            var ov = pdoc.getElementById('ar-dl-overlay');
            if (!ov || ov.dataset.arDismissed) return;
            ov.dataset.arDismissed = '1';
            var se = pdoc.getElementById('ar-dl-status-msg');
            if (se) se.textContent = 'Ready!';
            ov.style.opacity       = '0';
            ov.style.pointerEvents = 'none';
            setTimeout(function () {
                var el = pdoc.getElementById('ar-dl-overlay');
                if (el && el.parentNode) el.parentNode.removeChild(el);
            }, 800);
        }

        /* Hard safety-net: always dismiss after 7 s */
        var maxTimer = setTimeout(dismiss, 7000);

        /* Early dismiss once Streamlit has rendered real content */
        var MO = (pdoc.defaultView || window.parent).MutationObserver;
        var observer = new MO(function () {
            var mainBlock =
                pdoc.querySelector('[data-testid="stMainBlockContainer"]') ||
                pdoc.querySelector('[data-testid="stAppViewBlockContainer"]') ||
                pdoc.querySelector('.main .block-container');
            if (mainBlock && mainBlock.children.length >= 2) {
                setTimeout(function () {
                    clearTimeout(maxTimer);
                    dismiss();
                    observer.disconnect();
                }, 400);
            }
        });
        observer.observe(pdoc.body, { childList: true, subtree: true });
    }

    /* Start after a short delay to let the parent frame settle */
    setTimeout(init, 50);
})();
</script>
"""


def inject_dashboard_loader() -> None:
    """
    Inject a full-screen transition overlay that covers the viewport while the
    dashboard modules are loading after login.

    The overlay HTML/CSS is injected via ``st.markdown()`` (renders fine) and
    the dismiss logic is injected via ``components.html(height=0)`` (scripts
    actually execute in the iframe and reach the overlay via window.parent).

    Call this at the very top of the dashboard render path, guarded by
    ``st.session_state.pop('_dashboard_loading', False)`` so it only shows
    on the first post-login render cycle.
    """
    st.markdown(_DASHBOARD_LOADER_HTML, unsafe_allow_html=True)
    components.html(_DASHBOARD_LOADER_JS, height=0)


# ─── Step-Based Progress Bar Context Manager ─────────────────────────────────

class _ProgressTracker:
    """Internal tracker passed to the `loading_bar` context manager."""

    def __init__(self, bar, status, label: str, total_steps: int):
        self._bar = bar
        self._status = status
        self._label = label
        self._total = max(total_steps, 1)
        self._step = 0

    def advance(self, message: str = "") -> None:
        """Advance one step and update the displayed message."""
        self._step += 1
        pct = min(int(self._step / self._total * 100), 99)
        msg = message or self._label
        self._bar.progress(pct, text=msg)
        self._status.update(label=msg)


@contextmanager
def loading_bar(label: str = "Processing…", total_steps: int = 1):
    """
    Context manager that shows a step-based progress bar + collapsible status
    block for any block of code.

    Parameters
    ----------
    label : str
        Initial label shown in both the status widget and the progress bar.
    total_steps : int
        Total number of steps.  Call ``tracker.advance(msg)`` once per step.

    Example
    -------
    ::

        with loading_bar("Loading market data", total_steps=3) as pb:
            prices = fetch_prices(symbol)
            pb.advance("Calculating indicators…")
            indicators = calc_indicators(prices)
            pb.advance("Generating chart…")
            fig = build_chart(indicators)
            pb.advance("Done")
    """
    status = st.status(label, expanded=False)
    bar    = st.progress(0, text=label)
    tracker = _ProgressTracker(bar, status, label, total_steps)

    try:
        yield tracker
        bar.progress(100, text="Complete")
        status.update(label=f"{label} — complete", state="complete", expanded=False)
        time.sleep(0.35)
        bar.empty()
        status.empty()
    except Exception as exc:
        bar.empty()
        status.update(label=f"Error: {exc}", state="error", expanded=True)
        raise
