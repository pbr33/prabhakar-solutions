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
# The overlay is created ENTIRELY via components.html() JS using
# window.parent.document.body.appendChild().  This keeps the element OUTSIDE
# React's virtual DOM tree so React's reconciliation can never conflict with
# our manual DOM removal — eliminating the NotFoundError: removeChild crash.

_DASHBOARD_LOADER_JS = """
<script>
(function () {
    var pdoc = window.parent.document;

    /* ── CSS injected into parent <head> ───────────────────────────────── */
    var CSS = [
        '#ar-dl-overlay{position:fixed;top:0;left:0;right:0;bottom:0;',
        'background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);',
        'z-index:2147483647;display:flex;flex-direction:column;',
        'align-items:center;justify-content:center;gap:1.5rem;',
        'opacity:1;transition:opacity .7s ease;',
        "font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;}",

        '#ar-dl-overlay::before{content:"";position:absolute;inset:0;',
        'background:radial-gradient(ellipse at 20% 50%,rgba(16,185,129,.06) 0%,transparent 50%),',
        'radial-gradient(ellipse at 80% 50%,rgba(59,130,246,.06) 0%,transparent 50%);',
        'pointer-events:none;}',

        /* Avatar image wrapper (holds glow + float animation) */
        '.ar-dl-avatar-wrap{position:relative;width:92px;height:92px;border-radius:50%;',
        'box-shadow:0 0 0 3px rgba(16,185,129,.25),0 0 48px rgba(16,185,129,.45),0 0 90px rgba(59,130,246,.2);',
        'animation:ar-dl-float 2.5s ease-in-out infinite;flex-shrink:0;}',

        '.ar-dl-avatar-wrap::after{content:"";position:absolute;inset:-4px;border-radius:50%;',
        'background:conic-gradient(from 0deg,#10b981,#3b82f6,#8b5cf6,#10b981);',
        'z-index:-1;opacity:.5;filter:blur(6px);}',

        /* Avatar image */
        '.ar-dl-avatar{width:92px;height:92px;border-radius:50%;',
        'display:block;object-fit:cover;position:relative;z-index:1;}',

        '.ar-dl-logo-wrap{display:flex;flex-direction:column;align-items:center;gap:.7rem;}',

        '.ar-dl-app-name{font-size:1.9rem;font-weight:800;letter-spacing:.07em;',
        'background:linear-gradient(90deg,#10b981 0%,#3b82f6 50%,#8b5cf6 100%);',
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}',

        '.ar-dl-tagline{font-size:.75rem;color:rgba(255,255,255,.28);',
        'letter-spacing:.2em;text-transform:uppercase;}',

        '.ar-dl-spinner-wrap{position:relative;width:64px;height:64px;}',
        '.ar-dl-ring{position:absolute;inset:0;border-radius:50%;}',
        '.ar-dl-ring-1{border:3px solid transparent;border-top-color:#10b981;',
        'border-right-color:#10b981;animation:ar-dl-spin .85s linear infinite;}',
        '.ar-dl-ring-2{inset:9px;border:2px solid transparent;border-top-color:#3b82f6;',
        'animation:ar-dl-spin 1.35s linear infinite reverse;}',
        '.ar-dl-ring-3{inset:18px;border:2px solid transparent;border-top-color:#8b5cf6;',
        'animation:ar-dl-spin 1.9s linear infinite;}',
        '.ar-dl-center-dot{position:absolute;inset:0;display:flex;',
        'align-items:center;justify-content:center;}',
        '.ar-dl-center-dot::after{content:"";width:8px;height:8px;border-radius:50%;',
        'background:#10b981;box-shadow:0 0 10px #10b981;',
        'animation:ar-dl-pulse-dot 1s ease-in-out infinite;}',

        '.ar-dl-progress-track{width:220px;height:2px;',
        'background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden;}',
        '.ar-dl-progress-fill{height:100%;border-radius:2px;',
        'background:linear-gradient(90deg,#10b981,#3b82f6,#8b5cf6);',
        'animation:ar-dl-progress 4s ease-in-out forwards;}',

        '.ar-dl-status{font-size:.85rem;color:rgba(255,255,255,.42);',
        'letter-spacing:.03em;min-height:1.3em;transition:opacity .3s ease;}',

        '@keyframes ar-dl-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}',
        '@keyframes ar-dl-spin{to{transform:rotate(360deg)}}',
        '@keyframes ar-dl-pulse-dot{0%,100%{transform:scale(1);opacity:.8}50%{transform:scale(1.6);opacity:1}}',
        '@keyframes ar-dl-progress{0%{width:0}20%{width:20%}50%{width:55%}80%{width:82%}100%{width:95%}}'
    ].join('');

    /* ── Only run once per page load ───────────────────────────────────── */
    if (pdoc.getElementById('ar-dl-overlay')) return;

    var styleEl = pdoc.createElement('style');
    styleEl.id  = 'ar-dl-css';
    styleEl.textContent = CSS;
    pdoc.head.appendChild(styleEl);

    /* ── Build overlay HTML and append DIRECTLY to <body> ──────────────
       By appending to body (not inside any React component) we stay
       completely outside React's virtual DOM — React can never trigger
       a removeChild conflict on this element.                           */
    var overlay = pdoc.createElement('div');
    overlay.id  = 'ar-dl-overlay';
    overlay.innerHTML = [
        '<div class="ar-dl-logo-wrap">',
        '  <div class="ar-dl-avatar-wrap"><img class="ar-dl-avatar" src="data:image/png;base64,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"/></div>',
        '  <div class="ar-dl-app-name">Agent RICH</div>',
        '  <div class="ar-dl-tagline">Real-time Investment Capital Hub</div>',
        '</div>',
        '<div class="ar-dl-spinner-wrap">',
        '  <div class="ar-dl-ring ar-dl-ring-1"></div>',
        '  <div class="ar-dl-ring ar-dl-ring-2"></div>',
        '  <div class="ar-dl-ring ar-dl-ring-3"></div>',
        '  <div class="ar-dl-center-dot"></div>',
        '</div>',
        '<div class="ar-dl-progress-track"><div class="ar-dl-progress-fill"></div></div>',
        '<div class="ar-dl-status" id="ar-dl-status-msg">Initializing dashboard\u2026</div>'
    ].join('');
    pdoc.body.appendChild(overlay);

    /* ── Status message rotation ────────────────────────────────────── */
    var messages = [
        'Initializing dashboard\u2026',
        'Loading AI agents\u2026',
        'Fetching market data\u2026',
        'Configuring analytics\u2026',
        'Preparing workspace\u2026',
        'Almost ready\u2026'
    ];
    var msgIdx   = 0;
    var statusEl = pdoc.getElementById('ar-dl-status-msg');
    var msgTimer = setInterval(function () {
        msgIdx = Math.min(msgIdx + 1, messages.length - 1);
        if (statusEl) {
            statusEl.style.opacity = '0';
            setTimeout(function () {
                if (statusEl) { statusEl.textContent = messages[msgIdx]; statusEl.style.opacity = '1'; }
            }, 150);
        }
        if (msgIdx === messages.length - 1) clearInterval(msgTimer);
    }, 650);

    /* ── Dismiss ────────────────────────────────────────────────────── */
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
            /* el.remove() is safe even if the element was already gone */
            var el = pdoc.getElementById('ar-dl-overlay');
            if (el) { try { el.remove(); } catch (e) {} }
            var css = pdoc.getElementById('ar-dl-css');
            if (css) { try { css.remove(); } catch (e) {} }
        }, 800);
    }

    /* Hard safety-net: always dismiss after 7 s */
    var maxTimer = setTimeout(dismiss, 7000);

    /* Early dismiss once Streamlit renders real content */
    var MO = (pdoc.defaultView || window.parent).MutationObserver;
    var observer = new MO(function () {
        var mainBlock =
            pdoc.querySelector('[data-testid="stMainBlockContainer"]')    ||
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
})();
</script>
"""


def inject_dashboard_loader() -> None:
    """
    Inject a full-screen transition overlay after login.

    The overlay is created entirely via ``components.html(height=0)`` JS
    using ``window.parent.document.body.appendChild()``.  Because the element
    lives directly in ``<body>`` — outside React's virtual DOM — React's
    reconciliation can never trigger a ``removeChild`` conflict.

    Guarded by ``st.session_state.pop('_dashboard_loading', False)`` so it
    only shows once on the first post-login render cycle.
    """
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
