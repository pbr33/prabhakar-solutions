"""
Global loading utilities for Agent RICH.

Usage
-----
1.  Call `inject_global_loader()` once per page render (already wired into
    `apply_professional_theme()` in sidebar.py — no manual call needed).

2.  For step-based operations use `loading_bar`:

        with loading_bar("Fetching market data", total_steps=3) as pb:
            data = fetch_data()
            pb.advance("Running analysis…")
            result = analyse(data)
            pb.advance("Rendering charts…")
"""

import time
import streamlit as st
from contextlib import contextmanager


# ─── Global Top-Bar Loader ────────────────────────────────────────────────────

_LOADER_HTML = """
<style>
/* ── Top progress bar ─────────────────────────────────────────────── */
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
    transition: opacity 0.25s ease;
}
#ar-topbar.ar-loading {
    opacity: 1;
    width: 85%;
    animation: ar-slide 1.8s linear infinite, ar-shimmer 1.8s linear infinite;
    transition: width 0.4s ease, opacity 0.25s ease;
}
#ar-topbar.ar-done {
    width: 100% !important;
    opacity: 0;
    animation: none;
    transition: width 0.2s ease, opacity 0.45s ease 0.15s;
}
@keyframes ar-slide {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@keyframes ar-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Floating "Processing" badge ──────────────────────────────────── */
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
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.18);
    letter-spacing: 0.02em;
    font-family: 'Inter', -apple-system, sans-serif;
    pointer-events: none;
}
#ar-badge.ar-visible {
    display: flex !important;
    animation: ar-fadein 0.2s ease;
}
@keyframes ar-fadein {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ar-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
    animation: ar-pulse 1s ease-in-out infinite;
}
@keyframes ar-pulse {
    0%, 100% { transform: scale(1);   opacity: 1; }
    50%       { transform: scale(1.6); opacity: 0.55; }
}
</style>

<div id="ar-topbar"></div>
<div id="ar-badge"><span class="ar-dot"></span>Processing&hellip;</div>

<script>
(function () {
    var bar   = document.getElementById('ar-topbar');
    var badge = document.getElementById('ar-badge');
    if (!bar || !badge) return;

    var doneTimer = null;
    var active    = false;

    function show() {
        if (active) return;
        active = true;
        clearTimeout(doneTimer);
        bar.classList.remove('ar-done');
        bar.classList.add('ar-loading');
        badge.classList.add('ar-visible');
    }

    function hide() {
        if (!active) return;
        active = false;
        bar.classList.remove('ar-loading');
        bar.classList.add('ar-done');
        badge.classList.remove('ar-visible');
        doneTimer = setTimeout(function () {
            bar.classList.remove('ar-done');
        }, 700);
    }

    /* Watch the DOM for Streamlit spinners */
    var observer = new MutationObserver(function () {
        var spinner =
            document.querySelector('[data-testid="stSpinner"]') ||
            document.querySelector('.stSpinner')               ||
            document.querySelector('[data-testid="stStatusWidget"] [data-testid="stSpinner"]');
        if (spinner) { show(); } else { hide(); }
    });

    observer.observe(document.body, {
        childList:      true,
        subtree:        true,
        attributes:     true,
        attributeFilter: ['class', 'data-testid', 'aria-label']
    });

    /* Also catch the initial Streamlit "connecting" state */
    window.addEventListener('load', function () {
        setTimeout(function () {
            var spinner =
                document.querySelector('[data-testid="stSpinner"]') ||
                document.querySelector('.stSpinner');
            if (!spinner) { hide(); }
        }, 500);
    });
})();
</script>
"""


def inject_global_loader() -> None:
    """
    Inject the global top-progress-bar and processing badge into the page.

    This only needs to be called ONCE per render cycle.  `apply_professional_theme()`
    in sidebar.py already calls it, so you do NOT need to call it manually.

    The loader auto-shows/hides whenever any ``st.spinner()`` is active — no
    changes to existing spinner calls are required.
    """
    st.markdown(_LOADER_HTML, unsafe_allow_html=True)


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
