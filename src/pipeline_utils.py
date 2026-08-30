"""
pipeline_utils.py

Small shared helpers used by every phase script, factored out here so
behavior stays consistent as the project grows past Phase 1 (rather
than copy-pasting the same log()/save_and_display() into every new
script and having them drift apart).
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # see the same fix (and its full reasoning) applied at the top
    # of every pipeline script that imports matplotlib -- duplicated here as a
    # defensive backstop in case this module is ever imported before a script's own
    # matplotlib import in some other context, not as the primary fix location.
import matplotlib.pyplot as plt


def running_interactively() -> bool:
    """True inside VS Code's Interactive Window / Jupyter, False when run
    as a plain script -- used to decide whether it's safe to call
    plt.show() without blocking the whole script (see save_and_display)."""
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


SHOW_PLOTS = running_interactively()


def log(msg: str) -> None:
    """Timestamped progress print, so a running script is never silent."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def save_and_display(fig, path: Path) -> None:
    """Always saves the figure; only opens a pop-up window when running
    inside an interactive kernel. In plain 'python script.py' mode this
    never blocks waiting for a window to be closed."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log(f"Saved figure -> {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


class Timer:
    """Context manager for a script-total elapsed-time readout, used in
    the final summary banner of each phase script."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start
