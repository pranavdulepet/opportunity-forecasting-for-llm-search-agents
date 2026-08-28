"""Shared visual style for manuscript plots.

The manuscript uses compact two-column figures, so plot assets are generated
close to their final printed size. This keeps tick labels and legends readable
after LaTeX scales the PDFs.
"""

from __future__ import annotations

from typing import Any


BASE_PROMPT = "#315F8C"
REGRESSION_HEAD = "#C76E00"
ORACLE = "#2F3437"
NEUTRAL = "#6B7280"
GRID = "#D8DEE6"
TEXT = "#111827"
BEST_FIXED_FILL = "#F2A51A"
HEATMAP_COLORS = ("#F7F7F2", "#C8D9D2", "#6FA49B", "#2D6468", "#143B46")


def apply_paper_style(plt: Any, *, base_font_size: float = 9.5) -> None:
    """Apply ACL-sized, print-friendly matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 0.8,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size - 1.2,
            "ytick.labelsize": base_font_size - 1.2,
            "legend.fontsize": base_font_size - 1.4,
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
