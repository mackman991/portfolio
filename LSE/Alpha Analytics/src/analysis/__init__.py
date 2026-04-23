"""Analysis modules: technical indicators, event studies, cross-section."""
from .cross_section import (
    analyze_correlations,
    calculate_event_returns,
    calculate_qoq_growth,
    merge_returns_and_growth,
)
from .event_study import build_event_windows, plot_event_panel, prepare_inputs
from .technical import TechnicalAnalysis

__all__ = [
    "TechnicalAnalysis",
    "build_event_windows",
    "plot_event_panel",
    "prepare_inputs",
    "calculate_event_returns",
    "calculate_qoq_growth",
    "merge_returns_and_growth",
    "analyze_correlations",
]
