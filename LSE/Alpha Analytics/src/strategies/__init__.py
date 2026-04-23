"""Trading strategies built on event-window data."""
from .earnings_strategies import (
    apply_cost,
    backtest_contrarian,
    backtest_contrarian_agnostic,
    backtest_post_earnings_momentum,
    backtest_pre_earnings_runup,
    compounded_return,
    equity_curve,
    prepare_event_frame,
    summarise_strategy,
    window_compounded,
)
from .pre_earnings import PreEarningsStrategy

__all__ = [
    "apply_cost",
    "backtest_contrarian",
    "backtest_contrarian_agnostic",
    "backtest_post_earnings_momentum",
    "backtest_pre_earnings_runup",
    "compounded_return",
    "equity_curve",
    "prepare_event_frame",
    "summarise_strategy",
    "window_compounded",
    "PreEarningsStrategy",
]
