"""Earnings Analytics Dashboard — SP100 Results."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = "data"

st.set_page_config(page_title="Earnings Analytics — SP100", layout="wide")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_strategy_summary():
    df = pd.read_csv(os.path.join(DATA_DIR, "strategy_summary_sp100.csv"))
    df = df.rename(columns={"Unnamed: 0": "Strategy"})
    for col in ["mean", "median", "hit_rate", "std", "se"]:
        df[col] = df[col].str.rstrip("%").astype(float)
    df["t_stat"] = pd.to_numeric(df["t_stat"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").astype(int)
    return df


@st.cache_data
def load_beat_miss():
    return pd.read_csv(os.path.join(DATA_DIR, "beat_miss_sp100.csv"))


@st.cache_data
def load_beat_miss_returns():
    df = pd.read_csv(os.path.join(DATA_DIR, "beat_miss_returns_sp100.csv"),
                     parse_dates=["eps_date"])
    df["beat_label"] = df["beat"].map({True: "Beat", False: "Miss"})
    df["surprise_pct"] = df["surprise"] * 100
    df["ret_1d_pct"] = df["ret_1d"] * 100
    df["ret_5d_pct"] = df["ret_5d"] * 100
    return df


@st.cache_data
def load_equity_curves():
    mom = pd.read_csv(os.path.join(DATA_DIR, "equity_curve_momentum_h6_sp100.csv"),
                      parse_dates=["eps_date"])
    pre = pd.read_csv(os.path.join(DATA_DIR, "equity_curve_pre_runup_p10_sp100.csv"),
                      parse_dates=["eps_date"])
    mom["strategy"] = "Momentum +1..+6"
    pre["strategy"] = "Pre Run-up -10..-1"
    return pd.concat([mom, pre], ignore_index=True)


@st.cache_data
def load_event_windows():
    return pd.read_csv(
        os.path.join(DATA_DIR, "eps_event_windows_15_sp100.csv"),
        parse_dates=["eps_date"],
        usecols=["ticker", "eps_date", "beat", "rel_day", "ret", "surprise"],
    )


@st.cache_data
def load_indicators():
    # indicators_sp100_v2.csv is the full 104-ticker pull (~130k rows,
    # 2021-04-19..2026-04-16). The older indicators_sp100.csv only covered
    # 5 tickers from a partial pull — fall back to it only if v2 is missing.
    path_v2 = os.path.join(DATA_DIR, "indicators_sp100_v2.csv")
    path_v1 = os.path.join(DATA_DIR, "indicators_sp100.csv")
    path = path_v2 if os.path.exists(path_v2) else path_v1
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_filtered_summary():
    """Hold/Cut regime-filter backtest summary (+ buy-and-hold benchmark)."""
    return pd.read_csv(os.path.join(DATA_DIR, "filtered_hold_cut_summary.csv"))


@st.cache_data
def load_filtered_equity_curves():
    """Hold/Cut variants + buy-and-hold benchmark, returned as long-form."""
    variants = {
        "Unfiltered Hold/Cut":    "equity_curve_hold_cut_unfiltered.csv",
        "+ MACD bullish":         "equity_curve_hold_cut_macd_bullish.csv",
        "+ Above SMA-50":         "equity_curve_hold_cut_above_sma50.csv",
        "+ Not RSI-oversold":     "equity_curve_hold_cut_not_rsi_oversold.csv",
        "Buy-and-Hold benchmark": "equity_curve_buyhold_benchmark.csv",
    }
    frames = []
    for label, fname in variants.items():
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["eps_date"])
        df["variant"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_regime_grid(strategy: str):
    path = os.path.join(DATA_DIR, f"regime_grid_{strategy}_sp100.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Earnings Analytics")
st.sidebar.caption("SP100 · 2020–2025 · 103 tickers · 2,234 events")
page = st.sidebar.radio(
    "View",
    [
        "Live Monitor",
        "Strategy Summary",
        "Equity Curves",
        "Event Study",
        "Beat/Miss Breakdown",
        "Technical Signals",
        "Regime Filters",
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Strategy Summary
# ══════════════════════════════════════════════════════════════════════════════

if page == "Strategy Summary":
    st.title("Strategy Backtest Summary")
    st.caption("All returns after 5 bps round-trip cost · equal-weight per event")

    df = load_strategy_summary()

    col1, col2, col3, col4 = st.columns(4)
    best = df.loc[df["t_stat"].idxmax()]
    col1.metric("Best Mean Return", f"{best['mean']:.2f}%", help=best["Strategy"])
    col2.metric("Best t-stat", f"{best['t_stat']:.2f}", help=best["Strategy"])
    col3.metric("Total Events", f"{df['n'].iloc[0]:,}")
    col4.metric("Best Hit Rate", f"{df['hit_rate'].max():.1f}%")

    st.divider()

    # Bar chart: mean return by strategy, coloured by t-stat significance
    fig_bar = go.Figure()
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in df["mean"]]
    fig_bar.add_trace(go.Bar(
        x=df["Strategy"], y=df["mean"],
        marker_color=colors,
        text=[f"t={t:.2f}" for t in df["t_stat"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}%<br><extra></extra>",
    ))
    fig_bar.update_layout(
        title="Mean Per-Event Return by Strategy (%, after cost)",
        xaxis_tickangle=-30,
        yaxis_title="Mean return (%)",
        height=420,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#333"),
    )
    st.plotly_chart(fig_bar, width="stretch")

    # Scatter: mean vs t-stat
    fig_scatter = px.scatter(
        df, x="t_stat", y="mean", text="Strategy",
        color="hit_rate", color_continuous_scale="RdYlGn",
        size=[abs(t) * 2 for t in df["t_stat"]],
        title="Mean Return vs t-stat (bubble size = |t-stat|, colour = hit rate %)",
        labels={"t_stat": "t-statistic", "mean": "Mean return (%)", "hit_rate": "Hit rate (%)"},
        height=400,
    )
    fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="gray",
                          annotation_text="t=2 threshold")
    fig_scatter.add_hline(y=0, line_color="gray")
    fig_scatter.update_traces(textposition="top center", textfont_size=10)
    st.plotly_chart(fig_scatter, width="stretch")

    st.subheader("Full Results Table")
    display_df = df.copy()
    display_df["mean"] = display_df["mean"].map("{:.2f}%".format)
    display_df["median"] = display_df["median"].map("{:.2f}%".format)
    display_df["hit_rate"] = display_df["hit_rate"].map("{:.1f}%".format)
    display_df["std"] = display_df["std"].map("{:.2f}%".format)
    display_df["t_stat"] = display_df["t_stat"].map("{:.2f}".format)
    st.dataframe(display_df.set_index("Strategy"), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Equity Curves
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Equity Curves":
    st.title("Equity Curves")
    st.caption("Equal-weight across all events on each EPS date · $1 starting capital · no reinvestment across strategies")

    ec = load_equity_curves()

    fig = px.line(
        ec, x="eps_date", y="equity", color="strategy",
        title="Cumulative Equity: Momentum vs Pre-Earnings Run-up (SP100, 2020–2025)",
        labels={"eps_date": "Date", "equity": "Equity ($1 start)", "strategy": "Strategy"},
        height=500,
        color_discrete_map={"Momentum +1..+6": "#3498db", "Pre Run-up -10..-1": "#e67e22"},
    )
    fig.update_layout(plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
    fig.add_hline(y=1, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    for strat, col in zip(["Momentum +1..+6", "Pre Run-up -10..-1"], [c1, c2]):
        sub = ec[ec["strategy"] == strat].copy()
        final = sub["equity"].iloc[-1]
        annual = (final ** (1 / 5) - 1) * 100
        col.metric(strat, f"${final:.1f}x", f"{annual:.1f}% CAGR (≈5yr)")

    # Per-event return distribution
    st.subheader("Per-Event Return Distribution")
    selected_strat = st.selectbox("Strategy", ["Momentum +1..+6", "Pre Run-up -10..-1"])
    ev = load_beat_miss_returns()
    event_windows = load_event_windows()

    if selected_strat == "Momentum +1..+6":
        per_event = (
            event_windows[(event_windows["rel_day"] >= 1) & (event_windows["rel_day"] <= 6)]
            .groupby(["ticker", "eps_date", "beat"])["ret"]
            .apply(lambda r: (1 + r).prod() - 1)
            .reset_index()
        )
        per_event["signed_ret"] = per_event.apply(
            lambda row: row["ret"] if row["beat"] else -row["ret"], axis=1
        )
        ret_col = "signed_ret"
    else:
        per_event = (
            event_windows[(event_windows["rel_day"] >= -10) & (event_windows["rel_day"] <= -1)]
            .groupby(["ticker", "eps_date"])["ret"]
            .apply(lambda r: (1 + r).prod() - 1)
            .reset_index()
        )
        ret_col = "ret"

    per_event["ret_pct"] = per_event[ret_col] * 100
    fig_hist = px.histogram(
        per_event, x="ret_pct", nbins=60,
        title=f"Per-Event Return Distribution: {selected_strat}",
        labels={"ret_pct": "Return (%)"},
        color_discrete_sequence=["#3498db"],
        height=350,
    )
    fig_hist.add_vline(x=0, line_color="red", line_dash="dash")
    fig_hist.add_vline(x=per_event["ret_pct"].mean(), line_color="green",
                       annotation_text=f"Mean {per_event['ret_pct'].mean():.2f}%")
    st.plotly_chart(fig_hist, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Event Study
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Event Study":
    st.title("Event Study: Returns Around Earnings")

    ev = load_event_windows()

    # Ticker selector
    tickers = sorted(ev["ticker"].unique())
    selected = st.multiselect("Tickers (empty = all SP100 pooled)", tickers, default=[])

    subset = ev[ev["ticker"].isin(selected)] if selected else ev

    mean_ret = (
        subset.groupby(["beat", "rel_day"])["ret"]
        .mean()
        .reset_index()
    )
    mean_ret["beat_label"] = mean_ret["beat"].map({True: "Beat", False: "Miss"})
    mean_ret["ret_pct"] = mean_ret["ret"] * 100

    n_beats = subset[subset["beat"] == True][["ticker", "eps_date"]].drop_duplicates().shape[0]
    n_misses = subset[subset["beat"] == False][["ticker", "eps_date"]].drop_duplicates().shape[0]

    fig = px.line(
        mean_ret, x="rel_day", y="ret_pct", color="beat_label",
        color_discrete_map={"Beat": "#27ae60", "Miss": "#e74c3c"},
        title=f"Mean Daily Return by Relative Day (Beats n={n_beats}, Misses n={n_misses})",
        labels={"rel_day": "Relative day (0 = earnings)", "ret_pct": "Mean return (%)", "beat_label": ""},
        markers=True, height=450,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#333", annotation_text="Earnings day")
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, width="stretch")

    # Cumulative return around event
    st.subheader("Cumulative Return Around Event")
    cum_ret = (
        subset.groupby(["beat", "rel_day"])["ret"]
        .mean()
        .groupby(level=0)
        .cumsum()
        .reset_index()
    )
    cum_ret["beat_label"] = cum_ret["beat"].map({True: "Beat", False: "Miss"})
    cum_ret["ret_pct"] = cum_ret["ret"] * 100

    fig2 = px.line(
        cum_ret, x="rel_day", y="ret_pct", color="beat_label",
        color_discrete_map={"Beat": "#27ae60", "Miss": "#e74c3c"},
        title="Cumulative Mean Return (simple sum) by Relative Day",
        labels={"rel_day": "Relative day", "ret_pct": "Cumulative return (%)", "beat_label": ""},
        markers=True, height=400,
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="#333")
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_layout(plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig2, width="stretch")

    # Surprise vs Day +1 return scatter
    st.subheader("EPS Surprise vs. Day +1 Return")
    bm = load_beat_miss_returns()
    if selected:
        bm = bm[bm["ticker"].isin(selected)]
    # clip extreme surprises for readability
    bm_clip = bm[bm["surprise_pct"].between(-100, 200)]
    fig3 = px.scatter(
        bm_clip, x="surprise_pct", y="ret_1d_pct",
        color="beat_label", opacity=0.5,
        color_discrete_map={"Beat": "#27ae60", "Miss": "#e74c3c"},
        trendline="ols",
        title="EPS Surprise (%) vs. Next-Day Return (%)",
        labels={"surprise_pct": "EPS surprise (%)", "ret_1d_pct": "Day +1 return (%)", "beat_label": ""},
        height=420,
    )
    fig3.add_hline(y=0, line_color="gray")
    fig3.add_vline(x=0, line_color="gray")
    st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Beat/Miss Breakdown
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Beat/Miss Breakdown":
    st.title("Beat/Miss Breakdown by Ticker")

    bm = load_beat_miss()
    for col in ["beat_rate"]:
        bm[col] = bm[col].str.rstrip("%").astype(float)
    bm["mean_surprise_pct"] = bm["mean_surprise_pct"].str.rstrip("%").astype(float)
    bm["mean_day0_ret"] = bm["mean_day0_ret"].str.rstrip("%").astype(float)

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("SP100 Overall Beat Rate", f"{bm['beat_rate'].mean():.1f}%")
    c2.metric("Avg Mean Surprise", f"{bm['mean_surprise_pct'].mean():.2f}%")
    c3.metric("Avg Day-0 Return", f"{bm['mean_day0_ret'].mean():.2f}%")

    st.divider()

    # Beat rate bar
    bm_sorted = bm.sort_values("beat_rate", ascending=True)
    fig = px.bar(
        bm_sorted, x="beat_rate", y="ticker",
        orientation="h",
        color="beat_rate",
        color_continuous_scale="RdYlGn",
        title="Beat Rate by Ticker (% of quarters beating EPS estimate)",
        labels={"beat_rate": "Beat rate (%)", "ticker": ""},
        height=900,
    )
    fig.add_vline(x=bm["beat_rate"].mean(), line_dash="dash", line_color="gray",
                  annotation_text=f"Avg {bm['beat_rate'].mean():.1f}%")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

    # Day-0 return vs beat rate scatter
    fig2 = px.scatter(
        bm, x="beat_rate", y="mean_day0_ret",
        text="ticker", size="n_events",
        color="mean_surprise_pct",
        color_continuous_scale="RdYlGn",
        title="Beat Rate vs. Mean Day-0 Return (size = # events, colour = mean surprise %)",
        labels={"beat_rate": "Beat rate (%)", "mean_day0_ret": "Mean day-0 return (%)",
                "mean_surprise_pct": "Mean surprise (%)"},
        height=500,
    )
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_traces(textposition="top center", textfont_size=8)
    st.plotly_chart(fig2, width="stretch")

    # Detailed table
    st.subheader("Full Table")
    st.dataframe(
        bm.sort_values("beat_rate", ascending=False).reset_index(drop=True),
        width="stretch",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Technical Signals
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Technical Signals":
    st.title("Technical Indicators")

    ind = load_indicators()
    tickers = sorted(ind["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)

    sub = ind[ind["ticker"] == ticker].sort_values("date")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=[f"{ticker} — SMAs", "RSI (14)", "MACD"],
        vertical_spacing=0.06,
    )

    # Price + SMAs
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["sma_20"], name="SMA 20",
                             line=dict(color="#3498db", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["sma_50"], name="SMA 50",
                             line=dict(color="#e67e22", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["sma_200"], name="SMA 200",
                             line=dict(color="#8e44ad", width=2)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["rsi_14"], name="RSI 14",
                             line=dict(color="#27ae60")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if "macd" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["macd"], name="MACD",
                                 line=dict(color="#2980b9")), row=3, col=1)
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["macd_signal"], name="Signal",
                                 line=dict(color="#e74c3c", dash="dash")), row=3, col=1)
        if "macd_hist" in sub.columns:
            colors_hist = ["#27ae60" if v >= 0 else "#e74c3c" for v in sub["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=sub["date"], y=sub["macd_hist"], name="Histogram",
                                 marker_color=colors_hist, opacity=0.5), row=3, col=1)

    fig.add_hline(y=0, line_color="gray", row=3, col=1)
    fig.update_layout(height=700, showlegend=True, plot_bgcolor="white",
                      yaxis=dict(gridcolor="#f0f0f0"),
                      yaxis2=dict(gridcolor="#f0f0f0", range=[0, 100]),
                      yaxis3=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, width="stretch")

    # Latest snapshot
    latest = sub.iloc[-1]
    st.subheader(f"Latest Snapshot ({latest['date'].date()})")
    cols = st.columns(5)
    for col, (label, val) in zip(cols, [
        ("RSI 14", f"{latest['rsi_14']:.1f}"),
        ("SMA 20", f"{latest['sma_20']:.2f}"),
        ("SMA 50", f"{latest['sma_50']:.2f}"),
        ("SMA 200", f"{latest['sma_200']:.2f}"),
        ("MACD", f"{latest.get('macd', float('nan')):.3f}"),
    ]):
        col.metric(label, val)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Regime Filters (Hold/Cut × technical indicators)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Regime Filters":
    st.title("Regime Filters on Hold/Cut Strategy")
    st.caption(
        "Hold/Cut baseline (thr=5%, ext=+10) conditioned on technical state at "
        "announcement. Indicator coverage = 77.5% of events "
        "(pre-2021-04-19 events excluded — FMP indicator history starts there)."
    )

    summary = load_filtered_summary()

    # ── Top KPIs: best filtered variant vs unfiltered baseline ──
    baseline = summary[summary["variant"] == "unfiltered_hold_cut"].iloc[0]
    best_idx = summary[summary["variant"].str.startswith("hold_cut_")]["info_ratio"].idxmax()
    best = summary.loc[best_idx]
    bh = summary[summary["variant"] == "buyhold_benchmark"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Unfiltered Mean",
        f"{baseline['mean']*100:+.2f}%",
        delta=f"IR {baseline['info_ratio']:.2f}",
    )
    c2.metric(
        f"Best Filter: {best['variant'].replace('hold_cut_', '+ ')}",
        f"{best['mean']*100:+.2f}%",
        delta=f"{(best['mean']-baseline['mean'])*100:+.2f}pp vs unfiltered",
    )
    c3.metric(
        "Best Hit Rate",
        f"{best['hit_rate']*100:.1f}%",
        delta=f"{(best['hit_rate']-baseline['hit_rate'])*100:+.1f}pp",
    )
    c4.metric(
        "Max DD (best vs unfiltered)",
        f"{best['max_drawdown']*100:+.1f}%",
        delta=f"{(best['max_drawdown']-baseline['max_drawdown'])*100:+.1f}pp",
        delta_color="normal",  # smaller DD is better; color manually below
    )

    st.divider()

    # ── Summary table ──
    st.subheader("Variant Summary")
    disp = summary.copy()
    disp["coverage"] = disp["coverage"].map(lambda v: f"{v*100:.0f}%")
    disp["mean"] = disp["mean"].map(lambda v: f"{v*100:+.2f}%")
    disp["median"] = disp["median"].map(lambda v: f"{v*100:+.2f}%")
    disp["hit_rate"] = disp["hit_rate"].map(lambda v: f"{v*100:.1f}%")
    disp["t_stat"] = disp["t_stat"].map(lambda v: f"{v:+.2f}")
    disp["info_ratio"] = disp["info_ratio"].map(lambda v: f"{v:.2f}")
    disp["max_drawdown"] = disp["max_drawdown"].map(lambda v: f"{v*100:+.1f}%")
    disp["std"] = disp["std"].map(lambda v: f"{v*100:.2f}%")
    # Hide CAGR / total_return / car_mdd — compounding assumes unrealistic
    # sequential reinvestment across overlapping events. Per-event metrics are
    # the honest comparisons.
    cols_keep = [
        "variant", "n", "coverage", "mean", "median", "hit_rate",
        "std", "t_stat", "info_ratio", "max_drawdown",
    ]
    st.dataframe(disp[cols_keep], width="stretch", hide_index=True)
    st.caption(
        "IR = mean / std (per-event, unitless). Max-DD from the compounded "
        "equity curve — useful as a **comparative** risk metric, not a realised "
        "P&L, because overlapping earnings events prevent 100% sequential "
        "reinvestment in practice."
    )

    st.divider()

    # ── Equity curve overlay ──
    st.subheader("Equity Curves — Hold/Cut Variants vs Buy-and-Hold")
    ec = load_filtered_equity_curves()
    if ec.empty:
        st.warning("No equity curve files found. Run src.analysis.filtered_hold_cut first.")
    else:
        fig_ec = px.line(
            ec.sort_values("eps_date"),
            x="eps_date", y="equity", color="variant", log_y=True,
            title="Compounded equity (log scale, $1 start)",
            labels={"eps_date": "Event date", "equity": "Equity ($, log)", "variant": ""},
            color_discrete_map={
                "Unfiltered Hold/Cut":    "#95a5a6",
                "+ MACD bullish":         "#27ae60",
                "+ Above SMA-50":         "#2980b9",
                "+ Not RSI-oversold":     "#f39c12",
                "Buy-and-Hold benchmark": "#e74c3c",
            },
            height=500,
        )
        fig_ec.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig_ec.update_layout(plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig_ec, width="stretch")
        st.caption(
            "Log-scale — the gap between '+ MACD bullish' and 'Unfiltered' is the "
            "regime filter's alpha contribution. The '+ MACD bullish' and "
            "'+ Above SMA-50' curves ride above everything else by design "
            "(they concentrate capital into the winner subset)."
        )

    st.divider()

    # ── Per-event mean-return bar ──
    st.subheader("Mean Per-Event Return by Variant")
    bars = summary.copy()
    bars["mean_pct"] = bars["mean"] * 100
    bars["hit_pct"] = bars["hit_rate"] * 100
    bars["label"] = bars["variant"].map({
        "unfiltered_hold_cut":       "Unfiltered Hold/Cut",
        "hold_cut_macd_bullish":     "+ MACD bullish",
        "hold_cut_above_sma50":      "+ Above SMA-50",
        "hold_cut_not_rsi_oversold": "+ Not RSI-oversold",
        "buyhold_benchmark":         "Buy-and-Hold",
    })
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bars["label"], y=bars["mean_pct"],
        marker_color=[
            "#95a5a6", "#27ae60", "#2980b9", "#f39c12", "#e74c3c",
        ],
        text=[f"n={int(n)}<br>hit={h:.0f}%<br>IR={ir:.2f}"
              for n, h, ir in zip(bars["n"], bars["hit_pct"], bars["info_ratio"])],
        textposition="outside",
    ))
    fig_bar.add_hline(
        y=baseline["mean"] * 100, line_dash="dash", line_color="#333",
        annotation_text="Unfiltered baseline",
    )
    fig_bar.update_layout(
        title="Mean per-event return (%) — error bars suppressed for clarity; "
              "see IR and t-stat in table for dispersion",
        yaxis_title="Mean (%)",
        xaxis_tickangle=-15,
        height=420,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#333"),
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.divider()

    # ── Regime grid heatmap ──
    st.subheader("Regime × Strategy Interaction Grid")
    strat_choice = st.selectbox(
        "Strategy",
        ["hold_cut", "momentum_1_H", "pre_runup_P_1"],
        format_func=lambda s: {
            "hold_cut":        "Hold/Cut (thr=5%, ext=+10)",
            "momentum_1_H":    "Momentum (+1..+H)",
            "pre_runup_P_1":   "Pre Run-up (-P..-1)",
        }[s],
    )
    grid = load_regime_grid(strat_choice)
    if grid.empty:
        st.warning(f"No regime grid found for {strat_choice}.")
    else:
        base = grid[grid["regime"] == "__baseline__"].iloc[0]
        body = grid[
            (grid["regime"] != "__baseline__")
            & (grid["n"] >= 50)
            & (grid["bucket"].astype(str) != "nan")
        ].copy()
        body["bucket"] = body["bucket"].astype(str)
        body["mean_uplift_pp"] = (body["mean"] - base["mean"]) * 100
        body["label"] = body["regime"] + " = " + body["bucket"]
        body = body.sort_values("mean_uplift_pp", ascending=True)

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=body["mean_uplift_pp"], y=body["label"], orientation="h",
            marker_color=[
                "#27ae60" if v >= 0 else "#e74c3c" for v in body["mean_uplift_pp"]
            ],
            text=[
                f"n={int(n)}  mean={m*100:+.2f}%  hit={h*100:.0f}%  t={t:+.1f}"
                for n, m, h, t in zip(body["n"], body["mean"], body["hit_rate"], body["t_stat"])
            ],
            textposition="outside",
        ))
        fig_h.add_vline(x=0, line_color="#333")
        fig_h.update_layout(
            title=f"Mean-return uplift (pp) vs baseline ({base['mean']*100:+.2f}%) — "
                  f"buckets with n ≥ 50",
            xaxis_title="Mean-return uplift (percentage points)",
            height=40 * len(body) + 140,
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"),
            margin=dict(l=280, r=140),
        )
        st.plotly_chart(fig_h, width="stretch")
        st.caption(
            "Hold/Cut splits cleanly into winner/loser halves under every trend "
            "feature. Momentum (+1..+H) and Pre Run-up (-P..-1) are much flatter "
            "— effectively regime-inert under these features. "
            "`macd_hist_positive` is mathematically identical to `macd_bullish` "
            "(since macd_hist = macd - macd_signal); treat as one feature when "
            "applying multiple-testing corrections."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Live Monitor (forward-looking watchlist + rolling edge metrics)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_monitor_watchlist():
    path = os.path.join(DATA_DIR, "watchlist.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["eps_date", "entry_date"])


@st.cache_data
def load_monitor_summary():
    path = os.path.join(DATA_DIR, "monitoring_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["eps_date"])


@st.cache_data
def load_monitor_health():
    path = os.path.join(DATA_DIR, "monitoring_health.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


if page == "Live Monitor":
    st.title("Live Monitor — MACD-bullish Hold/Cut")
    st.caption(
        "Forward-looking watchlist of upcoming S&P 100 earnings events with the "
        "strategy's MACD filter state at the day-−10 entry close, plus rolling "
        "edge metrics on realised trades. Refresh by running "
        "`python -m src.monitor` after every data pull."
    )

    wl = load_monitor_watchlist()
    rm = load_monitor_summary()
    health = load_monitor_health()

    if wl.empty:
        st.warning(
            "No watchlist.csv found under `data/`. Run `python -m src.monitor` "
            "after pulling indicators and the earnings calendar."
        )
        st.stop()

    # ── Top KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    actionable = wl[wl["status"].isin(["entry-imminent", "entered", "pre-watch"])]
    bullish_actionable = actionable[actionable["macd_bullish"] == True]  # noqa: E712
    c1.metric("Upcoming events", len(actionable))
    c2.metric(
        "MACD-bullish (filter fires)",
        len(bullish_actionable),
        delta=f"{(len(bullish_actionable) / max(len(actionable), 1) * 100):.0f}% pass",
    )
    # Latest rolling window metrics vs baseline.
    if not rm.empty:
        last = rm.iloc[-1]
        c3.metric(
            "Rolling-30 mean",
            f"{last['rolling_mean']*100:+.2f}%",
            delta=f"{(last['rolling_mean']-0.058)*100:+.2f}pp vs baseline",
        )
        c4.metric(
            "Rolling-30 hit-rate",
            f"{last['rolling_hit_rate']*100:.1f}%",
            delta=f"{(last['rolling_hit_rate']-0.81)*100:+.1f}pp vs baseline",
        )
    else:
        c3.metric("Rolling-30 mean", "—")
        c4.metric("Rolling-30 hit-rate", "—")

    # ── Data-quality badges ──
    if not health.empty:
        h = health.iloc[0]
        st.markdown("#### Pipeline health")
        b1, b2, b3, b4 = st.columns(4)

        def _badge(val, ok_val):
            if ok_val is True or ok_val == "True":
                return f"✅ {val}"
            if ok_val is False or ok_val == "False":
                return f"⚠️ {val}"
            return f"— {val}"

        b1.markdown(f"**Indicators fresh to**\n\n{h.get('indicators_last_date')}")
        b2.markdown(
            "**Calendar match**\n\n"
            + _badge(f"{float(h['calendar_match_rate'])*100:.1f}%", h["calendar_match_rate_ok"])
        )
        b3.markdown(
            "**SEC cross-check**\n\n"
            + _badge(f"{float(h['sec_match_rate'])*100:.1f}%", h["sec_match_rate_ok"])
        )
        b4.markdown(
            "**AMC shift delta**\n\n"
            + _badge(f"{float(h['amc_mean_delta_ret'])*100:+.2f}%", h["amc_delta_in_band"])
        )

    st.divider()

    # ── Watchlist table ──
    st.subheader("Watchlist")
    status_options = ["entry-imminent", "entered", "pre-watch", "post-event", "closed"]
    chosen_statuses = st.multiselect(
        "Show rows with status",
        options=status_options,
        default=["entry-imminent", "entered", "pre-watch"],
    )
    only_bullish = st.checkbox(
        "Only MACD-bullish (filter would fire)", value=True
    )
    wl_view = wl[wl["status"].isin(chosen_statuses)].copy()
    if only_bullish:
        wl_view = wl_view[wl_view["macd_bullish"] == True]  # noqa: E712

    display_cols = [
        "ticker", "eps_date", "timing", "entry_date",
        "days_to_entry", "days_to_eps", "status",
        "macd", "macd_signal", "macd_bullish", "rsi_14",
        "eps_estimate", "realised_ret",
    ]
    wl_disp = wl_view[display_cols].copy()
    wl_disp["eps_date"] = wl_disp["eps_date"].dt.date
    wl_disp["entry_date"] = wl_disp["entry_date"].dt.date
    for c in ("macd", "macd_signal"):
        wl_disp[c] = wl_disp[c].map(lambda v: f"{v:+.2f}" if pd.notnull(v) else "—")
    wl_disp["rsi_14"] = wl_disp["rsi_14"].map(lambda v: f"{v:.1f}" if pd.notnull(v) else "—")
    wl_disp["realised_ret"] = wl_disp["realised_ret"].map(
        lambda v: f"{v*100:+.2f}%" if pd.notnull(v) else "—"
    )
    wl_disp["eps_estimate"] = wl_disp["eps_estimate"].map(
        lambda v: f"{v:.2f}" if pd.notnull(v) else "—"
    )
    st.dataframe(wl_disp, width="stretch", hide_index=True)
    st.caption(
        "`days_to_entry` is trading days from today to the day-−10 entry close. "
        "Negative means we'd already be in the position. `macd_bullish = macd_line > "
        "macd_signal` at the entry close (for observed entries) or at the latest "
        "indicator close (preview, for future entries). **Rows above the filter "
        "line are the actionable trades.**"
    )

    st.divider()

    # ── Rolling edge metrics ──
    st.subheader("Rolling edge — MACD-bullish realised trades")
    if rm.empty:
        st.info("No realised MACD-bullish events available yet.")
    else:
        rm_plot = rm.dropna(subset=["rolling_mean"]).copy()
        fig_edge = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Rolling mean per event",
                                                  "Rolling hit-rate"),
                                  vertical_spacing=0.12)
        fig_edge.add_trace(
            go.Scatter(x=rm_plot["eps_date"], y=rm_plot["rolling_mean"]*100,
                       mode="lines", name="rolling mean %", line=dict(color="#2ecc71")),
            row=1, col=1,
        )
        fig_edge.add_hline(y=5.80, line_dash="dash", line_color="#27ae60",
                            annotation_text="baseline 5.80%", row=1, col=1)
        fig_edge.add_hline(y=3.00, line_dash="dot", line_color="#e74c3c",
                            annotation_text="alarm <3%", row=1, col=1)
        fig_edge.add_trace(
            go.Scatter(x=rm_plot["eps_date"], y=rm_plot["rolling_hit_rate"]*100,
                       mode="lines", name="rolling hit-rate %", line=dict(color="#3498db")),
            row=2, col=1,
        )
        fig_edge.add_hline(y=81.0, line_dash="dash", line_color="#2980b9",
                            annotation_text="baseline 81%", row=2, col=1)
        fig_edge.add_hline(y=70.0, line_dash="dot", line_color="#e74c3c",
                            annotation_text="alarm <70%", row=2, col=1)
        fig_edge.update_layout(height=520, showlegend=False,
                                margin=dict(l=40, r=20, t=40, b=40))
        fig_edge.update_yaxes(title_text="%", row=1, col=1)
        fig_edge.update_yaxes(title_text="%", row=2, col=1)
        st.plotly_chart(fig_edge, width="stretch")

        if bool(last["alarm_mean_confirmed"]) or bool(last["alarm_hit_confirmed"]):
            st.error(
                "Alarm: rolling edge has fallen outside the band for 2 "
                "consecutive windows — review filter and recent announcements "
                "before trading the next event."
            )
        else:
            st.success(
                f"Edge intact: rolling mean {last['rolling_mean']*100:+.2f}% "
                f"and hit-rate {last['rolling_hit_rate']*100:.1f}% both inside "
                f"the alarm band as of {last['eps_date']:%Y-%m-%d}."
            )

    st.caption(
        "Monitor logic lives in `src/monitor.py`. Re-run "
        "`python -m src.monitor` after each `src.pull_sp100` refresh. "
        "See `Report_PositionSizing.docx` for the sizing rule and "
        "`Report_DataQuality.docx` for the AMC entry-shift correction."
    )
