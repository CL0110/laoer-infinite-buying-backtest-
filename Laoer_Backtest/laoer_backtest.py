import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Laoer Infinite Buying Strategy Backtester", layout="wide")
st.title("Laoer Infinite Buying Strategy Backtester")
st.markdown("""
Does the Laoer infinite buying strategy work? This tool backtests the **daily 40-partition cost-averaging 
approach with LOC orders and 10% profit targets** on 3x leveraged ETFs (TQQQ, SOXL, etc.). 

**Core mechanics:**
- Divide capital into 40 equal daily tranches
- Each day: buy half at your average cost (LOC), buy half at high limit (forced fill)
- Exit entire position when it reaches +10% profit
- If capital exhausts before hitting +10%, deploy 1/4 loss rule to recover capital
- Track portfolio value, cycles completed, and capital depletion timeline

Adjust the ticker, capital, partition count, profit target, and date range in the sidebar to run scenarios.

**Data source:** Prices via [yfinance](https://pypi.org/project/yfinance/).
""")

with st.expander("ℹ️ About Laoer Strategy"):
    st.markdown("""
    The Laoer infinite buying strategy is a quant-based cost-averaging approach designed to exploit volatility 
    in 3x leveraged ETFs. By mechanically dividing capital and placing daily LOC orders, the strategy attempts 
    to profit from mean reversion in leveraged products. Key risks: leverage decay in sideways markets, capital 
    exhaustion in sustained downtrends, and volatility drag.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Strategy Parameters")

ticker = st.sidebar.selectbox("3x Leveraged ETF", ["TQQQ", "SOXL", "UPRO", "BULZ"], 
                               help="Select your 3x leveraged ETF")
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000,
                                          help="Total capital to divide into daily tranches")
partitions = st.sidebar.slider("Partition Count", min_value=10, max_value=60, value=40,
                               help="Total number of daily tranches (40 is Laoer's standard)")
profit_target = st.sidebar.slider("Profit Target (%)", min_value=5, max_value=30, value=10,
                                  help="Exit when portfolio reaches this profit % (Laoer: 10%)")
high_limit_pct = st.sidebar.slider("High Limit Buy (%)", min_value=5, max_value=20, value=10,
                                   help="Price level for forced-fill half of daily buy (vs. yesterday close)")

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")
loss_rule_enabled = st.sidebar.checkbox("Enable 1/4 Loss Rule", value=True,
                                        help="When capital exhausts, sell 25% to recover cash and keep buying")
use_hloc = st.sidebar.checkbox("Model Realistic LOC Fills", value=True,
                               help="Half-quantity fills only when price dips to avg cost; half forced at limit")

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST LOGIC
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner(f"Backtesting {ticker} from {start_date} to {end_date}..."):
    # Download price data
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}. Check ticker and date range.")
            st.stop()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()
    
    prices = data["Close"].dropna()
    dates = prices.index
    
    # Initialize backtest state
    daily_tranche = initial_capital / partitions
    
    portfolio_values = []
    dates_out = []
    shares_held = 0.0
    avg_cost = 0.0
    cash = initial_capital
    cycles_completed = 0
    capital_exhausted_date = None
    high_water_mark = initial_capital
    
    daily_logs = []  # For debugging/analysis
    
    for i in range(len(prices)):
        price = prices.iloc[i]
        dt = dates[i]
        
        # ─── DAILY ENTRY LOGIC ───
        if cash > 0 and i > 0:
            prev_close = prices.iloc[i - 1]
            high_limit = prev_close * (1 + high_limit_pct / 100.0)
            
            # Half-quantity at average cost (realistic LOC)
            if use_hloc and avg_cost > 0:
                # Only fill if price dips to avg cost or lower
                if price <= avg_cost and cash >= daily_tranche / 2:
                    shares_bought = (daily_tranche / 2) / avg_cost
                    shares_held += shares_bought
                    cash -= daily_tranche / 2
                    avg_cost = (avg_cost * (shares_held - shares_bought) + (daily_tranche / 2)) / shares_held if shares_held > 0 else avg_cost
            else:
                # First tranche or LOC disabled: buy at current price
                if cash >= daily_tranche / 2:
                    shares_bought = (daily_tranche / 2) / price
                    if avg_cost == 0:
                        avg_cost = price
                    else:
                        avg_cost = (avg_cost * shares_held + daily_tranche / 2) / (shares_held + shares_bought)
                    shares_held += shares_bought
                    cash -= daily_tranche / 2
            
            # Half-quantity at high limit (forced fill) — always executes to guarantee
            # daily deployment, filled at whichever is better: actual price or the high-limit cap
            if cash >= daily_tranche / 2:
                fill_price = min(price, high_limit)
                shares_bought = (daily_tranche / 2) / fill_price
                if avg_cost == 0:
                    avg_cost = fill_price
                else:
                    avg_cost = (avg_cost * shares_held + daily_tranche / 2) / (shares_held + shares_bought)
                shares_held += shares_bought
                cash -= daily_tranche / 2
            
            # Check if capital exhausted
            if cash < daily_tranche and capital_exhausted_date is None:
                capital_exhausted_date = dt
        
        # ─── EXIT LOGIC (10% profit target) ───
        if shares_held > 0 and avg_cost > 0:
            position_value = shares_held * price
            profit_pct = (price - avg_cost) / avg_cost
            
            if profit_pct >= profit_target / 100.0:
                # Exit entire position
                cash += position_value
                cycles_completed += 1
                shares_held = 0.0
                avg_cost = 0.0
        
        # ─── QUARTER-LOSS RULE ───
        if loss_rule_enabled and cash < daily_tranche and shares_held > 0:
            # Sell 25% to recover cash
            shares_to_sell = shares_held * 0.25
            proceeds = shares_to_sell * price
            cash += proceeds
            shares_held -= shares_to_sell
        
        # ─── PORTFOLIO VALUE ───
        position_value = shares_held * price if shares_held > 0 else 0
        total_value = cash + position_value
        portfolio_values.append(total_value)
        dates_out.append(dt)
        high_water_mark = max(high_water_mark, total_value)
        
        daily_logs.append({
            "date": dt,
            "price": price,
            "cash": cash,
            "shares": shares_held,
            "avg_cost": avg_cost,
            "position_value": position_value,
            "total_value": total_value,
            "profit_pct": (price - avg_cost) / avg_cost if avg_cost > 0 else 0
        })
    
    # Convert to series
    strat = pd.Series(portfolio_values, index=dates_out)
    
    # ─── PERFORMANCE METRICS ───
    def metrics(series):
        rets = series.pct_change().dropna()
        n_years = len(rets) / 252
        if n_years < 0.01:
            n_years = 0.01  # Avoid division by zero
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        roll_max = series.cummax()
        drawdown = ((series - roll_max) / roll_max).min()
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        return {
            "CAGR": f"{cagr:.2%}",
            "Total Return": f"{total_return:.2%}",
            "Volatility": f"{vol:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Max Drawdown": f"{drawdown:.2%}",
            "Final Value": f"${series.iloc[-1]:,.0f}"
        }
    
    s_metrics = metrics(strat)
    
    # Buy & hold benchmark
    benchmark = (prices / prices.iloc[0]) * initial_capital
    b_metrics = metrics(benchmark)
    
    # ─── CHART 1: Cumulative Portfolio Value ───
    st.subheader("Cumulative Portfolio Value")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strat.index, y=strat, name="Laoer Strategy",
                             line=dict(color="#0366d6", width=2)))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name=f"{ticker} Buy & Hold",
                             line=dict(color="#f0883e", width=2, dash="dash")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                      hovermode="x unified", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # ─── CHART 2: Drawdown ───
    st.subheader("Drawdown Analysis")
    strat_dd = (strat - strat.cummax()) / strat.cummax()
    bench_dd = (benchmark - benchmark.cummax()) / benchmark.cummax()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name="Laoer Strategy",
                              fill="tozeroy", line=dict(color="#0366d6")))
    fig2.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd, name=f"{ticker} Buy & Hold",
                              fill="tozeroy", line=dict(color="#f0883e")))
    fig2.update_layout(yaxis_tickformat=".0%", template="plotly_white", height=300)
    st.plotly_chart(fig2, use_container_width=True)
    
    # ─── METRICS ───
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Laoer Strategy")
        for k, v in s_metrics.items():
            st.metric(k, v)
    with col2:
        st.subheader(f"{ticker} Buy & Hold")
        for k, v in b_metrics.items():
            st.metric(k, v)
    
    # ─── STRATEGY STATS ───
    st.subheader("Strategy Statistics")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Cycles Completed", cycles_completed)
    with col4:
        if capital_exhausted_date:
            days_to_exhaust = (capital_exhausted_date - start_date).days
            st.metric("Days to Capital Exhaustion", days_to_exhaust)
        else:
            st.metric("Capital Exhaustion", "Never")
    with col5:
        final_shares = daily_logs[-1]["shares"]
        final_avg_cost = daily_logs[-1]["avg_cost"]
        st.metric("Final Shares Held", f"{final_shares:.0f}")
    
    # ─── DOWNLOAD ───
    results_df = pd.DataFrame({
        "Date": strat.index,
        "Laoer Strategy": strat.values,
        f"{ticker} Buy & Hold": benchmark.values,
        "Drawdown (%)": (strat_dd * 100).values
    })
    st.download_button("Download Results CSV", results_df.to_csv(index=False).encode(),
                       f"laoer_backtest_{ticker}.csv", "text/csv")
    
    # ─── DETAILED LOG (optional) ───
    with st.expander("📊 Daily Transaction Log"):
        log_df = pd.DataFrame(daily_logs)
        log_df["date"] = log_df["date"].dt.date
        st.dataframe(log_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Adjust parameters above to instantly re-run the backtest.")
