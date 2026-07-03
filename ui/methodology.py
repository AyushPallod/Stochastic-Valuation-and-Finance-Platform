import streamlit as st
import numpy as np


def render_methodology(loader, simulator, risk):
    """
    Explains the methodology used in the Monte Carlo simulation.
    """
    from src.utils import get_currency_symbol
    symbol = get_currency_symbol(loader.company_info.get("currency", "USD"))

    st.title("📚 Methodology")

    st.caption(
        "Understand how the simulation is performed and how each "
        "financial metric is calculated."
    )

    st.divider()

    # ==========================================================
    # Workflow
    # ==========================================================

    st.header("Simulation Workflow")

    st.markdown("""
    Historical Prices
│
▼
Daily Log Returns
│
▼
Estimate Drift (μ)
Estimate Volatility (σ)
│
▼
Geometric Brownian Motion
│
▼
Monte Carlo Simulation
│
▼
Risk Metrics & Visualization
                """)

    st.divider()

    # ==========================================================
    # Historical Data
    # ==========================================================

    with st.expander("① Historical Data"):

        st.markdown(f"""
**Ticker**

`{loader.ticker}`

**Historical Window**

{loader.data.index[0].date()} → {loader.data.index[-1].date()}

**Trading Days**

{len(loader.data)}

Yahoo Finance is used to download historical closing prices.
The historical period selected by the user is used to estimate
daily return and volatility.
""")

    # ==========================================================
    # Log Returns
    # ==========================================================

    with st.expander("② Daily Log Returns"):

        st.latex(
            r"r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)"
        )

        import numpy as np
        if simulator.is_portfolio:
            daily_mu = float(np.sum(simulator.mu * simulator.weights))
            mu_str = f"Weighted Portfolio μ: **{daily_mu:.6f}**\n\nIndividual Asset Drifts:\n"
            for ticker, val in simulator.mu.items():
                mu_str += f"- **{ticker}**: {val:.6f}\n"
        else:
            mu_str = f"**{simulator.mu:.6f}**"

        st.markdown(f"""
Average Daily Log Return (μ)

{mu_str}

Log returns are preferred over simple returns because they are
time additive and widely used in quantitative finance.
""")

    # ==========================================================
    # Volatility
    # ==========================================================

    with st.expander("③ Daily Volatility"):

        st.latex(
            r"\sigma=\mathrm{Std}(r_t)"
        )

        import numpy as np
        if simulator.is_portfolio:
            w = simulator.weights
            cov_matrix = simulator.cov.values
            portfolio_variance = w.T @ cov_matrix @ w
            daily_vol = np.sqrt(portfolio_variance)
            
            st.markdown(f"""
Estimated Daily Volatility

Weighted Portfolio Volatility: **{daily_vol:.6f}**

Annualized Historical Volatility

**{risk.annualized_volatility():.2f}%**

The annualized volatility is estimated from the selected
historical window.

##### Historical Covariance Matrix
""")
            st.dataframe(simulator.cov, width="stretch")
        else:
            st.markdown(f"""
Estimated Daily Volatility

**{simulator.sigma:.6f}**

Annualized Historical Volatility

**{risk.annualized_volatility():.2f}%**

The annualized volatility is estimated from the selected
historical window.
""")

    # ==========================================================
    # GBM
    # ==========================================================

    with st.expander("④ Geometric Brownian Motion"):

        st.latex(
            r"S_{t+1}=S_t\exp\left((\mu-\frac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z\right)"
        )

        import numpy as np
        if simulator.is_portfolio:
            daily_mu = float(np.sum(simulator.mu * simulator.weights))
            w = simulator.weights
            cov_matrix = simulator.cov.values
            daily_vol = np.sqrt(w.T @ cov_matrix @ w)
            gbm_desc = f"""
Where (Portfolio aggregates)

• μ (weighted drift) = {daily_mu:.6f}

• σ (portfolio daily volatility) = {daily_vol:.6f}

• Z ~ N(0, I) (correlated via Cholesky factor matrix L)
"""
        else:
            gbm_desc = f"""
Where

• μ = {simulator.mu:.6f}

• σ = {simulator.sigma:.6f}

• Z ~ N(0, 1)
"""

        st.markdown(f"""
{gbm_desc}

Each simulation randomly generates future daily prices using
this stochastic differential equation.
""")

    # ==========================================================
    # Monte Carlo
    # ==========================================================

    with st.expander("⑤ Monte Carlo Simulation"):

        st.markdown(f"""
Number of Simulations

**{simulator.simulations:,}**

Forecast Horizon

**{simulator.days} Trading Days**

Each simulation generates one possible future price path.

The Expected Price is the average ending price across all
simulations.
""")

    # ==========================================================
    # Annual Return
    # ==========================================================

    with st.expander("⑥ Annualized Historical Return"):

        st.latex(
            r"R=e^{\mu\times252}-1"
        )

        st.markdown(f"""
Estimated Annualized Historical Return

**{risk.annualized_return():.2f}%**

This is computed from the average daily log return over the
selected historical window.

It is **NOT** a prediction of next year's return.

Instead, it is the historical drift used in the GBM model.
""")

    # ==========================================================
    # Sharpe Ratio
    # ==========================================================

    with st.expander("⑦ Sharpe Ratio"):

        st.latex(
            r"Sharpe=\frac{R-R_f}{\sigma}"
        )

        st.markdown(f"""
Current Sharpe Ratio

**{risk.sharpe_ratio():.2f}**

Interpretation

| Sharpe | Meaning |
|---------|---------|
| <0 | Poor |
| 0–1 | Moderate |
| 1–2 | Good |
| >2 | Excellent |
""")

    # ==========================================================
    # VaR
    # ==========================================================

    with st.expander("⑧ Value at Risk (VaR)"):

        st.markdown(f"""
Current VaR (95%)

**{symbol}{risk.value_at_risk():.2f}**

Interpretation

There is a 95% probability that losses will NOT exceed this
amount over the selected forecast horizon.

There remains a 5% probability of larger losses.
""")

    # ==========================================================
    # CVaR
    # ==========================================================

    with st.expander("⑨ Conditional Value at Risk (CVaR)"):

        st.markdown(f"""
Current CVaR

**{symbol}{risk.conditional_var():.2f}**

CVaR measures the average loss when the VaR threshold has
already been exceeded.

It focuses on the worst simulated outcomes.
""")

    st.divider()

    # ==========================================================
    # Assumptions
    # ==========================================================

    st.header("Model Assumptions")

    st.markdown("""
- Stock prices follow Geometric Brownian Motion.
- Daily returns are independent.
- Drift and volatility remain constant during the forecast.
- Returns are approximately normally distributed.
- No transaction costs or dividends are considered.
- No macroeconomic events are modeled.
""")

    st.divider()

    # ==========================================================
    # Limitations
    # ==========================================================

    st.header("Limitations")

    st.warning("""
This simulator is designed for educational and analytical
purposes.

Real financial markets exhibit:

- Volatility clustering
- Fat tails
- Market crashes
- Regime changes
- Interest rate effects
- Earnings surprises

These effects are not captured by the standard Geometric
Brownian Motion model.
""")