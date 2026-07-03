import streamlit as st
import pandas as pd
from src.utils import get_currency_symbol


def render_risk_analysis(risk, simulator, loader):
    """
    Render the Risk Analysis tab.
    """
    symbol = get_currency_symbol(loader.company_info.get("currency", "USD"))

    st.title("📉 Risk Analysis")

    st.caption(
        "Understand the downside risk and probability distribution "
        "of the Monte Carlo simulations."
    )

    st.divider()

    # ==========================================================
    # Probability Analysis
    # ==========================================================

    st.subheader("Probability Analysis")

    probability_table = pd.DataFrame({

        "Scenario":[

            "Price ≥ Current",

            "Price ≥ +5%",

            "Price ≥ +10%",

            "Price ≥ +20%",

            "Price ≤ -5%",

            "Price ≤ -10%",

            "Price ≤ -20%"

        ],

        "Probability":[

            f"{risk.probability_above(risk.current_price):.2f}%",

            f"{risk.probability_above(risk.current_price*1.05):.2f}%",

            f"{risk.probability_above(risk.current_price*1.10):.2f}%",

            f"{risk.probability_above(risk.current_price*1.20):.2f}%",

            f"{risk.probability_below(risk.current_price*0.95):.2f}%",

            f"{risk.probability_below(risk.current_price*0.90):.2f}%",

            f"{risk.probability_below(risk.current_price*0.80):.2f}%"

        ]

    })

    st.dataframe(
        probability_table,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # Percentile Analysis
    # ==========================================================

    st.subheader("Price Percentiles")

    percentile_table = pd.DataFrame({

        "Percentile":[
            "5%",
            "25%",
            "50% (Median)",
            "75%",
            "95%"
        ],

        "Final Price":[

            f"{symbol}{risk.percentile(5):.2f}",

            f"{symbol}{risk.percentile(25):.2f}",

            f"{symbol}{risk.percentile(50):.2f}",

            f"{symbol}{risk.percentile(75):.2f}",

            f"{symbol}{risk.percentile(95):.2f}"

        ]

    })

    st.dataframe(
        percentile_table,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # VaR & CVaR
    # ==========================================================

    st.subheader("Risk Metrics")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Value at Risk (95%)",
            f"{symbol}{risk.value_at_risk():.2f}"
        )

    with c2:

        st.metric(
            "Conditional VaR (95%)",
            f"{symbol}{risk.conditional_var():.2f}"
        )

    st.divider()

    # ==========================================================
    # Metric Explanations
    # ==========================================================

    st.subheader("Metric Explanations")

    with st.expander("📈 Annualized Return"):

        st.markdown(f"""
**What is it?**

Annualized historical return estimated from the downloaded historical data.

**Historical Window**

Used the average daily log return from the selected historical period.

**Important**

This is **NOT** the predicted return next year.

It is the historical drift (μ) used inside the
Geometric Brownian Motion simulation.
""")

    with st.expander("📊 Annualized Volatility"):

        st.markdown("""
Annualized volatility estimates how much the stock price
typically fluctuates over one year.

Formula

σ × √252

Higher values indicate greater uncertainty.
""")

    with st.expander("⚖️ Sharpe Ratio"):

        st.markdown("""
Measures risk-adjusted return.

Formula

(Return − Risk-Free Rate) / Volatility

Interpretation

• <0 Poor

• 0–1 Moderate

• 1–2 Good

• >2 Excellent
""")

    with st.expander("📉 Value at Risk (VaR)"):

        st.markdown(f"""
VaR estimates the maximum expected loss over the selected
forecast horizon.

Current Horizon

**{simulator.days} Trading Days**

Example

A VaR of **{symbol}{risk.value_at_risk():.2f}**

means there is only a **5% probability**
of losing more than this amount.
""")

    with st.expander("📉 Conditional VaR (CVaR)"):

        st.markdown(f"""
CVaR measures the **average loss**
if the loss exceeds VaR.

Current CVaR

**{symbol}{risk.conditional_var():.2f}**

It focuses on the worst simulated outcomes.
""")

    with st.expander("🎯 Expected Price"):

        st.markdown("""
Expected Price is the arithmetic average of all simulated
ending prices.

It should not be interpreted as a guaranteed future price.

Instead, it represents the center of the simulated
probability distribution.
""")

    with st.expander("📦 Confidence Interval"):

        lower, upper = risk.confidence_interval()

        st.markdown(f"""
95% Confidence Interval

**{symbol}{lower:.2f} — {symbol}{upper:.2f}**

Approximately 95% of simulated prices
ended inside this range.
""")