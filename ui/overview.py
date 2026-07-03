import streamlit as st
import pandas as pd
from src.utils import get_currency_symbol


def render_overview(loader, simulator, risk, visualizer):
    """
    Render the Overview dashboard.
    """
    symbol = get_currency_symbol(loader.company_info.get("currency", "USD"))

    # ==========================================================
    # Header
    # ==========================================================

    st.title("📈 Monte Carlo Stock Risk Analyzer")

    st.caption(
        "Geometric Brownian Motion (GBM) based stochastic stock price simulation."
    )

    st.divider()

    # ==========================================================
    # Simulation Summary
    # ==========================================================

    start = loader.data.index[0].date()
    end = loader.data.index[-1].date()

    trading_days = len(loader.data)

    st.info(
        f"""
**Simulation Summary**

• **Ticker:** {loader.ticker}

• **Historical Window:** {start} → {end}

• **Trading Days Used:** {trading_days}

• **Forecast Horizon:** {simulator.days} Trading Days

• **Monte Carlo Simulations:** {simulator.simulations:,}

• **Model:** {getattr(simulator, 'model_type', 'gbm').upper().replace('_', ' ')}
"""
    )

    st.divider()

    # ==========================================================
    # Metric Cards
    # ==========================================================

    st.subheader("Key Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"{symbol}{risk.current_price:.2f}"
    )

    c2.metric(
        "Expected Price",
        f"{symbol}{risk.expected_price():.2f}"
    )

    c3.metric(
        "Annualized Return",
        f"{risk.annualized_return():.2f}%"
    )

    c4.metric(
        "Sharpe Ratio",
        f"{risk.sharpe_ratio():.2f}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annual Volatility",
        f"{risk.annualized_volatility():.2f}%"
    )

    c2.metric(
        "VaR (95%)",
        f"{symbol}{risk.value_at_risk():.2f}"
    )

    c3.metric(
        "CVaR (95%)",
        f"{symbol}{risk.conditional_var():.2f}"
    )

    c4.metric(
        "P(Final ≥ Current)",
        f"{risk.probability_above(risk.current_price):.2f}%"
    )

    st.divider()

    # ==========================================================
    # Charts
    # ==========================================================

    st.plotly_chart(
        visualizer.plot_price_paths(),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            visualizer.plot_distribution(),
            use_container_width=True
        )

    with right:

        st.plotly_chart(
            visualizer.plot_fan_chart(),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            visualizer.plot_return_distribution(),
            use_container_width=True
        )

    with right:

        st.plotly_chart(
            visualizer.plot_var(),
            use_container_width=True
        )

    st.divider()

    # ==========================================================
    # Risk Table
    # ==========================================================

    st.subheader("Risk Summary")

    lower, upper = risk.confidence_interval()

    df = pd.DataFrame({

        "Metric":[

            "Expected Price",

            "Median Price",

            "5th Percentile",

            "95th Percentile",

            "Standard Deviation",

            "Annualized Return",

            "Annualized Volatility",

            "Sharpe Ratio",

            "95% Confidence Lower",

            "95% Confidence Upper",

            "VaR (95%)",

            "CVaR (95%)"

        ],

        "Value":[

            f"{symbol}{risk.expected_price():.2f}",

            f"{symbol}{risk.median_price():.2f}",

            f"{symbol}{risk.percentile(5):.2f}",

            f"{symbol}{risk.percentile(95):.2f}",

            f"{symbol}{risk.standard_deviation():.2f}",

            f"{risk.annualized_return():.2f}%",

            f"{risk.annualized_volatility():.2f}%",

            f"{risk.sharpe_ratio():.2f}",

            f"{symbol}{lower:.2f}",

            f"{symbol}{upper:.2f}",

            f"{symbol}{risk.value_at_risk():.2f}",

            f"{symbol}{risk.conditional_var():.2f}"

        ]

    })

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # Downloads
    # ==========================================================

    st.subheader("Downloads")

    csv = simulator.simulated_prices.to_csv(
        index=False
    ).encode()

    st.download_button(

        "📥 Download Monte Carlo Simulations",

        data=csv,

        file_name=f"{loader.ticker}_simulation.csv",

        mime="text/csv",

        use_container_width=True

    )