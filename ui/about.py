import streamlit as st


def render_about():
    """
    About the Monte Carlo Stock Risk Analyzer project.
    """

    st.title("ℹ️ About")

    st.caption(
        "Project information, architecture, assumptions and future roadmap."
    )

    st.divider()

    # ==========================================================
    # Project
    # ==========================================================

    st.header("Project")

    st.markdown("""
### Monte Carlo Stock Risk Analyzer

This application estimates possible future stock prices using
**Monte Carlo Simulation** based on the
**Geometric Brownian Motion (GBM)** model.

Instead of predicting one future price,
the simulator generates thousands of possible price paths,
allowing the user to understand uncertainty and downside risk.

The project combines financial mathematics,
probability theory and Python visualization to provide an
interactive quantitative finance dashboard.
""")

    st.divider()

    # ==========================================================
    # Features
    # ==========================================================

    st.header("Features")

    st.markdown("""
✅ Historical data download using Yahoo Finance

✅ Daily log return calculation

✅ Geometric Brownian Motion simulation

✅ Monte Carlo price forecasting

✅ Risk Analysis

- Expected Price
- Confidence Interval
- Percentiles
- Annualized Historical Return
- Annualized Historical Volatility
- Sharpe Ratio
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)

✅ Interactive visualizations

- Monte Carlo Paths
- Price Distribution
- Fan Chart
- Return Distribution
- VaR Analysis

✅ CSV Export
""")

    st.divider()

    # ==========================================================
    # Technologies
    # ==========================================================

    st.header("Technology Stack")

    st.markdown("""
| Component | Technology |
|-----------|------------|
| Language | Python |
| Dashboard | Streamlit |
| Numerical Computing | NumPy |
| Data Processing | pandas |
| Stock Data | yfinance |
| Visualization | Matplotlib |
| Risk Analytics | Custom Python |
""")

    st.divider()

    # ==========================================================
    # Mathematical Model
    # ==========================================================

    st.header("Mathematical Foundation")

    st.markdown("""
This project is based on the following concepts:

- Log Returns
- Drift Estimation (μ)
- Volatility Estimation (σ)
- Geometric Brownian Motion
- Monte Carlo Simulation
- Probability Distribution
- Confidence Intervals
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Sharpe Ratio
""")

    st.divider()

    # ==========================================================
    # Project Structure
    # ==========================================================

    st.header("Project Structure")

    st.code(
"""
MonteCarlo-Stock-Risk-Analyzer/

├── app.py
│
├── src/
│   ├── data_loader.py
│   ├── simulator.py
│   ├── risk.py
│   └── visualization.py
│
├── ui/
│   ├── theme.py
│   ├── sidebar.py
│   ├── overview.py
│   ├── risk_analysis.py
│   ├── methodology.py
│   └── about.py
│
├── outputs/
├── requirements.txt
└── README.md
"""
    )

    st.divider()

    # ==========================================================
    # Assumptions
    # ==========================================================

    st.header("Assumptions")

    st.info("""
The current implementation assumes:

• Stock prices follow Geometric Brownian Motion (GBM)

• Daily returns are independent

• Drift and volatility remain constant

• Returns approximately follow a normal distribution

• Markets are frictionless

• No dividends or transaction costs are included
""")

    st.divider()

    # ==========================================================
    # Limitations
    # ==========================================================

    st.header("Current Limitations")

    st.warning("""
The model does not currently account for:

• Volatility clustering

• Heavy-tailed return distributions

• Earnings announcements

• Macroeconomic events

• Interest rate changes

• Regime shifts

• Market crashes

• Portfolio optimization
""")

    st.divider()

    # ==========================================================
    # Disclaimer
    # ==========================================================

    st.header("Disclaimer")

    st.error("""
This application is intended for educational and research
purposes only.

It should not be considered financial advice or used as the
sole basis for investment decisions.
""")

    st.divider()

    # ==========================================================
    # Footer
    # ==========================================================

    st.markdown(
        """
---
**Monte Carlo Stock Risk Analyzer**

Developed as a quantitative finance and risk analytics project
using Python and Streamlit.
"""
    )