# app.py
import streamlit as st
import plotly.graph_objects as go

from src.data_loader import StockDataLoader
from src.simulator import MonteCarloSimulator
from src.risk import RiskAnalyzer
from src.visualizations import MonteCarloVisualizer

from ui.theme import apply_theme
from ui.sidebar import render_sidebar
from ui.overview import render_overview
from ui.risk_analysis import render_risk_analysis
from ui.methodology import render_methodology
from ui.about import render_about
from ui.dcf_valuation import render_dcf_valuation

st.set_page_config(
    page_title="Monte Carlo Stock Risk & Valuation Hub",
    page_icon="📈",
    layout="wide"
)

apply_theme()

# Top level App Suite Selector
app_mode = st.sidebar.selectbox(
    "🎯 Select Suite",
    ["Portfolio Risk Simulator", "stochastic DCF Valuation"]
)

if app_mode == "Portfolio Risk Simulator":
    @st.cache_data(show_spinner=False)
    def load_data(ticker, start, end):
        loader = StockDataLoader(ticker, str(start), str(end))
        loader.download_data()
        loader.calculate_log_returns()
        return loader

    params = render_sidebar()

    if params["run"]:
        try:
            with st.spinner("Running simulation..."):
                loader = load_data(params["ticker"], params["start_date"], params["end_date"])

                simulator = MonteCarloSimulator(
                    last_price=loader.get_last_price(),
                    log_returns=loader.log_returns,
                    days=params["forecast_days"],
                    simulations=params["simulations"],
                    seed=params["seed"],
                    model_type=params["model_type"],
                    jump_lambda=params["jump_lambda"],
                    jump_mu=params["jump_mu"],
                    jump_sigma=params["jump_sigma"],
                    weights=params["weights"]
                )
                simulator.simulate()

                st.session_state.loader = loader
                st.session_state.simulator = simulator
                st.session_state.risk = RiskAnalyzer(simulator)
                st.session_state.visualizer = MonteCarloVisualizer(simulator, currency=loader.company_info.get("currency", "USD"))

        except Exception as e:
            st.error(str(e))
            st.stop()

    if "loader" not in st.session_state:
        st.title("📈 Monte Carlo Stock Risk Analyzer")
        st.info("Configure the sidebar and click Run Simulation.")
        st.stop()

    loader = st.session_state.loader
    simulator = st.session_state.simulator
    risk = st.session_state.risk
    visualizer = st.session_state.visualizer

    info = loader.company_info

    st.title(info["name"])

    # Custom HTML cards for company details to prevent truncation and provide premium layout
    info_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px;">
        <div style="background-color: #1B1F27; border: 1px solid #30363D; border-radius: 8px; padding: 12px 16px; flex: 1; min-width: 120px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            <div style="color: #9CA3AF; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Exchange</div>
            <div style="color: #F5F5F5; font-size: 1.25rem; font-weight: 700;">{info['exchange']}</div>
        </div>
        <div style="background-color: #1B1F27; border: 1px solid #30363D; border-radius: 8px; padding: 12px 16px; flex: 1.5; min-width: 160px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            <div style="color: #9CA3AF; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Sector</div>
            <div style="color: #F5F5F5; font-size: 1.25rem; font-weight: 700;">{info['sector']}</div>
        </div>
        <div style="background-color: #1B1F27; border: 1px solid #30363D; border-radius: 8px; padding: 12px 16px; flex: 2.5; min-width: 240px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            <div style="color: #9CA3AF; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Industry</div>
            <div style="color: #F5F5F5; font-size: 1.25rem; font-weight: 700; line-height: 1.25;">{info['industry']}</div>
        </div>
        <div style="background-color: #1B1F27; border: 1px solid #30363D; border-radius: 8px; padding: 12px 16px; flex: 1.5; min-width: 160px; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
            <div style="color: #9CA3AF; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Market Cap</div>
            <div style="color: #F59E0B; font-size: 1.25rem; font-weight: 700;">{loader.format_market_cap()}</div>
        </div>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

    st.caption(f"{info['country']} • {info['currency']}")

    st.subheader("Historical Closing Price")
    if loader.is_portfolio:
        hist_series = (loader.data * simulator.weights).sum(axis=1)
        label = f"Weighted Portfolio ({', '.join(loader.tickers)})"
    else:
        hist_series = loader.data[loader.tickers[0]]
        label = loader.tickers[0]

    symbol = loader.company_info.get("currency", "USD")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_series.index,
        y=hist_series.values,
        mode="lines",
        name=label,
        line=dict(color="#5c9eff", width=2)
    ))
    fig_hist.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date",
        yaxis_title=f"Price ({symbol})",
        height=280,
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    overview, risk_tab, methodology, about = st.tabs(
        ["📊 Overview", "📉 Risk Analysis", "📚 Methodology", "ℹ️ About"]
    )

    with overview:
        render_overview(loader, simulator, risk, visualizer)

    with risk_tab:
        render_risk_analysis(risk, simulator, loader)

    with methodology:
        render_methodology(loader, simulator, risk)

    with about:
        render_about()

else:
    render_dcf_valuation()
