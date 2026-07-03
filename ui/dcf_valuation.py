import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.dcf_model import DCFValuationEngine


def render_dcf_valuation():
    """Renders the modular stochastic DCF Valuation Application View."""
    
    st.header("📊 ML-Powered stochastic DCF Valuation")
    st.markdown("""
    This module evaluates the intrinsic value of a company using a **10-Year Discounted Cash Flow (DCF)** model.
    It combines **ARIMA/Linear Regression ensemble forecasting** for baseline revenues with a **stochastic Monte Carlo Simulation** 
    to calculate valuation probability ranges under growth and cost uncertainty.
    """)
    
    # -------------------------------------------------------------
    # Sidebar parameter injections (using main sidebar placeholder)
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ DCF Valuation Parameters")
    
    dcf_ticker = st.sidebar.text_input("DCF Asset Ticker", value="AAPL").strip().upper()
    
    # Sliders for traditional parameters
    wacc = st.sidebar.slider(
        "WACC (%)",
        min_value=5.0,
        max_value=15.0,
        value=8.5,
        step=0.1,
        help="Weighted Average Cost of Capital (Discount Rate)"
    ) / 100
    
    terminal_growth = st.sidebar.slider(
        "Terminal Growth Rate (%)",
        min_value=0.5,
        max_value=4.0,
        value=2.5,
        step=0.1,
        help="Perpetual growth rate of Cash Flows after year 10"
    ) / 100
    
    tax_rate = st.sidebar.slider(
        "Tax Rate (%)",
        min_value=10.0,
        max_value=30.0,
        value=15.0,
        step=0.5
    ) / 100
    
    # Revenue Growth
    st.sidebar.subheader("Future Cash Assumptions")
    growth_scenario = st.sidebar.selectbox(
        "Scenario Baseline",
        ["Ensemble ML Forecast", "Base Growth (8%)", "Bull Growth (12%)", "Bear Growth (4%)"]
    )
    
    if growth_scenario == "Ensemble ML Forecast":
        rev_growth = 0.08  # Default baseline for initialization
    elif "Base" in growth_scenario:
        rev_growth = 0.08
    elif "Bull" in growth_scenario:
        rev_growth = 0.12
    else:
        rev_growth = 0.04
        
    ebitda_margin = st.sidebar.slider("EBITDA Margin (%)", 15, 60, 35, 1) / 100
    capex_pct = st.sidebar.slider("CapEx (% of Revenue)", 1.0, 10.0, 3.0, 0.5) / 100
    nwc_pct = st.sidebar.slider("NWC Change (% of Revenue Growth)", 0.0, 5.0, 2.0, 0.5) / 100

    # Initialize Engine
    engine = DCFValuationEngine(
        ticker=dcf_ticker,
        forecast_years=10,
        terminal_growth=terminal_growth,
        wacc=wacc,
        tax_rate=tax_rate
    )
    
    with st.spinner("Downloading financial statements and historical fundamentals..."):
        try:
            engine.fetch_historical_financials()
        except Exception as e:
            st.error(f"Failed to fetch data for ticker '{dcf_ticker}': {e}. Please ensure it is a valid Yahoo Finance ticker symbol.")
            return

    # If Ensemble ML option is selected, run regression and ARIMA to get baseline growth
    if growth_scenario == "Ensemble ML Forecast":
        with st.spinner("Running ARIMA/Regression ensemble forecast..."):
            try:
                ml_forecast = engine.run_ml_forecast()
                # Compute implied growth rate from Year 1 to Year 10
                implied_growth = (ml_forecast[-1] / engine.historical_data['Revenue'].iloc[-1]) ** (1 / 10) - 1
                rev_growth = max(0.01, min(0.25, implied_growth))
                st.info(f"🔮 Implied ML Ensemble Growth Baseline: **{rev_growth:.2%}**")
            except Exception:
                st.warning("Could not execute ML forecast - falling back to 8% base growth.")
                rev_growth = 0.08
                
    # Deterministic DCF Calc
    dcf = engine.calculate_dcf(rev_growth, ebitda_margin, capex_pct, nwc_pct)
    currency = engine.currency
    
    # -------------------------------------------------------------
    # TABS LAYOUT
    # -------------------------------------------------------------
    tab1, tab2 = st.tabs(["Traditional Valuation & Sensitivity", "🎲 stochastic Monte Carlo Analysis"])
    
    with tab1:
        # Key Metrics Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Intrinsic Value (Per Share)",
                f"{currency}{dcf['per_share']:.2f}",
                delta=f"{((dcf['per_share'] / engine.current_price - 1) * 100):+.1f}% vs Market",
                delta_color="normal" if dcf['per_share'] > engine.current_price else "inverse"
            )
        with col2:
            st.metric("Current Market Price", f"{currency}{engine.current_price:.2f}")
        with col3:
            upside = (dcf['per_share'] / engine.current_price - 1) * 100
            st.metric(
                "Implied Upside",
                f"{upside:+.2f}%",
                delta_color="normal" if upside > 0 else "inverse"
            )
            
        st.divider()
        
        # Charts Row
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("Revenue & Free Cash Flow Forecast")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dcf['forecast_df']['Year'],
                y=dcf['forecast_df']['Revenue'],
                name='Revenue',
                marker_color='#5c9eff',
                opacity=0.7
            ))
            fig.add_trace(go.Scatter(
                x=dcf['forecast_df']['Year'],
                y=dcf['forecast_df']['FCF'],
                name='Free Cash Flow',
                mode='lines+markers',
                line=dict(color='#0952c4', width=3),
                yaxis='y2'
            ))
            fig.update_layout(
                xaxis=dict(title='Forecast Year'),
                yaxis=dict(title='Revenue (Billions)'),
                yaxis2=dict(title='FCF (Billions)', overlaying='y', side='right'),
                hovermode='x unified',
                height=350,
                legend=dict(orientation="h", y=1.1, x=0.3),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_c2:
            st.subheader("Valuation Components")
            fig = go.Figure(data=[
                go.Pie(
                    labels=['PV of Cash Flows (Projection)', 'PV of Terminal Value (Perpetuity)'],
                    values=[dcf['pv_fcf'], dcf['pv_terminal']],
                    hole=0.3,
                    marker=dict(colors=['#5c9eff', '#0952c4'])
                )
            ])
            fig.update_traces(texttemplate='%{percent}<br>%{value:.1f}B')
            fig.update_layout(
                height=350,
                legend=dict(orientation="h", y=-0.1),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()
        
        # Sensitivity Matrix Heatmap
        st.subheader("Sensitivity Analysis: Value per Share (WACC vs Terminal Growth)")
        st.markdown("Assess how changes in the discount rate (WACC) and perpetual growth rate shift the valuation per share.")
        
        wacc_range = np.arange(wacc - 0.02, wacc + 0.025, 0.005)
        tgr_range = np.arange(terminal_growth - 0.01, terminal_growth + 0.015, 0.005)
        sens_matrix = np.zeros((len(wacc_range), len(tgr_range)))
        
        for i, w_val in enumerate(wacc_range):
            for j, t_val in enumerate(tgr_range):
                if w_val > t_val:
                    dfs = np.array([(1 / (1 + w_val) ** k) for k in range(1, 11)])
                    pv_fcf_sens = (dcf['forecast_df']['FCF'].values * dfs).sum()
                    t_fcf = dcf['forecast_df']['FCF'].iloc[-1] * (1 + t_val)
                    t_val_sens = t_fcf / (w_val - t_val)
                    pv_t = t_val_sens * dfs[-1]
                    per_share_sens = (pv_fcf_sens + pv_t - dcf['debt'] + dcf['cash']) / dcf['shares']
                    sens_matrix[i, j] = per_share_sens
                else:
                    sens_matrix[i, j] = np.nan
                    
        fig = go.Figure(data=go.Heatmap(
            z=sens_matrix,
            x=[f"{t:.1%}" for t in tgr_range],
            y=[f"{w:.1%}" for w in wacc_range],
            colorscale='RdYlGn',
            text=np.round(sens_matrix, 2),
            texttemplate=currency + '%{text:.2f}',
            colorbar=dict(title=f"Price ({currency})")
        ))
        fig.update_layout(
            xaxis_title='Terminal Growth Rate',
            yaxis_title='WACC',
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecast Details Table")
        st.dataframe(
            dcf['forecast_df'].round(2),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Year': st.column_config.NumberColumn(format='%d'),
                'Revenue': st.column_config.NumberColumn(format=f"{currency}%.2f B"),
                'EBITDA': st.column_config.NumberColumn(format=f"{currency}%.2f B"),
                'FCF': st.column_config.NumberColumn(format=f"{currency}%.2f B"),
            }
        )

    with tab2:
        st.subheader("🎲 stochastic Valuation Simulation Settings")
        st.markdown("""
        Specify the uncertainty parameters for the Monte Carlo simulation. 
        Each variable is modeled as a normal distribution around your baseline assumptions.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            sims_count = st.slider("Simulation Trials", 500, 5000, 2000, 500)
            growth_std = st.slider("Revenue Growth Std Dev (%)", 0.5, 4.0, 1.5, 0.5) / 100
            ebitda_std = st.slider("EBITDA Margin Std Dev (%)", 1.0, 8.0, 2.5, 0.5) / 100
        with col2:
            wacc_std = st.slider("WACC Std Dev (%)", 0.2, 2.0, 0.8, 0.2) / 100
            terminal_growth_std = st.slider("Terminal Growth Std Dev (%)", 0.1, 1.5, 0.4, 0.1) / 100
            
        if st.button("🚀 Run stochastic Valuation Simulation", use_container_width=True):
            with st.spinner("Executing stochastic simulation runs..."):
                sim_prices = engine.run_monte_carlo(
                    sims_count=sims_count,
                    revenue_growth=rev_growth,
                    growth_std=growth_std,
                    ebitda_margin=ebitda_margin,
                    ebitda_std=ebitda_std,
                    wacc_std=wacc_std,
                    terminal_growth_std=terminal_growth_std,
                    capex_pct=capex_pct,
                    nwc_pct=nwc_pct
                )
                
                # Calculations
                expected_val = np.mean(sim_prices)
                median_val = np.median(sim_prices)
                p10 = np.percentile(sim_prices, 10)
                p90 = np.percentile(sim_prices, 90)
                prob_undervalued = np.mean(sim_prices >= engine.current_price) * 100
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Expected Stock Value", f"{currency}{expected_val:.2f}",
                              delta=f"{((expected_val / engine.current_price - 1) * 100):+.1f}% vs Market")
                with col_m2:
                    st.metric("Median Stock Value", f"{currency}{median_val:.2f}")
                with col_m3:
                    st.metric("90% Confidence Bounds", f"{currency}{p10:.2f} - {currency}{p90:.2f}")
                with col_m4:
                    st.metric("Prob. of Undervaluation", f"{prob_undervalued:.1f}%")
                    
                # Distribution Chart
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=sim_prices,
                    nbinsx=45,
                    marker_color='#2fa839',
                    opacity=0.75,
                    hovertemplate='Price: ' + currency + '%{x:.2f}<br>Trials: %{y}'
                ))
                
                # Markers
                fig.add_vline(x=engine.current_price, line_width=3, line_dash="dash", line_color="red",
                              annotation_text="Market Price", annotation_position="top right")
                fig.add_vline(x=expected_val, line_width=3, line_color="blue",
                              annotation_text="Expected Intrinsic Value", annotation_position="top left")
                
                fig.update_layout(
                    title="Intrinsic Value Distribution Density Chart",
                    xaxis_title=f"Share Intrinsic Value ({currency})",
                    yaxis_title="Simulation Density Count",
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Percentile detailed table
                st.subheader("📋 Quantile Percentile Valuation Breakdown")
                percentiles = [5, 10, 25, 50, 75, 90, 95]
                vals = np.percentile(sim_prices, percentiles)
                
                pct_df = pd.DataFrame({
                    'Percentile': [f"{p}th Percentile" for p in percentiles],
                    'Simulated Intrinsic Price': [f"{currency}{v:.2f}" for v in vals],
                    'Premium / (Discount) vs Market': [f"{((v / engine.current_price - 1) * 100):+.2f}%" for v in vals]
                })
                
                st.dataframe(pct_df, use_container_width=True, hide_index=True)
