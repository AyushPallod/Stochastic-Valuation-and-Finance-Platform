import streamlit as st
from datetime import date


def render_sidebar():
    """
    Render the sidebar and return simulation parameters.
    """

    with st.sidebar:

        st.title("⚙️ Simulation Settings")

        st.markdown("---")

        # --------------------------------------------------
        # Stock Selection
        # --------------------------------------------------

        ticker = st.text_input(
            "Ticker Symbol(s)",
            value="SETL.NS",
            help="For portfolios, enter comma-separated tickers. Example: AAPL, MSFT, GOOG"
        ).upper().strip()

        # Parse tickers and show weight allocator if multi-asset
        tickers_list = [t.strip().upper() for t in ticker.split(",") if t.strip()]
        num_assets = len(tickers_list)
        weights = None

        if num_assets > 1:
            st.markdown("##### Portfolio Weights")
            default_weights = ", ".join([str(round(100.0/num_assets, 1)) for _ in range(num_assets)])
            weights_input = st.text_input(
                "Asset Allocations (%)",
                value=default_weights,
                help="Comma-separated weights corresponding to each asset. Must sum to 100."
            )
            try:
                weights = [float(w.strip()) / 100.0 for w in weights_input.split(",") if w.strip()]
                if len(weights) != num_assets:
                    st.warning(f"Number of weights ({len(weights)}) does not match tickers ({num_assets}). Equal weighting will be used.")
                    weights = None
                elif not np.isclose(sum(weights), 1.0, atol=1e-3):
                    st.warning(f"Weights sum to {sum(weights)*100:.1f}%. Normalizing to 100%.")
                    weights = [w / sum(weights) for w in weights]
            except Exception:
                st.error("Invalid weights format. Equal weighting will be used.")
                weights = None
        else:
            weights = [1.0]

        # --------------------------------------------------
        # Historical Window
        # --------------------------------------------------

        st.subheader("Historical Data")

        start_date = st.date_input(
            "Start Date",
            value=date(2025, 1, 1)
        )

        end_date = st.date_input(
            "End Date",
            value=date.today()
        )

        # --------------------------------------------------
        # Simulation Parameters
        # --------------------------------------------------

        st.subheader("Model Configuration")

        model_display = st.selectbox(
            "Simulation Model",
            options=[
                "Geometric Brownian Motion (GBM)",
                "Historical Bootstrap",
                "Merton Jump Diffusion"
            ],
            index=0
        )

        model_map = {
            "Geometric Brownian Motion (GBM)": "gbm",
            "Historical Bootstrap": "bootstrap",
            "Merton Jump Diffusion": "jump_diffusion"
        }
        model_type = model_map[model_display]

        jump_lambda = 0.1
        jump_mu = -0.05
        jump_sigma = 0.1

        if model_type == "jump_diffusion":
            st.markdown("##### Jump Parameters")
            jump_lambda = st.slider(
                "Jump Frequency (Jumps/Year)",
                min_value=0.05,
                max_value=2.00,
                value=0.10,
                step=0.05
            )
            jump_mu = st.slider(
                "Average Jump Magnitude",
                min_value=-0.50,
                max_value=0.50,
                value=-0.05,
                step=0.01
            )
            jump_sigma = st.slider(
                "Jump Volatility",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01
            )

        st.subheader("Simulation parameters")

        forecast_days = st.slider(
            "Forecast Horizon (Trading Days)",
            min_value=5,
            max_value=252,
            value=252,
            step=1
        )

        simulations = st.select_slider(
            "Monte Carlo Simulations",
            options=[
                1000,
                5000,
                10000,
                25000,
                50000
            ],
            value=10000
        )

        seed = st.number_input(
            "Random Seed",
            value=42,
            step=1
        )

        st.markdown("---")

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if start_date >= end_date:
            st.error("Start date must be before end date.")

        history_days = (end_date - start_date).days

        estimated_trading_days = int(history_days * 252 / 365)

        if estimated_trading_days < 60:

            st.warning(
                "Only a small amount of historical data "
                "is selected. Annualized statistics may "
                "be unstable."
            )

        elif estimated_trading_days < 252:

            st.info(
                "Less than one trading year of historical "
                "data is being used."
            )

        if forecast_days > estimated_trading_days:

            st.warning(
                "Forecast horizon is longer than the "
                "historical window used for estimation."
            )

        st.markdown("---")

        # --------------------------------------------------
        # Simulation Summary
        # --------------------------------------------------

        st.subheader("Summary")

        st.write(f"**Ticker:** {ticker}")

        st.write(f"**Model:** {model_type.upper().replace('_', ' ')}")

        st.write(
            f"**Historical Window:** "
            f"{start_date} → {end_date}"
        )

        st.write(
            f"**Estimated Trading Days:** "
            f"{estimated_trading_days}"
        )

        st.write(
            f"**Forecast Horizon:** "
            f"{forecast_days} Trading Days"
        )

        st.write(
            f"**Simulations:** "
            f"{simulations:,}"
        )

        st.markdown("---")

        run = st.button(
            "▶ Run Simulation",
            use_container_width=True
        )

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_days": forecast_days,
        "simulations": simulations,
        "seed": int(seed),
        "model_type": model_type,
        "jump_lambda": jump_lambda,
        "jump_mu": jump_mu,
        "jump_sigma": jump_sigma,
        "weights": weights,
        "run": run
    }