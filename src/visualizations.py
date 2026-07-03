import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.risk import RiskAnalyzer


class MonteCarloVisualizer:
    """
    Interactive Plotly visualizations for Monte Carlo stock price simulations.
    """

    def __init__(self, simulator, currency="USD"):
        self.simulator = simulator
        self.prices = simulator.simulated_prices
        self.final_prices = simulator.get_final_prices()
        if simulator.is_portfolio:
            self.current_price = float(np.sum(simulator.last_price * simulator.weights))
        else:
            self.current_price = float(simulator.last_price)
        self.risk = RiskAnalyzer(simulator)
        from src.utils import get_currency_symbol
        self.currency_symbol = get_currency_symbol(currency)

        # Color palette matching ui/theme.py
        self.theme_colors = {
            "background": "#0E1117",
            "card": "#1B1F27",
            "border": "#30363D",
            "primary": "#F59E0B",
            "success": "#22C55E",
            "danger": "#EF4444",
            "warning": "#FACC15",
            "text": "#F5F5F5",
            "subtext": "#9CA3AF"
        }

    def _apply_dark_theme(self, fig, title, xaxis_title, yaxis_title):
        fig.update_layout(
            title={
                "text": f"<b>{title}</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16, "color": self.theme_colors["primary"], "family": "Inter, sans-serif"}
            },
            xaxis={
                "title": {
                    "text": xaxis_title,
                    "font": {"color": self.theme_colors["text"]}
                },
                "gridcolor": self.theme_colors["border"],
                "zerolinecolor": self.theme_colors["border"],
                "tickfont": {"color": self.theme_colors["subtext"]}
            },
            yaxis={
                "title": {
                    "text": yaxis_title,
                    "font": {"color": self.theme_colors["text"]}
                },
                "gridcolor": self.theme_colors["border"],
                "zerolinecolor": self.theme_colors["border"],
                "tickfont": {"color": self.theme_colors["subtext"]}
            },
            paper_bgcolor=self.theme_colors["background"],
            plot_bgcolor=self.theme_colors["card"],
            legend={
                "font": {"color": self.theme_colors["text"]},
                "bgcolor": self.theme_colors["background"],
                "bordercolor": self.theme_colors["border"],
                "borderwidth": 1
            },
            margin=dict(l=50, r=30, t=60, b=50),
            hovermode="x unified"
        )

    # ==========================================================
    # 1. Monte Carlo Price Paths
    # ==========================================================

    def plot_price_paths(self, max_paths=150):
        fig = go.Figure()

        paths_to_plot = self.prices.iloc[:, :max_paths]
        x = np.arange(len(paths_to_plot))

        # Add paths
        for col in paths_to_plot.columns:
            fig.add_trace(go.Scatter(
                x=x,
                y=paths_to_plot[col],
                mode='lines',
                line=dict(width=0.7, color='rgba(245, 158, 11, 0.06)'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add current price line
        fig.add_trace(go.Scatter(
            x=[0, len(self.prices) - 1],
            y=[self.current_price, self.current_price],
            mode='lines',
            line=dict(color=self.theme_colors["danger"], width=2, dash='dash'),
            name='Current Price'
        ))

        self._apply_dark_theme(fig, "Monte Carlo Price Simulation Paths", "Trading Days", "Stock Price")
        # For price paths, unified hover is annoying because of 150 traces. Let's make it simpler.
        fig.update_layout(hovermode="closest")
        return fig

    # ==========================================================
    # 2. Distribution of Final Prices
    # ==========================================================

    def plot_distribution(self):
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.final_prices,
            nbinsx=50,
            marker_color=self.theme_colors["primary"],
            marker_line=dict(width=0.5, color=self.theme_colors["background"]),
            opacity=0.75,
            name="Final Prices",
            hovertemplate="Price Range: %{x}<br>Count: %{y}<extra></extra>"
        ))

        # Add current price line
        fig.add_shape(type="line",
            x0=self.current_price, y0=0, x1=self.current_price, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["danger"], width=2, dash="dash")
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["danger"], width=2, dash="dash"),
            name="Current Price"
        ))

        # Expected price line
        expected = self.risk.expected_price()
        fig.add_shape(type="line",
            x0=expected, y0=0, x1=expected, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash")
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash"),
            name=f"Expected Price ({self.currency_symbol}{expected:.2f})"
        ))

        self._apply_dark_theme(fig, "Distribution of Simulated Final Prices", "Final Stock Price", "Frequency")
        fig.update_layout(hovermode="closest")
        return fig

    # ==========================================================
    # 3. Fan Chart
    # ==========================================================

    def plot_fan_chart(self):
        percentiles = [5, 25, 50, 75, 95]
        bands = np.percentile(self.prices, percentiles, axis=1)
        x = np.arange(len(self.prices))

        fig = go.Figure()

        # 90% interval
        fig.add_trace(go.Scatter(
            x=x, y=bands[4],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bands[0],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.08)',
            line=dict(width=0),
            name='90% Confidence Interval'
        ))

        # 50% interval
        fig.add_trace(go.Scatter(
            x=x, y=bands[3],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bands[1],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.20)',
            line=dict(width=0),
            name='50% Confidence Interval'
        ))

        # Median
        fig.add_trace(go.Scatter(
            x=x, y=bands[2],
            mode='lines',
            line=dict(color=self.theme_colors["primary"], width=2.5),
            name='Median Path'
        ))

        # Current Price
        fig.add_trace(go.Scatter(
            x=[0, len(self.prices)-1],
            y=[self.current_price, self.current_price],
            mode='lines',
            line=dict(color=self.theme_colors["danger"], width=1.5, dash='dash'),
            name='Current Price'
        ))

        self._apply_dark_theme(fig, "Monte Carlo Price Fan Chart", "Trading Days", "Stock Price")
        fig.update_layout(hovermode="closest")
        return fig

    # ==========================================================
    # 4. Return Distribution
    # ==========================================================

    def plot_return_distribution(self):
        returns = self.risk.returns()
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color=self.theme_colors["primary"],
            marker_line=dict(width=0.5, color=self.theme_colors["background"]),
            opacity=0.75,
            name="Returns",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"
        ))

        # Mean return line
        mean_ret = returns.mean()
        fig.add_shape(type="line",
            x0=mean_ret, y0=0, x1=mean_ret, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash")
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash"),
            name=f"Mean Return ({mean_ret:.2f}%)"
        ))

        # Break-even line
        fig.add_shape(type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["danger"], width=2, dash="dash")
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["danger"], width=2, dash="dash"),
            name="Break-even (0.00%)"
        ))

        self._apply_dark_theme(fig, "Distribution of Simulated Returns", "Return (%)", "Frequency")
        fig.update_layout(hovermode="closest")
        return fig

    # ==========================================================
    # 5. VaR / CVaR Visualization
    # ==========================================================

    def plot_var(self):
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.final_prices,
            nbinsx=50,
            marker_color=self.theme_colors["primary"],
            marker_line=dict(width=0.5, color=self.theme_colors["background"]),
            opacity=0.75,
            name="Final Prices",
            hovertemplate="Price Range: %{x}<br>Count: %{y}<extra></extra>"
        ))

        var_price = self.current_price - self.risk.value_at_risk()
        cvar_price = self.current_price - self.risk.conditional_var()

        # Current price line
        fig.add_shape(type="line",
            x0=self.current_price, y0=0, x1=self.current_price, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash")
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["success"], width=2, dash="dash"),
            name="Current Price"
        ))

        # VaR Line
        fig.add_shape(type="line",
            x0=var_price, y0=0, x1=var_price, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["warning"], width=2)
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["warning"], width=2),
            name=f"VaR 95% ({self.currency_symbol}{var_price:.2f})"
        ))

        # CVaR Line
        fig.add_shape(type="line",
            x0=cvar_price, y0=0, x1=cvar_price, y1=1,
            yref="paper",
            line=dict(color=self.theme_colors["danger"], width=2)
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=self.theme_colors["danger"], width=2),
            name=f"CVaR 95% ({self.currency_symbol}{cvar_price:.2f})"
        ))

        self._apply_dark_theme(fig, "Value at Risk Analysis", "Final Stock Price", "Frequency")
        fig.update_layout(hovermode="closest")
        return fig