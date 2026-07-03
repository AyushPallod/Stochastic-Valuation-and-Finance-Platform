import numpy as np
import pandas as pd


class RiskAnalyzer:
    """
    Performs statistical and risk analysis on Monte Carlo simulation results.
    Supports both single stock and multi-asset portfolio simulations.
    """

    TRADING_DAYS = 252

    def __init__(self, simulator):
        self.simulator = simulator
        self.final_prices = simulator.get_final_prices()

        if simulator.is_portfolio:
            # Current price of the portfolio is the weighted average of individual asset prices
            self.current_price = float(np.sum(simulator.last_price * simulator.weights))
        else:
            self.current_price = float(simulator.last_price)
            
        self.days = simulator.days

    # ==========================================================
    # Basic Statistics
    # ==========================================================

    def expected_price(self):
        """Mean of simulated final prices."""
        return self.final_prices.mean()

    def median_price(self):
        """Median of simulated final prices."""
        return self.final_prices.median()

    def standard_deviation(self):
        """Standard deviation of simulated final prices."""
        return self.final_prices.std()

    def percentile(self, level):
        """
        Returns any percentile of the simulated prices.
        """
        return np.percentile(self.final_prices, level)

    def confidence_interval(self, confidence_level=95):
        """
        Confidence interval of simulated prices.
        """
        alpha = (100 - confidence_level) / 2
        lower = np.percentile(self.final_prices, alpha)
        upper = np.percentile(self.final_prices, 100 - alpha)
        return lower, upper

    # ==========================================================
    # Probability Metrics
    # ==========================================================

    def probability_above(self, price):
        """
        Probability that final price >= price.
        """
        return (self.final_prices >= price).mean() * 100

    def probability_below(self, price):
        """
        Probability that final price <= price.
        """
        return (self.final_prices <= price).mean() * 100

    # ==========================================================
    # Return Statistics
    # ==========================================================

    def returns(self):
        """
        Simulated percentage returns.
        """
        return (
            (self.final_prices - self.current_price)
            / self.current_price
            * 100
        )

    # ==========================================================
    # Annual Metrics
    # ==========================================================

    def annualized_return(self):
        """
        Annualized return.
        For a portfolio, this calculates the weighted average of the expected drifts.
        """
        if self.simulator.is_portfolio:
            weighted_mu = np.sum(self.simulator.mu * self.simulator.weights)
            return (np.exp(weighted_mu * self.TRADING_DAYS) - 1) * 100
        else:
            return (np.exp(self.simulator.mu * self.TRADING_DAYS) - 1) * 100

    def annualized_volatility(self):
        """
        Annualized volatility.
        For a portfolio, this calculates the portfolio volatility using the weights vector
        and the covariance matrix: sqrt(w^T * Sigma * w) * sqrt(252).
        """
        if self.simulator.is_portfolio:
            w = self.simulator.weights
            cov_matrix = self.simulator.cov.values
            portfolio_variance = w.T @ cov_matrix @ w
            portfolio_vol = np.sqrt(portfolio_variance)
            return (portfolio_vol * np.sqrt(self.TRADING_DAYS)) * 100
        else:
            return (self.simulator.sigma * np.sqrt(self.TRADING_DAYS)) * 100

    def sharpe_ratio(self, risk_free_rate=0.05):
        """
        Annualized Sharpe Ratio.
        """
        annual_return = self.annualized_return() / 100
        annual_vol = self.annualized_volatility() / 100

        if annual_vol == 0:
            return np.nan

        return (annual_return - risk_free_rate) / annual_vol

    # ==========================================================
    # Risk Metrics
    # ==========================================================

    def value_at_risk(self, confidence_level=95):
        """
        Value at Risk (VaR).
        """
        percentile = 100 - confidence_level
        worst_price = np.percentile(
            self.final_prices,
            percentile
        )
        return self.current_price - worst_price

    def conditional_var(self, confidence_level=95):
        """
        Conditional Value at Risk (CVaR).
        """
        percentile = 100 - confidence_level
        threshold = np.percentile(
            self.final_prices,
            percentile
        )
        tail_losses = self.final_prices[
            self.final_prices <= threshold
        ]
        return self.current_price - tail_losses.mean()

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self):
        lower, upper = self.confidence_interval()

        print("\n" + "=" * 65)
        print("              Monte Carlo Risk Analysis")
        print("=" * 65)

        print(f"Current Price              : {self.current_price:.2f}")
        print(f"Forecast Horizon           : {self.days} Trading Days")

        print("\n---------------- BASIC STATISTICS ----------------")

        print(f"Expected Price             : {self.expected_price():.2f}")
        print(f"Median Price               : {self.median_price():.2f}")
        print(f"Standard Deviation         : {self.standard_deviation():.2f}")

        print(f"5th Percentile             : {self.percentile(5):.2f}")
        print(f"95th Percentile            : {self.percentile(95):.2f}")

        print("\n-------------- CONFIDENCE INTERVAL ---------------")

        print(f"95% Confidence Interval    : [{lower:.2f}, {upper:.2f}]")

        print("\n---------------- PROBABILITIES -------------------")

        print(
            f"Price >= Current Price     : "
            f"{self.probability_above(self.current_price):.2f}%"
        )

        print(
            f"Price <= Current Price     : "
            f"{self.probability_below(self.current_price):.2f}%"
        )

        print("\n---------------- ANNUAL METRICS ------------------")

        print(
            f"Annualized Return          : "
            f"{self.annualized_return():.2f}%"
        )

        print(
            f"Annualized Volatility      : "
            f"{self.annualized_volatility():.2f}%"
        )

        print(
            f"Sharpe Ratio               : "
            f"{self.sharpe_ratio():.3f}"
        )

        print("\n----------------- RISK METRICS -------------------")

        print(
            f"VaR (95%)                  : "
            f"{self.value_at_risk():.2f}"
        )

        print(
            f"CVaR (95%)                 : "
            f"{self.conditional_var():.2f}"
        )

        print("=" * 65)