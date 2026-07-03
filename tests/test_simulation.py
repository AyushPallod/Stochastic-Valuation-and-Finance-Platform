import pytest
import numpy as np
import pandas as pd
from src.data_loader import StockDataLoader
from src.simulator import MonteCarloSimulator
from src.risk import RiskAnalyzer


@pytest.fixture
def mock_market_data():
    """Generates synthetic stock price returns for testing."""
    dates = pd.date_range(start="2025-01-01", periods=100)
    # Generate two correlated price series
    np.random.seed(42)
    ret1 = np.random.normal(0.0005, 0.015, size=100)
    ret2 = ret1 * 0.5 + np.random.normal(0.0002, 0.01, size=100) # correlated
    
    prices1 = 100 * np.exp(np.cumsum(ret1))
    prices2 = 150 * np.exp(np.cumsum(ret2))
    
    df = pd.DataFrame({
        "AAPL": prices1,
        "MSFT": prices2
    }, index=dates)
    
    log_returns = np.log(df / df.shift(1)).dropna()
    return df, log_returns


def test_data_loader_initialization():
    """Tests StockDataLoader setup for single and multi-asset tickers."""
    loader_single = StockDataLoader("AAPL", "2025-01-01", "2025-06-01")
    assert loader_single.is_portfolio is False
    assert loader_single.tickers == ["AAPL"]

    loader_multi = StockDataLoader("AAPL, MSFT, GOOG", "2025-01-01", "2025-06-01")
    assert loader_multi.is_portfolio is True
    assert loader_multi.tickers == ["AAPL", "MSFT", "GOOG"]


def test_single_asset_gbm_simulation(mock_market_data):
    """Tests single asset GBM simulation dimensions and output properties."""
    df, log_returns = mock_market_data
    last_price = df["AAPL"].iloc[-1]
    single_returns = log_returns["AAPL"]

    simulator = MonteCarloSimulator(
        last_price=last_price,
        log_returns=single_returns,
        days=10,
        simulations=500,
        seed=101,
        model_type="gbm"
    )
    
    sim_prices = simulator.simulate(use_numba=False)
    assert sim_prices.shape == (11, 500) # days + 1, simulations
    assert not sim_prices.isnull().values.any()
    assert np.allclose(sim_prices.iloc[0], last_price)


def test_portfolio_gbm_simulation(mock_market_data):
    """Tests portfolio correlated GBM simulation with Cholesky decomposition."""
    df, log_returns = mock_market_data
    last_prices = df.iloc[-1]
    
    simulator = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=15,
        simulations=300,
        seed=42,
        model_type="gbm",
        weights=[0.6, 0.4]
    )
    
    sim_prices = simulator.simulate(use_numba=False)
    assert sim_prices.shape == (16, 300) # days + 1, simulations
    
    # Starting price must be weighted sum of last prices
    expected_start = 0.6 * last_prices["AAPL"] + 0.4 * last_prices["MSFT"]
    assert np.allclose(sim_prices.iloc[0], expected_start)


def test_portfolio_bootstrap_simulation(mock_market_data):
    """Tests aligned historical bootstrap for portfolio simulations."""
    df, log_returns = mock_market_data
    last_prices = df.iloc[-1]
    
    simulator = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=5,
        simulations=100,
        seed=123,
        model_type="bootstrap",
        weights=[0.5, 0.5]
    )
    
    sim_prices = simulator.simulate(use_numba=False)
    assert sim_prices.shape == (6, 100)
    assert not sim_prices.isnull().values.any()


def test_merton_jump_diffusion_simulation(mock_market_data):
    """Tests Merton Jump Diffusion path generation."""
    df, log_returns = mock_market_data
    last_prices = df.iloc[-1]
    
    simulator = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=10,
        simulations=100,
        seed=99,
        model_type="jump_diffusion",
        jump_lambda=0.5,
        jump_mu=-0.02,
        jump_sigma=0.05
    )
    
    sim_prices = simulator.simulate(use_numba=False)
    assert sim_prices.shape == (11, 100)
    assert not sim_prices.isnull().values.any()


def test_numba_acceleration_equivalence(mock_market_data):
    """Tests that Numba JIT accelerated paths match numpy math structures."""
    df, log_returns = mock_market_data
    last_prices = df.iloc[-1]
    
    sim_numpy = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=10,
        simulations=100,
        seed=42,
        model_type="gbm"
    )
    prices_numpy = sim_numpy.simulate(use_numba=False)
    
    sim_numba = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=10,
        simulations=100,
        seed=42, # identical seed
        model_type="gbm"
    )
    prices_numba = sim_numba.simulate(use_numba=True)
    
    # Assert statistical outputs are close (due to JIT random number sequences)
    assert np.abs(prices_numpy.iloc[-1].mean() - prices_numba.iloc[-1].mean()) < 5.0
    assert prices_numpy.shape == prices_numba.shape


def test_risk_analyzer_metrics(mock_market_data):
    """Tests VaR, CVaR, Sharpe ratio and portfolio risk statistics calculations."""
    df, log_returns = mock_market_data
    last_prices = df.iloc[-1]
    
    simulator = MonteCarloSimulator(
        last_price=last_prices,
        log_returns=log_returns,
        days=10,
        simulations=1000,
        seed=42,
        model_type="gbm",
        weights=[0.5, 0.5]
    )
    simulator.simulate()
    
    analyzer = RiskAnalyzer(simulator)
    
    # Check current price matches weighted start
    weighted_start = 0.5 * last_prices["AAPL"] + 0.5 * last_prices["MSFT"]
    assert np.allclose(analyzer.current_price, weighted_start)
    
    # Check returns dimension
    returns = analyzer.returns()
    assert len(returns) == 1000
    
    # Check VaR is positive for standard simulations
    var_95 = analyzer.value_at_risk(95)
    cvar_95 = analyzer.conditional_var(95)
    assert cvar_95 >= var_95
    
    # Annualized portfolio statistics
    ann_return = analyzer.annualized_return()
    ann_vol = analyzer.annualized_volatility()
    sharpe = analyzer.sharpe_ratio(risk_free_rate=0.03)
    
    assert not np.isnan(ann_return)
    assert ann_vol > 0
    assert not np.isnan(sharpe)
