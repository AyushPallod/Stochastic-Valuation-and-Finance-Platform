import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.dcf_model import DCFValuationEngine


@patch('src.dcf_model.yf.Ticker')
def test_dcf_valuation_engine(mock_ticker):
    """Tests the DCF Engine data parsing, deterministic DCF, and Monte Carlo trials."""
    
    # Set up simulated financials
    mock_instance = MagicMock()
    mock_ticker.return_value = mock_instance
    
    dates = ['2025-12-31', '2024-12-31', '2023-12-31', '2022-12-31']
    financials_data = {
        'Total Revenue': [400e9, 360e9, 320e9, 280e9],
        'Operating Income': [160e9, 140e9, 120e9, 100e9],
        'Net Income': [120e9, 100e9, 85e9, 70e9]
    }
    mock_instance.income_stmt = pd.DataFrame(
        financials_data, 
        index=dates
    ).T
    
    balance_sheet_data = {
        'Cash Cash Equivalents And Short Term Investments': [50e9, 45e9, 40e9, 35e9],
        'Total Debt': [80e9, 75e9, 70e9, 65e9]
    }
    mock_instance.balance_sheet = pd.DataFrame(
        balance_sheet_data, 
        index=dates
    ).T
    
    mock_instance.info = {
        'sharesOutstanding': 5e9,
        'currentPrice': 150.0,
        'financialCurrency': 'USD'
    }
    
    # Init engine
    engine = DCFValuationEngine(
        ticker="AAPL", 
        forecast_years=5, 
        terminal_growth=0.02, 
        wacc=0.08, 
        tax_rate=0.20
    )
    engine.fetch_historical_financials()
    
    # Check parsing
    assert engine.shares_outstanding == 5.0
    assert engine.current_price == 150.0
    assert engine.current_cash == 50.0
    assert engine.total_debt == 80.0
    assert engine.currency == '$'
    assert len(engine.historical_data) == 4
    
    # Check ARIMA/Regression Ensemble runs
    ml_forecast = engine.run_ml_forecast()
    assert len(ml_forecast) == 5
    
    # Check traditional DCF calculation
    dcf = engine.calculate_dcf(
        revenue_growth=0.08, 
        ebitda_margin=0.40, 
        capex_pct=0.03, 
        nwc_pct=0.02
    )
    assert 'per_share' in dcf
    assert 'ev' in dcf
    assert dcf['forecast_df'].shape == (5, 4) # 5 years, columns: Year, Revenue, EBITDA, FCF
    
    # Check stochastic Monte Carlo
    sims = engine.run_monte_carlo(
        sims_count=50,
        revenue_growth=0.08,
        growth_std=0.02,
        ebitda_margin=0.40,
        ebitda_std=0.03,
        wacc_std=0.01,
        terminal_growth_std=0.005,
        capex_pct=0.03,
        nwc_pct=0.02
    )
    assert len(sims) == 50
    assert not np.isnan(sims).any()
