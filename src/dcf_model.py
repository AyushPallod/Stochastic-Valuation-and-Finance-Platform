import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression


class DCFValuationEngine:
    """
    Modular DCF Valuation & Stochastic Monte Carlo Simulation Engine.
    Integrates historical data parsing, ARIMA/Linear Regression forecasting,
    and parameter distribution sampling.
    """
    def __init__(self, ticker, forecast_years=10, terminal_growth=0.025, wacc=0.082, tax_rate=0.15):
        self.ticker = ticker.strip().upper()
        self.forecast_years = forecast_years
        self.terminal_growth = terminal_growth
        self.wacc = wacc
        self.tax_rate = tax_rate
        
        self.stock = yf.Ticker(self.ticker)
        self.historical_data = None
        self.shares_outstanding = None
        self.current_cash = None
        self.total_debt = None
        self.current_price = None
        self.currency = "$"
        
    def fetch_historical_financials(self):
        """Downloads and parses historical statements from Yahoo Finance."""
        annual_financials = self.stock.income_stmt
        if annual_financials.empty:
            raise ValueError(f"No historical financials found for ticker: {self.ticker}")
            
        # Extract rows (check case variants)
        revenue_row = None
        operating_income_row = None
        net_income_row = None
        
        for idx in annual_financials.index:
            idx_lower = idx.lower()
            if 'total revenue' in idx_lower or 'revenue' in idx_lower:
                revenue_row = idx
            elif 'operating income' in idx_lower or 'operatingincome' in idx_lower:
                operating_income_row = idx
            elif 'net income' in idx_lower or 'netincome' in idx_lower:
                net_income_row = idx
                
        if revenue_row is None:
            raise ValueError(f"Could not find Total Revenue row in financials for {self.ticker}")
            
        revenue_history = annual_financials.loc[revenue_row] / 1e9
        operating_income = annual_financials.loc[operating_income_row] / 1e9 if operating_income_row else pd.Series(0, index=revenue_history.index)
        net_income = annual_financials.loc[net_income_row] / 1e9 if net_income_row else pd.Series(0, index=revenue_history.index)
        
        self.historical_data = pd.DataFrame({
            'Year': range(len(revenue_history)),
            'Revenue': revenue_history.values[::-1],
            'Operating_Income': operating_income.values[::-1],
            'Net_Income': net_income.values[::-1]
        })
        
        self.historical_data['Revenue_Growth'] = self.historical_data['Revenue'].pct_change() * 100
        self.historical_data['Operating_Margin'] = (self.historical_data['Operating_Income'] / self.historical_data['Revenue']) * 100
        
        # Retrieve Cash and Debt from balance sheet
        balance_sheet = self.stock.balance_sheet
        cash_val = 0.0
        debt_val = 0.0
        
        if not balance_sheet.empty:
            for idx in balance_sheet.index:
                idx_lower = idx.lower()
                if 'cash cash equivalents' in idx_lower or 'cash and cash equivalents' in idx_lower or 'cash' in idx_lower:
                    cash_val = balance_sheet.loc[idx].iloc[0] / 1e9
                    break
            
            for idx in balance_sheet.index:
                idx_lower = idx.lower()
                if 'total debt' in idx_lower:
                    debt_val = balance_sheet.loc[idx].iloc[0] / 1e9
                    break
                    
        self.current_cash = cash_val if cash_val > 0 else 5.0
        self.total_debt = debt_val if debt_val > 0 else 10.0
        
        # Shares and price information
        info = self.stock.info
        self.shares_outstanding = info.get('sharesOutstanding', 7.645e9) / 1e9
        self.current_price = info.get('currentPrice', 430.0)
        
        # Format Currency Symbol
        financial_currency = info.get('financialCurrency', 'USD')
        if financial_currency == 'INR':
            self.currency = '₹'
        elif financial_currency == 'USD':
            self.currency = '$'
        else:
            self.currency = financial_currency + ' '
            
    def run_ml_forecast(self):
        """Runs ARIMA and Linear Regression ensemble for future revenue baseline."""
        if self.historical_data is None:
            self.fetch_historical_financials()
            
        revenue_data = self.historical_data['Revenue'].dropna().values
        years_historical = np.arange(len(revenue_data))
        
        lr_model = LinearRegression()
        lr_model.fit(years_historical.reshape(-1, 1), revenue_data)
        lr_forecast = lr_model.predict(np.arange(len(revenue_data), len(revenue_data) + self.forecast_years).reshape(-1, 1))
        
        try:
            arima_model = ARIMA(revenue_data, order=(1, 1, 0))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.get_forecast(steps=self.forecast_years)
            arima_revenue = arima_forecast.predicted_mean.values
            forecast_revenue = (arima_revenue + lr_forecast) / 2
        except Exception:
            forecast_revenue = lr_forecast
            
        return forecast_revenue

    def calculate_dcf(self, revenue_growth, ebitda_margin, capex_pct, nwc_pct):
        """Performs a standard deterministic 10-year DCF calculation."""
        if self.historical_data is None:
            self.fetch_historical_financials()
            
        base_revenue = self.historical_data['Revenue'].iloc[-1]
        
        years = np.arange(1, self.forecast_years + 1)
        revenues = np.zeros(self.forecast_years)
        ebitda = np.zeros(self.forecast_years)
        fcf = np.zeros(self.forecast_years)
        
        for i in range(self.forecast_years):
            if i == 0:
                revenues[i] = base_revenue * (1 + revenue_growth)
            else:
                revenues[i] = revenues[i-1] * (1 + revenue_growth)
            
            ebitda[i] = revenues[i] * ebitda_margin
            da = revenues[i] * 0.03
            ebit = ebitda[i] - da
            nopat = ebit * (1 - self.tax_rate)
            capex = revenues[i] * capex_pct
            nwc_change = revenues[i] * nwc_pct
            fcf[i] = nopat + da - capex - nwc_change
            
        forecast_df = pd.DataFrame({
            'Year': years,
            'Revenue': revenues,
            'EBITDA': ebitda,
            'FCF': fcf
        })
        
        # Discount Factor calculation
        discount_factors = np.array([(1 / (1 + self.wacc) ** i) for i in range(1, self.forecast_years + 1)])
        pv_fcf = (fcf * discount_factors).sum()
        
        # Terminal Value
        terminal_fcf = fcf[-1] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth)
        pv_terminal = terminal_value * discount_factors[-1]
        
        ev = pv_fcf + pv_terminal
        equity_value = ev - self.total_debt + self.current_cash
        per_share = equity_value / self.shares_outstanding
        
        return {
            'forecast_df': forecast_df,
            'ev': ev,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal,
            'terminal_value': terminal_value,
            'equity_value': equity_value,
            'per_share': per_share,
            'shares': self.shares_outstanding,
            'cash': self.current_cash,
            'debt': self.total_debt
        }

    def run_monte_carlo(self, sims_count, revenue_growth, growth_std, ebitda_margin, ebitda_std,
                        wacc_std, terminal_growth_std, capex_pct, nwc_pct):
        """Runs a stochastic simulation sampling parameters from normal distributions."""
        if self.historical_data is None:
            self.fetch_historical_financials()
            
        base_revenue = self.historical_data['Revenue'].iloc[-1]
        simulated_prices = []
        
        for _ in range(sims_count):
            g_sampled = np.random.normal(revenue_growth, growth_std)
            ebitda_sampled = np.random.normal(ebitda_margin, ebitda_std)
            wacc_sampled = np.random.normal(self.wacc, wacc_std)
            tgr_sampled = np.random.normal(self.terminal_growth, terminal_growth_std)
            
            # Truncation boundaries
            g_sampled = max(-0.15, min(0.35, g_sampled))
            ebitda_sampled = max(0.1, min(0.7, ebitda_sampled))
            wacc_sampled = max(0.04, min(0.2, wacc_sampled))
            tgr_sampled = max(0.005, min(0.05, tgr_sampled))
            
            if wacc_sampled <= tgr_sampled:
                wacc_sampled = tgr_sampled + 0.005
                
            rev = base_revenue
            fcf_list = []
            for y in range(self.forecast_years):
                rev = rev * (1 + g_sampled)
                ebitda_val = rev * ebitda_sampled
                da_val = rev * 0.03
                ebit_val = ebitda_val - da_val
                nopat_val = ebit_val * (1 - self.tax_rate)
                capex_val = rev * capex_pct
                nwc_val = rev * nwc_pct
                fcf_val = nopat_val + da_val - capex_val - nwc_val
                fcf_list.append(fcf_val)
                
            fcf_arr = np.array(fcf_list)
            discount_factors = np.array([(1 / (1 + wacc_sampled) ** k) for k in range(1, self.forecast_years + 1)])
            pv_fcf = (fcf_arr * discount_factors).sum()
            
            terminal_fcf = fcf_arr[-1] * (1 + tgr_sampled)
            terminal_value = terminal_fcf / (wacc_sampled - tgr_sampled)
            pv_terminal = terminal_value * discount_factors[-1]
            
            ev = pv_fcf + pv_terminal
            equity_val = ev - self.total_debt + self.current_cash
            per_share = equity_val / self.shares_outstanding
            simulated_prices.append(per_share)
            
        return np.array(simulated_prices)
