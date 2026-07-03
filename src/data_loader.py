import yfinance as yf
import pandas as pd
import numpy as np


class StockDataLoader:
    """
    Downloads historical market data and metadata for single or multiple stock tickers,
    aligning dates for multi-asset portfolio analysis.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
        self.is_portfolio = len(self.tickers) > 1
        self.start_date = start_date
        self.end_date = end_date

        self.stocks = {t: yf.Ticker(t) for t in self.tickers}
        if not self.is_portfolio:
            self.stock = self.stocks[self.tickers[0]]
            
        self.data = None
        self.log_returns = None
        self.company_info = {}
        self.portfolio_info = []

    def download_data(self):
        individual_prices = {}
        self.portfolio_info = []

        for t, ticker_obj in self.stocks.items():
            # Get historical daily prices
            hist = ticker_obj.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            if hist.empty:
                raise ValueError(f"No data found for ticker {t} between {self.start_date} and {self.end_date}.")
            individual_prices[t] = hist['Close']

            # ── Metadata: fast_info first (cloud-friendly, avoids rate-limits)
            # then fall back to full ticker.info, then derive from ticker symbol
            fast = {}
            info = {}
            try:
                fast = ticker_obj.fast_info   # lightweight endpoint — works on cloud
            except Exception:
                pass
            try:
                info = ticker_obj.info or {}
                if not isinstance(info, dict):
                    info = {}
            except Exception:
                pass

            # Derive currency from ticker suffix (.NS → INR, default USD)
            def _guess_currency(tkr: str) -> str:
                if tkr.endswith(".NS") or tkr.endswith(".BO"):
                    return "INR"
                if tkr.endswith(".L"):
                    return "GBp"
                if tkr.endswith(".HK"):
                    return "HKD"
                return "USD"

            # Resolve market cap: fast_info uses market_cap, info uses marketCap
            market_cap = (
                getattr(fast, "market_cap", None)
                or info.get("marketCap", None)
            )

            asset_info = {
                "ticker": t,
                "name": info.get("longName") or info.get("shortName") or t,
                "exchange": (
                    getattr(fast, "exchange", None)
                    or info.get("exchange", "N/A")
                    or "N/A"
                ),
                "sector": info.get("sector", "N/A") or "N/A",
                "industry": info.get("industry", "N/A") or "N/A",
                "currency": (
                    getattr(fast, "currency", None)
                    or info.get("currency", None)
                    or _guess_currency(t)
                ),
                "market_cap": market_cap,
                "country": info.get("country", "N/A") or "N/A",
            }
            self.portfolio_info.append(asset_info)

        # Merge individual Close prices on aligned dates
        prices_df = pd.DataFrame(individual_prices)
        prices_df = prices_df.dropna() # Keep only dates where ALL assets have quotes
        
        if prices_df.empty:
            raise ValueError("No overlapping trading dates found for the specified portfolio assets.")
        
        # Strip timezone offsets to prevent Streamlit line chart rendering bugs
        if prices_df.index.tz is not None:
            prices_df.index = prices_df.index.tz_localize(None)
            
        self.data = prices_df

        # Aggregate company information for display
        if not self.is_portfolio:
            self.company_info = self.portfolio_info[0]
        else:
            total_mcap = sum(x["market_cap"] for x in self.portfolio_info if x["market_cap"] is not None)
            self.company_info = {
                "name": f"Custom Portfolio ({', '.join(self.tickers)})",
                "exchange": "Multi-Exchange",
                "sector": "Multi-Sector",
                "industry": "Multi-Industry",
                "currency": self.portfolio_info[0]["currency"], # Default to first asset currency
                "market_cap": total_mcap if total_mcap > 0 else None,
                "country": "Multi-Country"
            }
        
        return self.data

    def calculate_log_returns(self):
        if self.data is None:
            raise ValueError("Data not downloaded. Please call download_data() first.")

        if not self.is_portfolio:
            self.log_returns = np.log(self.data[self.tickers[0]] / self.data[self.tickers[0]].shift(1)).dropna()
        else:
            self.log_returns = np.log(self.data / self.data.shift(1)).dropna()

        return self.log_returns

    def get_last_price(self):
        if self.data is None:
            raise ValueError("Data not downloaded.")
        if not self.is_portfolio:
            return self.data[self.tickers[0]].iloc[-1]
        return self.data.iloc[-1] # returns pd.Series of last prices

    def format_market_cap(self):
        from src.utils import get_currency_symbol
        symbol = get_currency_symbol(self.company_info.get("currency", "USD"))
        market_cap = self.company_info.get("market_cap")
        if market_cap is None:
            return "N/A"
        if market_cap >= 1_000_000_000_000:
            return f"{symbol}{market_cap/1_000_000_000_000:.2f} T"
        if market_cap >= 1_000_000_000:
            return f"{symbol}{market_cap/1_000_000_000:.2f} B"
        if market_cap >= 1_000_000:
            return f"{symbol}{market_cap/1_000_000:.2f} M"
        return f"{symbol}{market_cap:,.0f}"

    def summary(self):
        print("="*50)
        print(f"Ticker(s)    : {', '.join(self.tickers)}")
        print(f"Portfolio?   : {self.is_portfolio}")
        print(f"Observations : {len(self.data)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        if "stock" in state:
            del state["stock"]
        if "stocks" in state:
            del state["stocks"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stocks = {t: yf.Ticker(t) for t in self.tickers}
        if not self.is_portfolio:
            self.stock = self.stocks[self.tickers[0]]
