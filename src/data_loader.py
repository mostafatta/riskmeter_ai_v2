import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


class TadawulDataLoader:
    def __init__(self, tickers=None, data_dir=None):
        if tickers is None:
            self.tickers = [
                # === Banks (القطاع البنكي) ===
                '1120.SR', '1010.SR', '1180.SR', '1080.SR',
                # === Energy & Industry (الطاقة والصناعة) ===
                '2222.SR', '2010.SR', '2310.SR', '2020.SR',
                # === Telecom & Retail (الاتصالات والتجزئة) ===
                '7010.SR', '4190.SR', '4200.SR', '4030.SR',
                # === Cement (الاسمنت) ===
                '3030.SR', '3040.SR', '3050.SR', '3060.SR',
                # === Services & Mining (خدمات وتعدين) ===
                '4003.SR', '4008.SR', '2150.SR', '1211.SR'
            ]
        else:
            self.tickers = tickers

        # ==========================================
        # Sector Mapping: Each stock -> its sector
        # ==========================================
        self.sector_map = {
            '1120.SR': 'Banks', '1010.SR': 'Banks',
            '1180.SR': 'Banks', '1080.SR': 'Banks',
            '2222.SR': 'Energy', '2010.SR': 'Energy',
            '2310.SR': 'Industry', '2020.SR': 'Industry',
            '7010.SR': 'Telecom', '4190.SR': 'Retail',
            '4200.SR': 'Retail', '4030.SR': 'Retail',
            '3030.SR': 'Cement', '3040.SR': 'Cement',
            '3050.SR': 'Cement', '3060.SR': 'Cement',
            '4003.SR': 'Services', '4008.SR': 'Services',
            '2150.SR': 'Mining', '1211.SR': 'Mining'
        }

        self.market_ticker = "^TASI.SR"

        # === Fix path to always resolve correctly ===
        if data_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(base, '..', 'data', 'raw')
        else:
            self.data_dir = data_dir

        today = datetime.today()
        one_year_ago = today - timedelta(days=365)
        self.end_date = today.strftime('%Y-%m-%d')
        self.start_date = one_year_ago.strftime('%Y-%m-%d')

        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_stock_data(self):
        """Download stock prices from Yahoo Finance."""
        print(f"Fetching data for {len(self.tickers)} stocks...")
        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )['Adj Close']

            if isinstance(data, pd.Series):
                data = data.to_frame()

            file_path = os.path.join(self.data_dir, "stocks_prices.csv")
            data.to_csv(file_path)
            print(f"  Stock prices saved to {file_path}")
            return data
        except Exception as e:
            print(f"  Error downloading stocks: {e}")
            return None

    def fetch_market_data(self):
        """Download TASI market index from Yahoo Finance."""
        print(f"Fetching Market Index ({self.market_ticker})...")
        try:
            market_data = yf.download(
                self.market_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )['Adj Close']

            market_data.name = "TASI_Index"
            file_path = os.path.join(self.data_dir, "market_prices.csv")
            market_data.to_csv(file_path)
            print(f"  Market data saved to {file_path}")
            return market_data
        except Exception as e:
            print(f"  Error downloading market data: {e}")
            return None

    def fetch_metadata(self):
        """Fetch Market Cap data for each stock."""
        print("Fetching Market Cap metadata...")
        metadata_list = []
        for t in self.tickers:
            try:
                stock = yf.Ticker(t)
                mkt_cap = stock.info.get('marketCap', 0)
                if mkt_cap > 50_000_000_000:
                    cap_score = 3.0   # Large Cap
                elif mkt_cap > 10_000_000_000:
                    cap_score = 2.0   # Mid Cap
                else:
                    cap_score = 1.0   # Small Cap
            except Exception:
                cap_score = 2.0       # Default Mid Cap

            metadata_list.append({
                "Ticker": t,
                "Market_Cap_Score": cap_score,
                "Sector": self.sector_map.get(t, "Unknown")
            })

        df = pd.DataFrame(metadata_list)
        file_path = os.path.join(self.data_dir, "stocks_metadata.csv")
        df.to_csv(file_path, index=False)
        print(f"  Metadata saved to {file_path}")
        return df