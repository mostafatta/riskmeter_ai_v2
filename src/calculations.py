import pandas as pd
import numpy as np
import os


class RiskCalculator:
    def __init__(self, data_dir=None):
        # === Always resolve path relative to this file ===
        if data_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(base_path, '..', 'data', 'raw')
        else:
            self.data_dir = data_dir

        self.stock_prices = None
        self.market_prices = None
        self.returns = None
        self.market_returns = None
        self.tickers = None

    def load_data(self):
        """Load locally saved CSV data for stocks and market."""
        stocks_path = os.path.join(self.data_dir, "stocks_prices.csv")
        market_path = os.path.join(self.data_dir, "market_prices.csv")

        self.stock_prices = pd.read_csv(
            stocks_path, index_col=0, parse_dates=True
        )
        self.market_prices = pd.read_csv(
            market_path, index_col=0, parse_dates=True
        )
        self.market_prices.columns = ['TASI_Index']
        self.tickers = self.stock_prices.columns.tolist()

        # Align dates
        common_index = self.stock_prices.index.intersection(
            self.market_prices.index
        )
        self.stock_prices = self.stock_prices.loc[common_index]
        self.market_prices = self.market_prices.loc[common_index]

    def calculate_daily_returns(self):
        """
        Calculate logarithmic daily returns.
        R_t = ln(P_t / P_{t-1})
        """
        self.returns = np.log(
            self.stock_prices / self.stock_prices.shift(1)
        ).dropna()

        self.market_returns = np.log(
            self.market_prices / self.market_prices.shift(1)
        ).dropna()

        # Align dates again after dropna
        common_index = self.returns.index.intersection(
            self.market_returns.index
        )
        self.returns = self.returns.loc[common_index]
        self.market_returns = self.market_returns.loc[common_index]

    def get_individual_metrics(self):
        """
        Calculate individual stock volatility and beta.

        Volatility: sigma_i = std(R_i) * sqrt(252)
        Beta:       beta_i  = Cov(R_i, R_m) / Var(R_m)
        """
        # Annualized volatility
        volatility = self.returns.std() * np.sqrt(252)

        # Market variance (daily)
        market_var = self.market_returns['TASI_Index'].var()

        betas = {}
        for ticker in self.tickers:
            cov_matrix = np.cov(
                self.returns[ticker],
                self.market_returns['TASI_Index']
            )
            # Beta = Cov(stock, market) / Var(market)
            betas[ticker] = cov_matrix[0, 1] / market_var

        return volatility, pd.Series(betas)

    def calculate_portfolio_risk(self, weights):
        """
        Step 1 & Step 2 from PDF:
        - Portfolio Volatility = sqrt(w^T * Sigma * w)
        - Portfolio Beta = sum(w_i * beta_i)
        """
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError(
                f"Received {len(weights)} weights, "
                f"expected {len(self.tickers)}."
            )

        # ==========================================
        # 2.1.1 Step 1: Portfolio Volatility
        # sigma_p = sqrt(w^T * COV_annual * w)
        # ==========================================
        cov_matrix_annual = self.returns.cov() * 252
        portfolio_variance = np.dot(
            weights.T, np.dot(cov_matrix_annual, weights)
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        # ==========================================
        # 2.1.2 Step 2: Portfolio Beta
        # beta_p = sum(w_i * beta_i)
        # ==========================================
        individual_vols, individual_betas = self.get_individual_metrics()
        portfolio_beta = np.sum(weights * individual_betas.values)

        return {
            "Portfolio_Volatility_Percentage": round(
                portfolio_volatility * 100, 2
            ),
            "Portfolio_Beta": round(portfolio_beta, 3),
            "Stock_Betas": individual_betas.to_dict(),
            "Stock_Volatilities": individual_vols.to_dict()
        }

    def calculate_sector_metrics(self, sector_tickers):
        """
        Calculate REAL sector volatility and beta
        from the stocks belonging to that sector.

        sigma_sector = std of equal-weighted sector portfolio * sqrt(252)
        beta_sector  = Cov(R_sector, R_market) / Var(R_market)
        """
        valid = [t for t in sector_tickers if t in self.returns.columns]
        if not valid:
            return 0.15, 1.0  # Default fallback

        # Equal-weighted sector return
        sector_returns = self.returns[valid].mean(axis=1)

        # Annualized sector volatility
        sector_vol = sector_returns.std() * np.sqrt(252)

        # Sector beta
        market_var = self.market_returns['TASI_Index'].var()
        cov = np.cov(sector_returns, self.market_returns['TASI_Index'])
        sector_beta = cov[0, 1] / market_var

        return sector_vol, sector_beta