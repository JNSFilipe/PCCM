import yfinance as yf
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time


class StockTickerFetcher:
    def __init__(self):
        self.base_url = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{}"
        self.cboe_url = "https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv"
        
    def get_all_tickers(self):
        tickers = {}
        page = 1
        total_pages = None
        
        tiks = {}
        while total_pages is None or page <= total_pages:
            try:
                response = requests.get(self.base_url.format(page))
                response.raise_for_status()
                data = response.json()
                
                if total_pages is None:
                    total_pages = data['pages']

                for t in data["results"]:
                    tickers[t["ticker"]] = t
                
                break
                page += 1
                time.sleep(1)
                
            except requests.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                break
                
        return tickers

    def get_optionable_tickers(self):
        try:
            response = requests.get(self.cboe_url)
            response.raise_for_status()
            cboe_tickers = set(line.split(',')[1].replace('"', '') for line in response.text.splitlines()[1:])
            tickers = self.get_all_tickers()
            return {k: tickers[k] for k in tickers.keys() if k in cboe_tickers}
        except requests.RequestException as e:
            print(f"Error fetching CBOE data: {e}")
            return []


class QualityScore:
    def __init__(self):
        pass
        
    def get_z_score(self, series):
        """Convert a series to z-scores"""
        return (series - series.mean()) / series.std()
    
    def get_financial_data(self, ticker, max_price):
        """Get key financial metrics from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            info = stock.info
            income = stock.income_stmt
            balance = stock.balance_sheet
            cash = stock.cash_flow

            if info['open'] > max_price:
                return pd.Series()
            
            # Get price data for market cap and returns
            hist = stock.history(period="2y")
            
            metrics = {}
            
            if info: # info not empty
                metrics['roa'] = info['returnOnAssets']
                metrics['roe'] = info['returnOnEquity']
                metrics['beta'] = info['beta']
                metrics['gross_margin'] = info['grossMargins']
                metrics['current_ratio'] = info['currentRatio']
                # metrics['earnings_growth'] = info['earningsGrowth']
                metrics['revenue_growth'] = info['revenueGrowth']

            # Profitability metrics
            if not income.empty:
                # metrics['gross_margin'] = income.loc['Gross Profit'].iloc[0] / income.loc['Total Revenue'].iloc[0]
                # metrics['roa'] = income.loc['Net Income'].iloc[0] / balance.loc['Total Assets'].iloc[0]
                # metrics['roe'] = income.loc['Net Income'].iloc[0] / balance.loc['Stockholders Equity'].iloc[0]
                
                # Operating cash flow ratio
                metrics['cf_ratio'] = cash.loc['Operating Cash Flow'].iloc[0] / income.loc['Total Revenue'].iloc[0]
                
            # Growth metrics (1-year)
            if len(income.columns) >= 2:
                #     metrics['revenue_growth'] = (income.loc['Total Revenue'].iloc[0] / 
                #                               income.loc['Total Revenue'].iloc[1]) - 1
                metrics['earnings_growth'] = (income.loc['Net Income'].iloc[0] / 
                                              income.loc['Net Income'].iloc[1]) - 1
                
            # Safety metrics
            if not balance.empty:
                metrics['leverage'] = balance.loc['Total Assets'].iloc[0] / balance.loc['Stockholders Equity'].iloc[0]
                # metrics['current_ratio'] = (balance.loc['Current Assets'].iloc[0] / 
                #                          balance.loc['Current Liabilities'].iloc[0])
            
            # Volatility and beta
            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Get market returns for beta
                # market = yf.download(self.market_ticker, start=hist.index[0], end=hist.index[-1])['Adj Close']
                # market_returns = market.pct_change().dropna()
                
                # # Calculate beta
                # common_idx = returns.index.intersection(market_returns.index)
                # if len(common_idx) > 0:
                #     stock_returns = returns[common_idx]
                #     market_returns = market_returns[common_idx]
                #     metrics['beta'] = np.cov(stock_returns, market_returns)[0,1] / np.var(market_returns)
            
            return pd.Series(metrics)
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            return pd.Series()
    
    def compute_quality_score(self, metrics):
        """Compute overall quality score from metrics"""
        if metrics.empty:
            return np.nan
        
        # Profitability score
        profitability = self.get_z_score(pd.Series({
            'gross_margin': metrics.get('gross_margin', np.nan),
            'roa': metrics.get('roa', np.nan),
            'roe': metrics.get('roe', np.nan),
            'cf_ratio': metrics.get('cf_ratio', np.nan)
        })).mean()
        
        # Growth score
        growth = self.get_z_score(pd.Series({
            'revenue_growth': metrics.get('revenue_growth', np.nan),
            'earnings_growth': metrics.get('earnings_growth', np.nan)
        })).mean()
        
        # Safety score (negative of risk metrics)
        safety = self.get_z_score(pd.Series({
            'leverage': -metrics.get('leverage', np.nan),  # Negative because lower leverage is safer
            'beta': -metrics.get('beta', np.nan),         # Negative because lower beta is safer
            'volatility': -metrics.get('volatility', np.nan),  # Negative because lower volatility is safer
            'current_ratio': metrics.get('current_ratio', np.nan)
        })).mean()
        
        # Overall quality score
        return (profitability + growth + safety) / 3


def main():
    MAX_PRICE = 50

    # Initialize Ticker fetcher
    fetcher = StockTickerFetcher()
    tickers = fetcher.get_optionable_tickers()
    import pudb; pu.db
    print(tickers)

    # Initialize quality scorer
    scorer = QualityScore()
    
    # Store results
    results = {}
    metrics_data = {}
    
    # Process each ticker
    print("Processing stocks...")
    for i, ticker in enumerate(tickers.keys()):
        print(f"Processing {ticker} ({i+1}/{len(tickers)})")
        
        # Get metrics
        metrics = scorer.get_financial_data(ticker, max_price=MAX_PRICE)
        if not metrics.empty:
            metrics_data[ticker] = metrics
            results[ticker] = scorer.compute_quality_score(metrics)
        
        # Pause to avoid rate limiting
        time.sleep(1)
    
    # Convert to DataFrame
    scores_df = pd.DataFrame({
        'quality_score': results
    }).sort_values('quality_score', ascending=False)
    
    # Add metrics for analysis
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
    final_df = pd.concat([scores_df, metrics_df], axis=1)
    
    # Display results
    print("\nTop 10 Quality Stocks:")
    print(final_df.head(10))
    print("\nBottom 10 Quality Stocks:")
    print(final_df.tail(10))
    
    return final_df

if __name__ == "__main__":
    df = main()
