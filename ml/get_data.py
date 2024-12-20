import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime  # , timedelta
import os
from tqdm import tqdm


def load_tickers(filename):
    """Load tickers from a file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f]


def download_stock_data(ticker, start_date, end_date):
    """Download OHLCV data for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def normalize_with_rolling_mean(df, window=52):
    """Normalize OHLCV data by dividing by 52-week rolling mean."""
    # Convert window from weeks to business days
    window_days = window * 5

    # Calculate weekly rolling mean for each column
    cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
    normalized_df = df.copy()

    for col in cols_to_normalize:
        rolling_mean = df[col].rolling(
            window=window_days, min_periods=1).mean()
        normalized_df[col] = df[col] / rolling_mean

    return normalized_df


def get_friday_dates(df):
    """Get all Friday dates from the dataframe."""
    return df[df.index.dayofweek == 4].index


# 252 trading days ≈ 1 year
def create_sequences_and_labels(df, normalized_df, sequence_length=252):
    """Create sequences of daily data and corresponding labels for each Friday."""
    friday_dates = get_friday_dates(df)

    sequences = []
    labels = []
    dates = []

    for i in range(len(friday_dates) - 1):
        current_friday = friday_dates[i]
        next_friday = friday_dates[i + 1]

        # Get one year of daily data before current Friday
        # Extra buffer for non-trading days
        sequence_start = current_friday - \
            pd.Timedelta(days=sequence_length * 1.5)
        sequence_end = current_friday

        # Get the daily sequence data
        sequence_mask = (normalized_df.index > sequence_start) & (
            normalized_df.index <= sequence_end)
        sequence_data = normalized_df[sequence_mask]

        # Only use sequences that have enough data
        if len(sequence_data) >= sequence_length:
            # Take the last 'sequence_length' days of data
            sequence_data = sequence_data.iloc[-sequence_length:]

            # Create label using original Adj Close prices
            current_adj_close = df.loc[current_friday, 'Close']
            next_adj_close = df.loc[next_friday, 'Close']
            # label = 1 if next_adj_close > current_adj_close else 0
            # label = 1 if current_adj_close*1.05 < next_adj_close else 0
            # label = 1 if current_adj_close*0.95 > next_adj_close else 0
            label = 1 if next_adj_close < current_adj_close * \
                1.05 and next_adj_close > current_adj_close*0.95 else 0

            # Store sequence, label, and date
            sequences.append(
                sequence_data[['Open', 'High', 'Low', 'Close', 'Volume']].values)
            labels.append(label)
            dates.append(current_friday)

    return np.array(sequences), np.array(labels), dates


def main():
    # Parameters
    start_date = '2010-01-01'  # Adjust as needed
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = 'ml_data'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load tickers
    tickers = load_tickers('ml/tickers.txt')

    # Process each ticker
    for ticker in tqdm(tickers):
        print(f"\nProcessing {ticker}...")

        # Download data
        df = download_stock_data(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        # Normalize data
        normalized_df = normalize_with_rolling_mean(df)

        # Create sequences and labels
        sequences, labels, dates = create_sequences_and_labels(
            df, normalized_df)

        # Save to disk
        if len(sequences) > 0:
            np.save(f'{output_dir}/{ticker}_sequences.npy', sequences)
            np.save(f'{output_dir}/{ticker}_labels.npy', labels)

            # Save dates as strings
            date_strings = [str(date) for date in dates]
            np.save(f'{output_dir}/{ticker}_dates.npy', date_strings)

            print(f"Saved {len(sequences)} sequences for {ticker}")


if __name__ == "__main__":
    main()
