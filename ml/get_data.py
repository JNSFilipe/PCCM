import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import ta
from cons import K, STgt


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


# def add_technical_indicators(df):
#     """Add RSI, ATR, and Parabolic SAR indicators to the dataframe."""
#     # RSI
#     df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
#
#     # ATR
#     df['ATR'] = ta.volatility.AverageTrueRange(
#         high=df['High'],
#         low=df['Low'],
#         close=df['Close']
#     ).average_true_range()
#
#     # Parabolic SAR
#     sar_indicator = ta.trend.PSARIndicator(
#         high=df['High'],
#         low=df['Low'],
#         close=df['Close']
#     )
#     df['SAR'] = sar_indicator.psar()
#     df['SAR_Up'] = sar_indicator.psar_up()
#     df['SAR_Down'] = sar_indicator.psar_down()
#
#     return df


def add_technical_indicators(df):
    """Add trading signals based on technical indicators."""
    # Moving Average signals
    sma20 = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    sma50 = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['MA_Signal'] = np.where(
        sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))

    # MACD signals
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Signal'] = np.where(macd.macd() > macd.macd_signal(), 1, -1)

    # Bollinger Bands signals
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Signal'] = np.where(df['Close'] < bb.bollinger_lband(), 1,
                               np.where(df['Close'] > bb.bollinger_hband(), -1, 0))

    # RSI signals
    rsi = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['RSI_Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))

    # Parabolic SAR signals
    sar_indicator = ta.trend.PSARIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )
    df['SAR_Signal'] = np.where(df['Close'] > sar_indicator.psar(), 1, -1)

    # Ichimoku Cloud signals
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    span_a = ichimoku.ichimoku_a()
    span_b = ichimoku.ichimoku_b()
    df['Cloud_Signal'] = np.where(
        (df['Close'] > span_a) & (df['Close'] > span_b), 1,
        np.where((df['Close'] < span_a) & (df['Close'] < span_b), -1, 0)
    )

    # New indicators (https://arxiv.org/pdf/2310.09903)

    # 1. Squeeze Pro Signal
    # Using Bollinger Bands and Keltner Channels for squeeze detection
    bb = ta.volatility.BollingerBands(df['Close'])
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()

    kc = ta.volatility.KeltnerChannel(
        high=df['High'], low=df['Low'], close=df['Close'])
    kc_width = kc.keltner_channel_hband() - kc.keltner_channel_lband()

    # Squeeze is on when Bollinger Bands are inside Keltner Channel
    df['Squeeze_Signal'] = np.where(bb_width < kc_width,
                                    np.where(df['Close'] > bb.bollinger_hband(), 1,
                                             np.where(df['Close'] < bb.bollinger_lband(), -1, 0)),
                                    0)

    # 2. Thermo Signal (using momentum indicators)
    roc = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
    mfi = ta.volume.MFIIndicator(high=df['High'], low=df['Low'],
                                 close=df['Close'], volume=df['Volume']).money_flow_index()

    df['Thermo_Signal'] = np.where((roc > 0) & (mfi > 80), 1,
                                   np.where((roc < 0) & (mfi < 20), -1, 0))

    # 3. PPO (Percentage Price Oscillator) Signal
    ppo = ta.momentum.PercentagePriceOscillator(df['Close'])
    df['PPO_Signal'] = np.where(ppo.ppo() > ppo.ppo_signal(), 1, -1)

    # 4. Decay Signal (using exponential decay of price movements)
    ema_short = ta.trend.EMAIndicator(df['Close'], window=5).ema_indicator()
    ema_long = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    price_decay = (df['Close'] - ema_short) / (ema_long - ema_short)

    df['Decay_Signal'] = np.where(price_decay > 0.8, -1,  # Overbought
                                  # Oversold
                                  np.where(price_decay < 0.2, 1, 0))

    return df


def normalize_with_rolling_mean(df, window=52):
    """Normalize OHLCV data and technical indicators by dividing by 52-week rolling mean."""
    # Convert window from weeks to business days
    window_days = window * 5

    # Calculate weekly rolling mean for each column
    cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
    normalized_df = df.copy()

    # Normalize OHLCV data
    for col in cols_to_normalize:
        rolling_mean = df[col].rolling(
            window=window_days, min_periods=1).mean()
        normalized_df[col] = df[col] / rolling_mean

    # if K.INDICATORS:
    #     # Normalize technical indicators
    #     # RSI is already normalized (0-100), so we don't normalize it
    #
    #     # Normalize ATR
    #     rolling_mean = df['ATR'].rolling(
    #         window=window_days, min_periods=1).mean()
    #     rolling_mean = rolling_mean.replace(0, np.nan)
    #     normalized_df['ATR'] = df['ATR'] / rolling_mean
    #     normalized_df['ATR'] = normalized_df['ATR'].fillna(0)
    #
    #     # Normalize SAR indicators relative to price
    #     for col in ['SAR', 'SAR_Up', 'SAR_Down']:
    #         # Normalize relative to price
    #         normalized_df[col] = df[col] / df['Close']
    #         normalized_df[col] = normalized_df[col].fillna(0)

    return normalized_df


def get_friday_dates(df):
    """Get all Friday dates from the dataframe."""
    return df[df.index.dayofweek == 4].index


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

            # Create label using original Close prices
            current_close = df.loc[current_friday, 'Close']
            next_close = df.loc[next_friday, 'Close']
            match K.TARGET:
                case STgt.UP:
                    label = 1 if next_close > current_close * (1+K.PCT) else 0
                case STgt.DWN:
                    label = 1 if next_close < current_close * (1-K.PCT) else 0
                case STgt.CONS:
                    label = 1 if next_close < current_close * (1+K.PCT) \
                        and next_close > current_close * (1-K.PCT) else 0

            # Store sequence, label, and date
            # Include normalized OHLCV data and technical indicators
            if K.INDICATORS:
                sequences.append(sequence_data[['Open', 'High', 'Low', 'Close', 'Volume',
                                                'MA_Signal', 'MACD_Signal', 'BB_Signal',
                                                'RSI_Signal', 'SAR_Signal', 'Cloud_Signal',
                                                'Squeeze_Signal', 'Thermo_Signal',
                                                'PPO_Signal', 'Decay_Signal']].values)
            else:
                sequences.append(
                    sequence_data[['Open', 'High', 'Low', 'Close', 'Volume']].values)

            labels.append(label)
            dates.append(current_friday)

    return np.array(sequences), np.array(labels), dates


def main():
    # Parameters
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = K.DATA_DIR

    # Remove prior data
    os.system(f'rm -rf {output_dir}')

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load tickers
    tickers = load_tickers(K.TICKERS_DIR)

    # Process each ticker
    for ticker in tqdm(tickers):
        print(f"\nProcessing {ticker}...")

        # Download data
        df = download_stock_data(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        if K.INDICATORS:
            # Add technical indicators
            df = add_technical_indicators(df)

            # Handle any NaN values that might have been introduced
            df = df.fillna(method='ffill').fillna(method='bfill')

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
            # This will show (n_samples, sequence_length, n_features)
            print(f"Data shape: {sequences.shape}")


if __name__ == "__main__":
    main()
