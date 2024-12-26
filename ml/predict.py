import torch
from datetime import datetime, timedelta
from get_data import (
    load_tickers,
    download_stock_data,
    normalize_with_rolling_mean,
    add_technical_indicators
)
from models import StockPriceCNN
from colorama import init, Fore, Style
from cons import K

# Initialize colorama
init()


def get_last_friday():
    """Get the date of the last Friday."""
    today = datetime.now()
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)
    return last_friday.replace(hour=16, minute=0, second=0, microsecond=0)


def get_next_friday():
    """Get the date of the next Friday."""
    today = datetime.now()
    days_until_friday = (4 - today.weekday()) % 7  # 4 represents Friday
    if days_until_friday == 0 and today.hour >= 16:  # After market close
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    return next_friday.replace(hour=16, minute=0, second=0, microsecond=0)


def get_end_date():
    """Get the appropriate end date based on current day."""
    today = datetime.now()
    if today.weekday() == 4:  # If today is Friday
        return today if today.hour < 16 else get_last_friday()
    return get_last_friday()


def prepare_sequence(df, normalized_df, sequence_length=252):
    """Prepare the most recent sequence for prediction."""
    if len(df) < sequence_length:
        return None

    # Get the most recent sequence_length days of data
    sequence_data = normalized_df.iloc[-sequence_length:]

    # Prepare the sequence in the correct format
    sequence = sequence_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Add batch and channel dimensions
    sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    # Transpose to [batch, channels, sequence_length]
    sequence = sequence.transpose(1, 2)

    return sequence


def format_prediction(prediction, confidence):
    """Format prediction with appropriate color."""
    pred_str = "Stable" if prediction == 1 else "Volatile"

    # Base color based on prediction
    if prediction == 1:
        color = Fore.GREEN
    else:
        color = Fore.RED

    return f"{color}{pred_str:>12}{Style.RESET_ALL}"


def format_confidence(confidence):
    """Format confidence with appropriate color."""

    # Override color if confidence is high
    if confidence >= 80:
        return f"{Fore.BLUE}{confidence:>11.1f}%{Style.RESET_ALL}"
    elif confidence >= 70:
        return f"{Fore.YELLOW}{confidence:>11.1f}%{Style.RESET_ALL}"
    return f"{confidence:>11.1f}%"


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = StockPriceCNN().to(device)
    model.load_state_dict(torch.load(K.MDL_DIR, map_location=device))
    model.eval()

    # Load tickers
    tickers = load_tickers(K.TICKERS_DIR)

    # Calculate dates
    end_date = get_end_date()
    start_date = end_date - timedelta(days=252 * 2)
    next_friday = get_next_friday()

    # Print date information
    print("\nData Range:")
    print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End Date:   {end_date.strftime('%Y-%m-%d')}")
    print(f"Predicting for: {next_friday.strftime('%Y-%m-%d')}\n")

    print(f"{'Ticker':<8} {'Current':>12} {
          '-3%':>12} {'+3%':>12} {'Prediction':>12} {'Confidence':>12}")
    print("-" * 70)

    with torch.no_grad():
        for ticker in tickers:
            # Download and prepare data
            df = download_stock_data(ticker, start_date.strftime('%Y-%m-%d'),
                                     end_date.strftime('%Y-%m-%d'))

            if df is None or df.empty:
                continue

            normalized_df = normalize_with_rolling_mean(df)
            sequence = prepare_sequence(df, normalized_df)

            if sequence is None:
                print(f"Insufficient data for {ticker}")
                continue

            # Make prediction
            sequence = sequence.to(device)
            outputs = model(sequence)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

            # Get current price and calculate ranges
            current_price = df['Close'].iloc[-1]
            lower_bound = current_price * (1 - K.PCT)
            upper_bound = current_price * (1 + K.PCT)

            # Print results with colors
            pred_str = format_prediction(prediction, confidence)
            conf_str = format_confidence(confidence)
            print(f"{ticker:<8} ${current_price:>11.2f} ${lower_bound:>11.2f} ${
                  upper_bound:>11.2f} {pred_str} {conf_str}")


if __name__ == "__main__":
    main()
