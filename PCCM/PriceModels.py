import numpy as np
from time import time

# Set random seed for reproducibility
np.random.seed(int(time()))

def static_vol_brownian(initial_price, volatility, time, risk_free_rate=0.02, num_simulations=1):
    """
    Simulate stock price using geometric Brownian motion.

    Parameters:
    - initial_price: Starting stock price
    - volatility: Annualized volatility (standard deviation of returns)
    - time: Time period in years
    - risk_free_rate: Risk-free interest rate (default 2%)
    - num_simulations: Number of independent price path simulations to generate

    Returns:
    - Final simulated stock price(s)
    """
    # Number of time steps (assume daily simulation)
    num_steps = int(time * 252)  # 252 trading days in a year

    # Generate random walk
    # Standard deviation of daily returns
    daily_vol = volatility / np.sqrt(252)

    # Generate random normal distribution for price movements
    daily_returns = np.random.normal(
        loc=(risk_free_rate - 0.5 * volatility**2) / 252,  # drift term
        scale=daily_vol,
        size=(num_simulations, num_steps)
    )

    # Simulate price paths
    price_paths = np.zeros((num_simulations, num_steps + 1))
    price_paths[:, 0] = initial_price

    for t in range(1, num_steps + 1):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(daily_returns[:, t-1])

    # Return final simulated prices
    return price_paths[:, -1]
