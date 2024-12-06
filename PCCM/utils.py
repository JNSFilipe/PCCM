import numpy as np
from time import time
from enum import Enum

# Set random seed for reproducibility
# np.random.seed(42)
np.random.seed(int(time()))


class OExp(Enum):
    ONE_WEEK = 1/52  # 1 week (in years)
    TWO_WEEKS = 2/52  # 2 weeks (in years)
    THREE_WEEKS = 3/52  # 3 weeks (in years)
    ONE_MONTH = 1/12  # 1 month (in years)
    TWO_MONTHS = 2/12  # 2 months (in years)
    THREE_MONTHS = 3/12  # 3 months (in years)
    FOUR_MONTHS = 4/12  # 4 months (in years)
    SIX_MONTHS = 6/12  # 6 months (in years)
    EIGHT_MONTHS = 8/12  # 8 months (in years)
    NINE_MONTHS = 9/12  # 9 months (in years)
    ONE_YEAR = 1  # 1 year (in years)

    def __str__(self):
        return self.name.replace('_', ' ').title()


def simulate_stock_price(initial_price, volatility, time, risk_free_rate=0.02, num_simulations=1):
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
