import numpy as np
from time import time
from abc import ABC, abstractmethod


class PriceModel(ABC):
    """
    Abstract base class for price models.
    Provides common functionality for handling returns and price calculations.
    """

    def __init__(self, S0):
        self.S0 = S0

    @abstractmethod
    def simulate(self, time: float, weekends: bool = False) -> np.ndarray:
        """
        Simulate price paths according to the model.
        Must be implemented by concrete classes.

        Parameters:
        - time (float): Time horizon in years
        - weekends (bool): Whether to include weekend returns

        Returns:
        - np.ndarray: Array of simulated prices
        """
        pass

    def _update_seed(self):
        """
        Private method to update random seed.
        """
        if self.seed is None:
            self.seed = int(time())
        np.random.seed(self.seed)
        self.seed = 0x5f3759df - (self.seed >> 1)  # DOOM

    def _add_weekends(self, returns_array, trading_days=5, weekend_returns=(0, 0)):
        """
        Private method to insert weekend returns into the returns array.
        """
        returns_array = np.array(returns_array)
        num_complete_weeks = len(returns_array) // trading_days
        remaining_days = len(returns_array) % trading_days

        new_size = len(returns_array) + \
            (num_complete_weeks * len(weekend_returns))
        if remaining_days > 0:
            new_size += len(weekend_returns)

        result = np.zeros(new_size)
        original_idx = 0
        result_idx = 0

        while original_idx < len(returns_array):
            chunk_size = min(trading_days, len(returns_array) - original_idx)
            result[result_idx:result_idx +
                   chunk_size] = returns_array[original_idx:original_idx + chunk_size]
            result_idx += chunk_size
            original_idx += chunk_size

            if chunk_size == trading_days or original_idx >= len(returns_array):
                result[result_idx:result_idx +
                       len(weekend_returns)] = weekend_returns
                result_idx += len(weekend_returns)

        return result

    def _returns_to_prices(self, returns, initial_price):
        """
        Private method to convert returns to prices.
        """
        returns = np.array(returns)
        returns[0] = 0  # set initial return to 0
        multipliers = returns + 1
        cumulative_returns = np.cumprod(multipliers)
        prices = initial_price * cumulative_returns
        return prices


class GaussianProcess(PriceModel):
    """
    Gaussian process implementation of PriceModel.
    Simulates price paths using geometric Brownian motion.
    """

    def __init__(self, S0, sigma, r=0.02, seed=None):
        super().__init__(S0)
        self.sigma = sigma
        self.r = r
        self.seed = seed

    def simulate(self, time, weekends=False):
        """
        Simulate price paths using geometric Brownian motion.
        """
        self._update_seed()
        num_steps = int(time * 252)  # 252 trading days in a year

        daily_vol = self.sigma / np.sqrt(252)
        # daily_returns = np.random.normal(
        #     loc=(self.r - 0.5 * self.sigma**2) / 252,  # drift term
        #     scale=daily_vol,
        #     size=num_steps
        # )
        rng = np.random.default_rng()
        daily_returns = rng.normal(
            loc=(self.r - 0.5 * self.sigma**2) / 252,  # drift term
            scale=daily_vol,
            size=num_steps
        )

        if weekends:
            daily_returns = self._add_weekends(daily_returns)

        prices = self._returns_to_prices(daily_returns, self.S0)
        return prices


class StudentTProcess(PriceModel):
    """
    Student's t-process implementation of PriceModel.
    Simulates price paths using a Student's t-distribution to account for fat tails.
    """

    def __init__(self, S0, sigma, nu, r=0.02, seed=None):
        """
        Initialize the StudentTProcess.

        Parameters:
        - S0 (float): Initial stock price
        - sigma (float): Volatility parameter
        - nu (float): Degrees of freedom for the Student's t-distribution
        - r (float): Risk-free rate (default: 0.02)
        - seed (int, optional): Seed for random number generator
        """
        super().__init__(S0)
        self.sigma = sigma
        self.nu = nu
        self.r = r
        self.seed = seed

    def simulate(self, time, weekends=False):
        """
        Simulate price paths using a Student's t-distribution.

        Parameters:
        - time (float): Time horizon in years
        - weekends (bool): Whether to include weekend returns

        Returns:
        - np.ndarray: Array of simulated prices
        """
        self._update_seed()
        num_steps = int(time * 252)  # 252 trading days in a year

        daily_vol = self.sigma / np.sqrt(252)
        rng = np.random.default_rng()

        # Calculate scaling factor for Student's t to match variance
        # The variance of Student's t is sigma^2 if scaled appropriately
        scaling_factor = np.sqrt(
            (self.nu - 2) / self.nu) if self.nu > 2 else 1.0

        # Generate Student's t-distributed returns
        daily_returns = rng.standard_t(
            df=self.nu, size=num_steps) * daily_vol * scaling_factor

        # Add drift term
        drift = (self.r - 0.5 * self.sigma**2) / 252
        daily_returns += drift

        if weekends:
            daily_returns = self._add_weekends(daily_returns)

        prices = self._returns_to_prices(daily_returns, self.S0)
        return prices
