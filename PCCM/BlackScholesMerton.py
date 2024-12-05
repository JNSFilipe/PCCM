import numpy as np
from enum import Enum, auto
from scipy.stats import norm
from scipy.optimize import brentq

class OT(Enum):
    CALL = auto()
    PUT = auto()

class BSM:
    @staticmethod
    def _d1_d2(S, K, r, sigma, T):
        """
        Calculate d1 and d2 parameters used in Black-Scholes-Merton model
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        
        Returns:
        tuple: (d1, d2)
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def premium(S, K, r, sigma, T, option_type):
        """
        Calculate option premium using Black-Scholes-Merton model
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        option_type (OT): Type of option (CALL or PUT)
        
        Returns:
        float: Option premium
        """
        d1, d2 = BSM._d1_d2(S, K, r, sigma, T)
        
        match option_type:
            case OT.CALL:
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            case OT.PUT:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, r, sigma, T, option_type):
        """
        Calculate option delta
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        option_type (OT): Type of option (CALL or PUT)
        
        Returns:
        float: Option delta
        """
        d1, _ = BSM._d1_d2(S, K, r, sigma, T)
        
        match option_type:
            case OT.CALL:
                return norm.cdf(d1)
            case OT.PUT:
                return -norm.cdf(-d1)
    
    @staticmethod
    def gamma(S, K, r, sigma, T):
        """
        Calculate option gamma (same for calls and puts)
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        
        Returns:
        float: Option gamma
        """
        d1, _ = BSM._d1_d2(S, K, r, sigma, T)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, r, sigma, T, option_type, days_in_year=252):
        """
        Calculate option theta
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        option_type (OT): Type of option (CALL or PUT)
        days_in_year (int): Trading days in a year (default 252)
        
        Returns:
        float: Option theta (daily rate)
        """
        d1, d2 = BSM._d1_d2(S, K, r, sigma, T)
        
        match option_type:
            case OT.CALL:
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - \
                        r * K * np.exp(-r * T) * norm.cdf(d2)
            case OT.PUT:
                theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + \
                        r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return theta / days_in_year  # Convert to daily theta
    
    @staticmethod
    def vega(S, K, r, sigma, T):
        """
        Calculate option vega (same for calls and puts)
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        
        Returns:
        float: Option vega
        """
        d1, _ = BSM._d1_d2(S, K, r, sigma, T)
        return S * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def rho(S, K, r, sigma, T, option_type):
        """
        Calculate option rho
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        option_type (OT): Type of option (CALL or PUT)
        
        Returns:
        float: Option rho
        """
        d1, d2 = BSM._d1_d2(S, K, r, sigma, T)
        
        match option_type:
            case OT.CALL:
                return K * T * np.exp(-r * T) * norm.cdf(d2)
            case OT.PUT:
                return -K * T * np.exp(-r * T) * norm.cdf(-d2)

    @staticmethod
    def strike_from_delta(delta, S, r, sigma, T, option_type):
        """
        Calculate the strike price (K) for a given option delta using a robust root-finding method
        
        Parameters:
        delta (float): Desired option delta (must be between -1 and 1)
        S (float): Current stock price
        r (float): Risk-free interest rate
        sigma (float): Volatility of underlying asset
        T (float): Time to expiration (in years)
        option_type (OT): Type of option (CALL or PUT)

        Returns:
        float: The strike price (K) that results in the given delta
        """
        if not (-1 <= delta <= 1):
            raise ValueError(f"Delta must be between -1 and 1, but got {delta}")

        def delta_diff(K):
            return BSM.delta(S, K, r, sigma, T, option_type) - delta

        # Define search bounds based on option type and delta
        if option_type == OT.CALL:
            lower_bound = S * 0.1  # 10% of stock price
            upper_bound = S * 3.0  # 300% of stock price
        else:  # PUT
            lower_bound = S * 0.1  # 10% of stock price
            upper_bound = S * 3.0  # 300% of stock price

        try:
            K = brentq(delta_diff, lower_bound, upper_bound)
            return K
        except ValueError:
            raise ValueError(f"Could not find strike price for delta {delta}")

    @staticmethod
    def premium_from_delta(delta, S, r, sigma, T, option_type):

        sp = BSM.strike_from_delta(
            delta, 
            S, 
            r, 
            sigma, 
            T,
            option_type=option_type
        )
        # print(sp)

        premium = BSM.premium(
            S, 
            sp, 
            r, 
            sigma, 
            T,
            option_type=option_type
        )

        # print(premium)
        # print()

        return sp, premium
