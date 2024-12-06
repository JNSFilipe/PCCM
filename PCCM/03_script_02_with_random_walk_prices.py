import optuna
import numpy as np
from enum import Enum
from BlackScholesMerton import BSM, OT
from utils import simulate_stock_price
# from optuna.visualization import plot_intermediate_values


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


UNDERLYING_PRICE = 100
INTEREST_RATE = 0.05
VOLATILITY = 0.1
SHORT_DELTA = 0.15
LONG_DELTA = 0.7


def premium_from_delta(delta, S, t):
    # Calculate strike price and premium for the long Call to buy
    return BSM.premium_from_delta(
        delta,
        S,
        INTEREST_RATE,
        VOLATILITY,
        t.value,
        OT.CALL
    )


def PCCM_n_short_against_one_long_profit(n, long_expiration, short_expiration, N=100):

    # Calculate strike price and premium for the long Call to buy
    long_sp, long_premium_buy = premium_from_delta(
        LONG_DELTA, UNDERLYING_PRICE, long_expiration)

    profit = []
    time = []
    for _ in range(N):  # Repeat N times, to get an average result
        pnl = []
        exercised = False
        S = UNDERLYING_PRICE
        for _ in range(n):
            K, C = premium_from_delta(SHORT_DELTA, S, short_expiration)
            S = simulate_stock_price(
                S, VOLATILITY, short_expiration.value, risk_free_rate=INTEREST_RATE)[0]

            # Check if price drops bellow the short strike -> Assignement
            if S < K:  # If so, Assignement occurs
                # Must buy 100 at long_sp to cover
                # Must sell 100 at K
                # Must add the profit from selling the short call (C)
                pnl.append(100*(K-long_sp) + 100*C)
                exercised = True
                n = len(pnl)  # n stops early, so update it
                break  # Lost underlying, must stop
            else:  # There is no assignemnt, keep on going
                pnl.append(100*C)

        if not exercised:  # If the option did not get exercised as colateral, sell it
            # Calculate strike price and premium for the short Call to sell
            long_premium_sell = BSM.premium(
                S,
                long_sp,
                INTEREST_RATE,
                VOLATILITY,
                long_expiration.value - n*short_expiration.value,
                option_type=OT.CALL
            )
            pnl.append(100 * long_premium_sell)
        profit.append(sum(pnl) - 100*long_premium_buy)
        time.append(n*short_expiration.value)

    return sum(profit)/N, sum(time)/N


if __name__ == "__main__":

    def f(trial):
        long_exp = trial.suggest_categorical(
            'long_expiration',
            [
                OExp.TWO_WEEKS,
                OExp.THREE_WEEKS,
                OExp.ONE_MONTH,
                OExp.TWO_MONTHS,
                OExp.THREE_MONTHS,
                OExp.FOUR_MONTHS,
                OExp.SIX_MONTHS,
                OExp.EIGHT_MONTHS,
                OExp.NINE_MONTHS,
                OExp.ONE_YEAR,
            ])

        short_exp = trial.suggest_categorical(
            'short_expiration',
            [
                OExp.ONE_WEEK,
                OExp.TWO_WEEKS,
                OExp.THREE_WEEKS,
                OExp.ONE_MONTH,
                OExp.TWO_MONTHS,
            ])

        n = trial.suggest_int('N', 1, 12)

        if n*short_exp.value >= long_exp.value:
            return -np.inf

        pnl, dt = PCCM_n_short_against_one_long_profit(n, long_exp, short_exp)

        return round(pnl/dt, 2)

    study = optuna.create_study(direction='maximize')
    study.optimize(f, n_trials=10000)

    print()

    # Add these lines after study.optimize()
    print("\n--- Optuna Optimization Results ---")
    print(f"Best Value: {study.best_value}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"{param.capitalize()}: {value}")
