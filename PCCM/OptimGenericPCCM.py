import optuna
import numpy as np
from utils import OExp, OT
from OptionModels import BSM
from joblib import Parallel, delayed  
from PriceModels import static_vol_brownian
# from optuna.visualization import plot_intermediate_values

UNDERLYING_PRICE = 100
INTEREST_RATE = 0.05
VOLATILITY = 0.3
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

def PCCM_n_short_against_one_long_profit(n, long_delta, long_expiration, short_delta, short_expiration, N=500):

    # Calculate strike price and premium for the long Call to buy
    long_sp, long_premium_buy = premium_from_delta(
        long_delta, UNDERLYING_PRICE, long_expiration)

    def single_simulation():
        pnl = []
        exercised = False
        S = UNDERLYING_PRICE
        for _ in range(n):
            K, C = premium_from_delta(short_delta, S, short_expiration)
            S = static_vol_brownian(
                S, VOLATILITY, short_expiration.value, risk_free_rate=INTEREST_RATE)[0]

            # Check if price drops below the short strike -> Assignment
            if S < K:
                # Must buy 100 at long_sp to cover
                # Must sell 100 at K
                # Must add the profit from selling the short call (C)
                pnl.append(100*(K-long_sp) + 100*C)
                exercised = True
                break  # Lost underlying, must stop
            else:
                # There is no assignment, keep on going
                pnl.append(100*C)

        if not exercised:
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

        return sum(pnl) - 100*long_premium_buy, n*short_expiration.value

    # Use joblib's Parallel to run simulations in parallel
    results = Parallel(n_jobs=-1)(delayed(single_simulation)() for _ in range(N))

    # Separate the profit and time from the results
    profits, times = zip(*results)

    # Calculate the average profit and average time
    average_profit = np.mean(profits)
    average_time = np.mean(times)

    return average_profit, average_time


def optim_pccm():
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

        short_delta = trial.suggest_int('short_delta', 10, 90) / 100.0
        long_delta = trial.suggest_int('long_delta', 10, 90) / 100.0
        n = trial.suggest_int('N', 1, 12)

        if (n*short_exp.value >= long_exp.value) or (long_delta <= short_delta):
            return -np.inf

        pnl, dt = PCCM_n_short_against_one_long_profit(
            n, long_delta, long_exp, short_delta, short_exp)

        return round(pnl/dt, 2)

    study = optuna.create_study(direction='maximize')
    study.optimize(f, n_trials=1000)

    print()

    # Add these lines after study.optimize()
    print("\n--- Optuna Optimization Results ---")
    print(f"Best Value: {study.best_value}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"{param.capitalize()}: {value}")

if __name__ == "__main__":
    optim_pccm()
