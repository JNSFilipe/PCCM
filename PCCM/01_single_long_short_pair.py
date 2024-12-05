import optuna
import numpy as np
from enum import Enum
from BlackScholesMerton import BSM, OT
from optuna.visualization import plot_intermediate_values

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
VOLATILITY = 0.3
SHORT_DELTA = 0.15
LONG_DELTA = 0.7

def premium_from_delta(delta, expiration):
    ### Calculate strike price and premium for the long Call to buy
    return BSM.premium_from_delta(
        delta,
        UNDERLYING_PRICE,
        INTEREST_RATE,
        VOLATILITY,
        expiration.value,
        OT.CALL
    )

def PCCM_single_profit(long_expiration, short_expiration):

    ### Calculate strike price and premium for the long Call to buy
    long_sp, long_premium_buy = premium_from_delta(LONG_DELTA, long_expiration)

    ### Calculate strike price and premium for the short Call to sell
    short_sp, short_premium_sell = premium_from_delta(SHORT_DELTA, short_expiration)

    ### Calculate strike price and premium for the short Call to sell
    long_premium_sell = BSM.premium(
        UNDERLYING_PRICE, 
        long_sp, 
        INTEREST_RATE, 
        VOLATILITY, 
        long_expiration.value - short_expiration.value,
        option_type=OT.CALL
    )
    # print()
    # print(long_premium_sell)

    return 100*(short_premium_sell + long_premium_sell - long_premium_buy)


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

        if short_exp.value >= long_exp.value:
            return -np.inf

        return round(PCCM_single_profit(long_exp, short_exp)/short_exp.value, 2)


    study = optuna.create_study(direction='maximize')
    study.optimize(f, n_trials=1000)
    # study.optimize(f, n_trials=1000)

    print()

    # Add these lines after study.optimize()
    print("\n--- Optuna Optimization Results ---")
    print(f"Best Value: {study.best_value}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"{param.capitalize()}: {value}")