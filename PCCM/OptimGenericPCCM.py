import optuna
import numpy as np
from PCCM.utils import OExp, prob_of_profit
from PCCM.Strategies import PCCM
from joblib import Parallel, delayed


def PCCM_n_short_against_one_long_profit(
        S0,
        r,
        sigma,
        n,
        long_delta,
        long_expiration,
        short_delta,
        short_expiration,
        returns_model,
        option_model,
        prob_metric=True,
        N=500):

    # Create a PCCM object
    pccm = PCCM(option_model)
    gp = returns_model(S0, sigma, r=r)

    def single_simulation():
        price = gp.simulate(long_expiration.value, weekends=True)
        pnl = pccm.simulate(price, r, sigma, n, long_delta,
                            long_expiration, short_delta, short_expiration)

        if prob_metric:
            m = sum(pnl)
        else:
            m = sum(pnl) / ((len(pnl)-1)*short_expiration.value)
        return m

    # Use joblib's Parallel to run simulations in parallel
    profits = Parallel(n_jobs=-1)(delayed(single_simulation)()
                                  for _ in range(N))
    # profits = single_simulation()

    if prob_metric:
        pp = prob_of_profit(profits)
        return 100*pp

    # Calculate the average profit and average time
    average_profit = np.mean(profits)

    return average_profit


def optim_pccm(S, r, sigma, returns_model, option_model, prob_metric=True, n_sims=500, n_trails=1000):
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

        if (n * short_exp.value >= long_exp.value) or (long_delta <= short_delta):
            return -np.inf

        pnl = PCCM_n_short_against_one_long_profit(S, r, sigma, n,
                                                   long_delta, long_exp,
                                                   short_delta, short_exp,
                                                   returns_model, option_model,
                                                   prob_metric=prob_metric, N=n_sims)

        return round(pnl, 2)

    study = optuna.create_study(direction='maximize')
    study.optimize(f, n_trials=n_trails)

    print()

    # Add these lines after study.optimize()
    print("\n--- Optuna Optimization Results ---")
    print(f"Best Value: {study.best_value}")
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        print(f"{param.capitalize()}: {value}")
