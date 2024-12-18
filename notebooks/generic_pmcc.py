import marimo

__generated_with = "0.10.1"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import marimo as mo
    import sys
    sys.path.insert(0, '../PCCM/')
    return mo, sys


@app.cell
def _(mo):
    mo.md(r"""# Optimising Generic Poor Man's Covered Calls""")
    return


@app.cell
def _(mo):
    mo.md("""## Define Number of Simulations and Optimisation Trails""")
    return


@app.cell
def _():
    N_SIMS = 1000
    N_TRAILS = 200
    return N_SIMS, N_TRAILS


@app.cell
def _(mo):
    mo.md("""## Define Underlying Price, Vlatility, and Risk Free Interest Rate""")
    return


@app.cell
def _():
    UNDERLYING_PRICE = 100
    INTEREST_RATE = 0.05
    VOLATILITY = 0.3
    return INTEREST_RATE, UNDERLYING_PRICE, VOLATILITY


@app.cell
def _(mo):
    mo.md("""## Define Returns Simulation Model and Options Model""")
    return


@app.cell
def _():
    from PCCM.PriceModels import GaussianProcess
    from PCCM.OptionModels import BSM

    RM = GaussianProcess
    OM = BSM
    return BSM, GaussianProcess, OM, RM


@app.cell
def _(mo):
    mo.md("""## Preform Optimisation""")
    return


@app.cell
def _(
    INTEREST_RATE,
    N_SIMS,
    N_TRAILS,
    OM,
    RM,
    UNDERLYING_PRICE,
    VOLATILITY,
):
    from PCCM.OptimGenericPCCM import optim_pccm

    optim_pccm(UNDERLYING_PRICE, INTEREST_RATE,
               VOLATILITY, RM, OM, N_SIMS, N_TRAILS)
    return (optim_pccm,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The most common result is:

        |Best Parameters   |           |
        |------------------|-----------|
        | Long expiration  | Two Months|
        | Short expiration | One Week  |
        | Short $\Delta$   | 20        |
        | Long $\Delta$    | 90        |
        | N                | 7         |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Deeper Analysis""")
    return


@app.cell
def _(GaussianProcess):
    from PCCM.utils import OExp

    m = GaussianProcess(100, 0.3, 0.05)
    p = [m.simulate(OExp.ONE_YEAR.value, weekends=True) for _ in range(100)]
    return OExp, m, p


@app.cell
def _(mo, p):
    import plotly.express as px

    x = list(range(len(p[0])))

    fig = px.line(x=x, y=p[0])
    for l in p[1:]:
        fig.add_scatter(x=x, y=l)

    mo.ui.plotly(fig)
    return fig, l, px, x


@app.cell
def _(INTEREST_RATE, OExp, OM, VOLATILITY, p):
    from pprint import pprint
    from PCCM.Strategies import PCCM

    pccm = PCCM(option_model=OM)

    pnls = []
    for t in p:
        pnl = pccm.simulate(t, INTEREST_RATE, VOLATILITY, 7, 0.9, OExp.TWO_MONTHS, 0.2, OExp.ONE_WEEK)
        pnls.append(pnl)

    pprint([sum(i) for i in pnls])
    return PCCM, pccm, pnl, pnls, pprint, t


@app.cell
def _(pnls):
    from PCCM.utils import prob_of_profit

    prob_of_profit(pnls)
    return (prob_of_profit,)


if __name__ == "__main__":
    app.run()
