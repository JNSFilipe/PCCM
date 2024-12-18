from PCCM.utils import OT

# TODO: maybe make abstract classes for strategies, options, and returns models


class PCCM():

    def __init__(self, option_model):
        self.option_model = option_model

    def simulate(self, ts, r, sigma, n, long_delta, long_expiration, short_delta, short_expiration):

        pnl = []
        weekend_dur = 2.0/365.0  # 2 days in years
        shrt_dur = int(short_expiration.value * 365)  # in days

        # Calculate strike price and premium for the long Call to buy
        long_sp, long_premium_buy = self.option_model.premium_from_delta(
            long_delta,
            ts[0],
            r,
            sigma,
            long_expiration.value - weekend_dur,  # Rm lst wknd, exp is on fri
            OT.CALL)
        pnl.append(-100*long_premium_buy)

        i = 0
        n_shorts = 0
        exercised = False
        expiration = False
        # from pudb import set_trace
        # set_trace()
        while not exercised and n_shorts < n:

            # Get today's price
            S = ts[i]

            if not expiration:  # Not exp, is monday, open short position

                K, C = self.option_model.premium_from_delta(
                    short_delta,
                    S,
                    r,
                    sigma,
                    short_expiration.value,
                    OT.CALL)

                # Setting up for next iteration
                i = i + shrt_dur - 2  # minus 2 days for last weekend
                i = i - 1  # minus 1 cause index starts at 0
                expiration = True

            else:  # Expired, is friday, check if exp worthless or exercised

                # Check if price goes above the short strike -> Assignment
                if S > K:
                    # Must buy 100 at long_sp to cover
                    # Must sell 100 at K
                    # Must add the profit from selling the short call (C)
                    pnl.append(100 * (K - long_sp) + 100 * C)
                    exercised = True
                else:
                    # There is no assignment, keep on going
                    # Add premium from selling the short call
                    pnl.append(100 * C)

                # Setting up for next iteration
                n_shorts += 1
                i = i + 2 + 1  # Add the weekend, go to monday
                expiration = False  # Reset for next week

        # If long option was not exercised, we need to close it
        if not exercised:
            # Calculate strike price and premium for the short Call to sell
            long_premium_sell = self.option_model.premium(
                S,
                long_sp,
                r,
                sigma,
                long_expiration.value - n*short_expiration.value - weekend_dur,
                option_type=OT.CALL
            )
            # Add prmium to the selling of the long call
            pnl[0] += 100 * long_premium_sell

        return pnl
