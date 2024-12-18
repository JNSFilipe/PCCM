from enum import Enum, auto


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


class OT(Enum):
    CALL = auto()
    PUT = auto()
