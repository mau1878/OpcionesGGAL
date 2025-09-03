import QuantLib as ql
from datetime import date

eval_date = date(2025, 9, 3)
expiration_date = date(2025, 10, 17)
evaluation_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
expiration = ql.Date(expiration_date.day, expiration_date.month, expiration_date.year)
ql.Settings.instance().evaluationDate = evaluation_date

spot_price = 5360.0
strike = 5378.5
risk_free_rate = 0.35
volatility = 0.3

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluation_date, risk_free_rate, ql.Actual365Fixed()))
volatility_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(evaluation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(evaluation_date, 0.0, ql.Actual365Fixed()))

try:
    process = ql.BlackScholesProcess(spot_handle, dividend_ts, risk_free_ts, volatility_ts)
    print("BlackScholesProcess initialized successfully")
except Exception as e:
    print(f"Error: {e}")