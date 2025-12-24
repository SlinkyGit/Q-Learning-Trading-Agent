import numpy as np
import pandas as pd

def compute_portfolio_values(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    starting_cash: float = 100000.0,
    commission: float = 0.0,
    impact: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate a daily mark-to-market portfolio given trade signals.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices for each symbol. Index = dates, columns = symbols.
    trades : pd.DataFrame
        Signed share trades for each symbol. Same shape/index/columns as `prices`.
        Positive values = buy shares, negative values = sell shares.
    starting_cash : float
        Initial cash balance at the beginning of the backtest.
    commission : float
        Fixed dollar commission per transaction.
    impact : float
        Slippage / market impact expressed as a fractional price penalty.
        For buys, the effective price is increased by (1 + impact).
        For sells, the effective price is decreased by (1 - impact).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame of total portfolio value per day.
    """
    if prices.shape != trades.shape:
        raise ValueError("prices and trades must have the same shape")

    symbols = list(prices.columns)
    dates = prices.index

    # Track per-day changes in holdings for each symbol
    position_changes = trades.copy().astype(float)

    # Track per-day changes in cash
    cash_changes = pd.Series(0.0, index=dates)
    cash_changes.iloc[0] = starting_cash

    # Loop through each trade to adjust cash for fills, commissions, and impact
    for date in dates:
        for symbol in symbols:
            shares = trades.at[date, symbol]
            if shares == 0:
                continue

            price = prices.at[date, symbol]

            if shares > 0:
                # Buy: pay a slightly worse price and subtract commission
                exec_price = price * (1.0 + impact)
                cash_delta = -(exec_price * shares + commission)
            else:
                # Sell: receive a slightly worse price and subtract commission
                exec_price = price * (1.0 - impact)
                cash_delta = exec_price * abs(shares) - commission

            cash_changes.at[date] += cash_delta

    # Cumulative holdings over time
    holdings = position_changes.cumsum()

    # Cumulative cash over time
    cash = cash_changes.cumsum()

    # Add cash as a "symbol" with price = 1
    prices_with_cash = prices.copy()
    prices_with_cash["CASH"] = 1.0

    holdings_with_cash = holdings.copy()
    holdings_with_cash["CASH"] = cash

    # Total value = sum over symbols + cash each day
    portfolio_values = (holdings_with_cash * prices_with_cash).sum(axis=1)

    return portfolio_values.to_frame("portfolio_value")


def portfolio_stats(values: pd.Series, trading_days: int = 252):
    """
    Compute basic performance statistics for a portfolio value series.

    Parameters
    ----------
    values : pd.Series
        Portfolio values over time.
    trading_days : int
        Number of trading days per year (used in Sharpe ratio).

    Returns
    -------
    tuple
        (cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio)
    """
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    daily_returns = values.pct_change().dropna()

    cum_ret = values.iloc[-1] / values.iloc[0] - 1.0
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std(ddof=1)
    sharpe = np.sqrt(trading_days) * (avg_daily_ret / std_daily_ret)

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe
