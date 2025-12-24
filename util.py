import os
import pandas as pd


def symbol_to_path(symbol: str, base_dir: str = "data") -> str:
    """Return the CSV file path for a given ticker symbol."""
    return os.path.join(base_dir, f"{symbol}.csv")


def get_data(symbols, dates, base_dir="data", col="Adj Close", add_spy=False):
    """
    Load adjusted close (or other column) prices for the given symbols over a date range.

    Parameters
    ----------
    symbols : list[str]
        Tickers to load.
    dates : pd.DatetimeIndex
        Date index for the output frame.
    base_dir : str
        Directory containing CSV files named like 'JPM.csv'.
    col : str
        Column name to read from each CSV (e.g., 'Adj Close', 'Close', 'Volume').
    add_spy : bool
        If True, include SPY as a reference series (if available).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `dates`, columns are symbols, values are the chosen column.
    """
    symbols = list(symbols)

    if add_spy and "SPY" not in symbols:
        symbols = ["SPY"] + symbols

    df = pd.DataFrame(index=dates)

    for sym in symbols:
        path = symbol_to_path(sym, base_dir=base_dir)
        df_sym = pd.read_csv(
            path,
            index_col="Date",
            parse_dates=True,
            usecols=["Date", col],
        ).rename(columns={col: sym})

        df = df.join(df_sym, how="left")

    return df
