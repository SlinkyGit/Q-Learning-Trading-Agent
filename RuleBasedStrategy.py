import numpy as np
import datetime as dt
import pandas as pd
import math
import matplotlib.pyplot as plt
import indicators as indicator
import util as ut
import marketsimcode as msc


class RuleBasedStrategy(object):
    """
            Tests learner using data outside of the training data

            Parameters
                symbol (str) – The stock symbol that you trained on on
                sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2009
                ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2010
                sv (int) – The starting value of the portfolio
            Returns
                A single column data frame, indexed by date, representing trades for each day. Legal values are +1000.0 indicating
                a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
                Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
                long so long as net holdings are constrained to -1000, 0, and 1000.

            Return type
                pandas.DataFrame
    """
  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        # copied from StrategyLearner.py
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def generate_trades(self, symbol='IBM', sd=dt.datetime(2009, 1, 1, 0, 0), ed=dt.datetime(2010, 1, 1, 0, 0), sv=100000):
        """
        Tests your learner using data outside of the training data

        Parameters
            symbol (str) – The stock symbol that you trained on on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2009
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2010
            sv (int) – The starting value of the portfolio
        Returns
            A single column data frame, indexed by date, representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.

        Return type
            pandas.DataFrame
        """

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        volume_all = ut.get_data([symbol], dates, colname="Volume")
        volumes = volume_all[[symbol]]
        prices = prices_all[[symbol]]

        sma, ub, lb, bbp = indicator.bollinger_bands(prices)

        price_series = prices[symbol] # series
        rsi = indicator.relative_strength_index(price_series)
        rsi = rsi.to_frame(name=symbol) # convert back to DF

        macd, macd_signal = indicator.moving_average_convergence_divergence(prices)
        obv = indicator.on_balance_volume(prices, volumes)

        obv_trend = obv.diff(5) # raw OBV is a huge number, so observe recent fluctuations
        
        trades = prices.copy()

        trades.values[:,:] = 0

        position = 0 # current holding: -1000, 0, or +1000

        if self.verbose:
            print("BBP head:\n", bbp.head())
            print("RSI head:\n", rsi.head())
            print("MACD head:\n", macd.head())
            print("OBV head:\n", obv.head())

        for date in prices.index:
            b = bbp.loc[date, symbol]
            r = rsi.loc[date, symbol]
            m = macd.loc[date, symbol]
            o = obv_trend.loc[date, symbol]
            
            # handle NaNs
            if np.isnan(b) or np.isnan(r) or np.isnan(m) or np.isnan(o):
                desired_pos = position
            
            else:

                # bullish setup -> “cheap, oversold, starting or in an uptrend, with buying volume behind it”
                if ((b < 0.35) # (price near lower band)
                    and (r < 40) # (oversold-ish)
                    and ((m > 0)  # (MACD says uptrend / momentum up)
                    or (o > 0))): # (OBV trend rising -> buying volume)
                    # go long -> strong buy signal
                    desired_pos = 1000

                # bearish setup -> “expensive, overbought, downtrend, with selling volume behind it”
                elif ((b > 0.65) # (price near upper band)
                    and (r > 55) # (overbought-ish)
                    and ((m < 0)  # (downtrend)
                    or (o < 0))): # (selling volume)
                    # go short -> strong sell signal
                    desired_pos = -1000

                else: # hold what you have
                    desired_pos = position

            trade = desired_pos - position
            trades.loc[date, symbol] = trade
            position = desired_pos

        return trades

def in_sample():

    # In-sample dates and params
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    ms = RuleBasedStrategy(verbose=False, impact=impact, commission=commission)

    # Manual Strategy trades & portfolio values
    df_trades = ms.generate_trades(symbol=symbol, sd=sd, ed=ed, sv=sv)

    portvals_ms = msc.compute_portvals(df_trades=df_trades, start_val=sv, commission=commission, impact=impact)

    # benchmark trades & portfolio values
    # buy 1000 shares on first day -> hold to the end
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    benchmark_trades = prices_all[[symbol]].copy()
    benchmark_trades.values[:, :] = 0 # all zeros
    first_day = benchmark_trades.index[0]
    benchmark_trades.loc[first_day, symbol] = 1000 # buy 1000 shares on first day

    portvals_bench = msc.compute_portvals(df_trades=benchmark_trades, start_val=sv, commission=commission, impact=impact)

    # normalize for comparison
    normalized_ms = portvals_ms / portvals_ms.iloc[0]
    normalized_bench = portvals_bench / portvals_bench.iloc[0]

    # compute stats (derived from marketsimcode)
    daily_returns_ms = normalized_ms.pct_change().iloc[1:]
    daily_returns_bench = normalized_bench.pct_change().iloc[1:]

    cum_return_ms = normalized_ms.iloc[-1] - 1.0
    cum_return_bench = normalized_bench.iloc[-1] - 1.0

    avg_daily_return_ms = daily_returns_ms.mean()
    avg_daily_return_bench = daily_returns_bench.mean()

    std_daily_return_ms = daily_returns_ms.std()
    std_daily_return_bench = daily_returns_bench.std()

    print("Manual Strategy vs Benchmark (in-sample 2008–2009)")
    print(f"Cumulative Return (Manual): {cum_return_ms}")
    print(f"Cumulative Return (Benchmark): {cum_return_bench}")
    print(f"Avg Daily Return (Manual): {avg_daily_return_ms}")
    print(f"Avg Daily Return (Benchmark): {avg_daily_return_bench}")
    print(f"Std Daily Return (Manual): {std_daily_return_ms}")
    print(f"Std Daily Return (Benchmark): {std_daily_return_bench}")

    plot_manual_vs_benchmark(
    dates=normalized_ms.index,
    normalized_ms=normalized_ms,
    normalized_bench=normalized_bench,
    trades=df_trades,
    symbol=symbol,
    title="Manual Strategy vs Benchmark (In-Sample 2008–2009)",
    filename="manual_in_sample.png",
    )

    return portvals_ms["portvals"], portvals_bench["portvals"]


def out_of_sample():
    # out-of-sample dates and params
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    ms = RuleBasedStrategy(verbose=False, impact=impact, commission=commission)

    # Manual Strategy trades & portfolio values
    df_trades = ms.generate_trades(symbol=symbol, sd=sd, ed=ed, sv=sv)

    portvals_ms = msc.compute_portvals(df_trades=df_trades, start_val=sv, commission=commission, impact=impact)

    # benchmark trades & portfolio values
    # buy 1000 shares on first day, hold to the end
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    benchmark_trades = prices_all[[symbol]].copy()
    benchmark_trades.values[:, :] = 0 # all zeros
    first_day = benchmark_trades.index[0]
    benchmark_trades.loc[first_day, symbol] = 1000 # buy 1000 shares on first day

    portvals_bench = msc.compute_portvals(df_trades=benchmark_trades, start_val=sv, commission=commission, impact=impact)

    # normalize for comparison
    normalized_ms = portvals_ms / portvals_ms.iloc[0]
    normalized_bench = portvals_bench / portvals_bench.iloc[0]

    # basic stats
    daily_returns_ms = normalized_ms.pct_change().iloc[1:]
    daily_returns_bench = normalized_bench.pct_change().iloc[1:]

    cum_return_ms = normalized_ms.iloc[-1] - 1.0
    cum_return_bench = normalized_bench.iloc[-1] - 1.0

    avg_daily_return_ms = daily_returns_ms.mean()
    avg_daily_return_bench = daily_returns_bench.mean()

    std_daily_return_ms = daily_returns_ms.std()
    std_daily_return_bench = daily_returns_bench.std()

    print("Manual Strategy vs Benchmark (out-of-sample 2010–2011)")
    print(f"Cumulative Return (Manual): {cum_return_ms}")
    print(f"Cumulative Return (Benchmark): {cum_return_bench}")
    print(f"Avg Daily Return (Manual): {avg_daily_return_ms}")
    print(f"Avg Daily Return (Benchmark): {avg_daily_return_bench}")
    print(f"Std Daily Return (Manual): {std_daily_return_ms}")
    print(f"Std Daily Return (Benchmark): {std_daily_return_bench}")

    plot_manual_vs_benchmark(
    dates=normalized_ms.index,
    normalized_ms=normalized_ms,
    normalized_bench=normalized_bench,
    trades=df_trades,
    symbol=symbol,
    title="Manual Strategy vs Benchmark (Out-of-Sample 2010–2011)",
    filename="manual_out_sample.png",
    )

    return portvals_ms["portvals"], portvals_bench["portvals"]


def plot_manual_vs_benchmark(dates, normalized_ms, normalized_bench, trades, symbol, title, filename):
    # extract positions from trades
    holdings = trades.cumsum()
    position = holdings[symbol]

    long_dates = []
    short_dates = []

    prev_position = 0

    for date in position.index:
        curr_position = position.loc[date]

        # long entry: from short (<=0) to long (>0)
        if prev_position <= 0 and curr_position > 0:
            long_dates.append(date)

        # short entry: from long (>=0) to short (<0)
        if prev_position >= 0 and curr_position < 0:
            short_dates.append(date)

        prev_position = curr_position

    plt.figure(figsize=(10, 6))
    plt.plot(normalized_bench.index, normalized_bench.values, label='Benchmark', color='purple')
    plt.plot(normalized_ms.index, normalized_ms.values, label='Manual Strategy', color='red')

    # vertical lines for entry points
    # ref -> https://www.geeksforgeeks.org/python/plot-a-vertical-line-in-matplotlib/
    for date in long_dates:
        plt.axvline(x=date, color='blue', linestyle='-', linewidth=1)

    for date in short_dates:
        plt.axvline(x=date, color='black', linestyle='-', linewidth=1)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def performance_summary():
    # obtain data from in-sample/out-of-sample testing
    ms_in, bench_in = in_sample()
    ms_out, bench_out = out_of_sample()
    
    ms_cr_in, ms_adr_in, ms_sddr_in, _ = msc.stats(ms_in)
    bench_cr_in, bench_adr_in, bench_sddr_in, _ = msc.stats(bench_in)

    ms_cr_out, ms_adr_out, ms_sddr_out, _ = msc.stats(ms_out)
    bench_cr_out, bench_adr_out, bench_sddr_out, _ = msc.stats(bench_out)

    summary = pd.DataFrame({
        "Portfolio": [
            "MS In-Sample", "Benchmark In-Sample",
            "MS Out-of-Sample", "Benchmark Out-of-Sample"
        ],
        "Cumulative Return": [
            ms_cr_in, bench_cr_in,
            ms_cr_out, bench_cr_out
        ],
        "Avg Daily Return": [
            ms_adr_in, bench_adr_in,
            ms_adr_out, bench_adr_out
        ],
        "Std Daily Return": [
            ms_sddr_in, bench_sddr_in,
            ms_sddr_out, bench_sddr_out
        ]
    })

    print(summary)
    return summary

if __name__ == "__main__":
    # in_sample()
    # print()
    # out_of_sample()

    performance_summary()
