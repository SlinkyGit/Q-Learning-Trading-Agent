import numpy as np
import datetime as dt
import pandas as pd
import math
import matplotlib.pyplot as plt
import indicators as indicator
import util as ut
import portfolio_simulator as ps


class RuleBasedStrategy(object):
    """
    A rule-based trading strategy that generates long, short, or neutral
    positions for a single stock using technical indicators.

    The strategy combines Bollinger Bands, RSI, MACD, and On-Balance Volume
    to identify overbought/oversold conditions and momentum shifts. It outputs
    a daily trades DataFrame indicating how many shares should be bought or
    sold to follow the strategy.
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
        Generate daily trading signals for a single stock using a rule-based strategy.
    
        The method computes Bollinger Bands, RSI, MACD, and OBV over the given
        date range, and uses them to determine when the strategy should be long,
        short, or neutral. The result is a DataFrame indicating the number of
        shares to buy or sell each day (positive = buy, negative = sell).
    
        Returns a DataFrame with positive values for buys, negative for sells,
        and zero when holding the current position.
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

    rb = RuleBasedStrategy(verbose=False, impact=impact, commission=commission)

    # Manual Strategy trades & portfolio values
    df_trades = rb.generate_trades(symbol=symbol, sd=sd, ed=ed, sv=sv)

    portvals_rb = ps.compute_portvals(df_trades=df_trades, start_val=sv, commission=commission, impact=impact)

    # benchmark trades & portfolio values
    # buy 1000 shares on first day -> hold to the end
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    benchmark_trades = prices_all[[symbol]].copy()
    benchmark_trades.values[:, :] = 0 # all zeros
    first_day = benchmark_trades.index[0]
    benchmark_trades.loc[first_day, symbol] = 1000 # buy 1000 shares on first day

    portvals_bench = ps.compute_portvals(df_trades=benchmark_trades, start_val=sv, commission=commission, impact=impact)

    # normalize for comparison
    normalized_rb = portvals_rb / portvals_rb.iloc[0]
    normalized_bench = portvals_bench / portvals_bench.iloc[0]

    # compute stats (derived from marketsimcode)
    daily_returns_rb = normalized_rb.pct_change().iloc[1:]
    daily_returns_bench = normalized_bench.pct_change().iloc[1:]

    cum_return_rb = normalized_rb.iloc[-1] - 1.0
    cum_return_bench = normalized_bench.iloc[-1] - 1.0

    avg_daily_return_rb = daily_returns_rb.mean()
    avg_daily_return_bench = daily_returns_bench.mean()

    std_daily_return_rb = daily_returns_rb.std()
    std_daily_return_bench = daily_returns_bench.std()

    print("Manual Strategy vs Benchmark (in-sample 2008–2009)")
    print(f"Cumulative Return (Manual): {cum_return_rb}")
    print(f"Cumulative Return (Benchmark): {cum_return_bench}")
    print(f"Avg Daily Return (Manual): {avg_daily_return_rb}")
    print(f"Avg Daily Return (Benchmark): {avg_daily_return_bench}")
    print(f"Std Daily Return (Manual): {std_daily_return_rb}")
    print(f"Std Daily Return (Benchmark): {std_daily_return_bench}")

    plot_manual_vs_benchmark(
    dates=normalized_rb.index,
    normalized_rb=normalized_rb,
    normalized_bench=normalized_bench,
    trades=df_trades,
    symbol=symbol,
    title="Manual Strategy vs Benchmark (In-Sample 2008–2009)",
    filename="manual_in_sample.png",
    )

    return portvals_rb["portvals"], portvals_bench["portvals"]


def out_of_sample():
    # out-of-sample dates and params
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    rb = RuleBasedStrategy(verbose=False, impact=impact, commission=commission)

    # Manual Strategy trades & portfolio values
    df_trades = rb.generate_trades(symbol=symbol, sd=sd, ed=ed, sv=sv)

    portvals_rb = ps.compute_portvals(df_trades=df_trades, start_val=sv, commission=commission, impact=impact)

    # benchmark trades & portfolio values
    # buy 1000 shares on first day, hold to the end
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    benchmark_trades = prices_all[[symbol]].copy()
    benchmark_trades.values[:, :] = 0 # all zeros
    first_day = benchmark_trades.index[0]
    benchmark_trades.loc[first_day, symbol] = 1000 # buy 1000 shares on first day

    portvals_bench = ps.compute_portvals(df_trades=benchmark_trades, start_val=sv, commission=commission, impact=impact)

    # normalize for comparison
    normalized_rb = portvals_rb / portvals_rb.iloc[0]
    normalized_bench = portvals_bench / portvals_bench.iloc[0]

    # basic stats
    daily_returns_rb = normalized_rb.pct_change().iloc[1:]
    daily_returns_bench = normalized_bench.pct_change().iloc[1:]

    cum_return_rb = normalized_rb.iloc[-1] - 1.0
    cum_return_bench = normalized_bench.iloc[-1] - 1.0

    avg_daily_return_rb = daily_returns_rb.mean()
    avg_daily_return_bench = daily_returns_bench.mean()

    std_daily_return_rb = daily_returns_rb.std()
    std_daily_return_bench = daily_returns_bench.std()

    print("Manual Strategy vs Benchmark (out-of-sample 2010–2011)")
    print(f"Cumulative Return (Manual): {cum_return_rb}")
    print(f"Cumulative Return (Benchmark): {cum_return_bench}")
    print(f"Avg Daily Return (Manual): {avg_daily_return_rb}")
    print(f"Avg Daily Return (Benchmark): {avg_daily_return_bench}")
    print(f"Std Daily Return (Manual): {std_daily_return_rb}")
    print(f"Std Daily Return (Benchmark): {std_daily_return_bench}")

    plot_manual_vs_benchmark(
    dates=normalized_rb.index,
    normalized_rb=normalized_rb,
    normalized_bench=normalized_bench,
    trades=df_trades,
    symbol=symbol,
    title="Manual Strategy vs Benchmark (Out-of-Sample 2010–2011)",
    filename="manual_out_sample.png",
    )

    return portvals_rb["portvals"], portvals_bench["portvals"]


def plot_manual_vs_benchmark(dates, normalized_rb, normalized_bench, trades, symbol, title, filename):
    # extract positions from trades
    holdings = trades.cursum()
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
    plt.plot(normalized_rb.index, normalized_rb.values, label='Manual Strategy', color='red')

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
    rb_in, bench_in = in_sample()
    rb_out, bench_out = out_of_sample()
    
    rb_cr_in, rb_adr_in, rb_sddr_in, _ = ps.stats(rb_in)
    bench_cr_in, bench_adr_in, bench_sddr_in, _ = ps.stats(bench_in)

    rb_cr_out, rb_adr_out, rb_sddr_out, _ = ps.stats(rb_out)
    bench_cr_out, bench_adr_out, bench_sddr_out, _ = ps.stats(bench_out)

    summary = pd.DataFrame({
        "Portfolio": [
            "rb In-Sample", "Benchmark In-Sample",
            "rb Out-of-Sample", "Benchmark Out-of-Sample"
        ],
        "Cumulative Return": [
            rb_cr_in, bench_cr_in,
            rb_cr_out, bench_cr_out
        ],
        "Avg Daily Return": [
            rb_adr_in, bench_adr_in,
            rb_adr_out, bench_adr_out
        ],
        "Std Daily Return": [
            rb_sddr_in, bench_sddr_in,
            rb_sddr_out, bench_sddr_out
        ]
    })

    print(summary)
    return summary

if __name__ == "__main__":
    # in_sample()
    # print()
    # out_of_sample()

    performance_summary()
