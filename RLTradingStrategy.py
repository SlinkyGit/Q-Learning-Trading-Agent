import datetime as dt
import random

import pandas as pd
import util as ut
import QLearner as ql
import indicators as indicator
import numpy as np

class RLTradingStrategy(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """
    Reinforcement-learning trading agent built on tabular Q-learning.

    The agent observes a discretized state made from:
      - current position (short / flat / long)
      - binned technical indicators (BBP, RSI, MACD)

    It learns a policy over three actions (short / flat / long) by simulating
    trades on historical data with transaction costs (commission + market impact).
    """
  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Initialize the RL trading agent.
    
        Parameters
        ----------
        verbose : bool
            If True, prints debug information during training/inference.
        impact : float
            Slippage / market impact modeled as an adverse price adjustment on trades.
        commission : float
            Fixed dollar fee charged per transaction.
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        self.num_bins = 3 # per indicator
        self.num_holdings = 3 # short/flat/long
        self.num_indicators = 3 # BBP/RSI/MACD
        self.num_states = self.num_holdings * (self.num_bins ** self.num_indicators)
        self.num_actions = 3 # short/flat/long

        self.learner = ql.QLearner(
            num_states = self.num_states,
            num_actions = self.num_actions,
            alpha = 0.08, # learn faster
            gamma = 0.9, # care about future rewards
            rar = 0.85,
            radr = 0.994,
            dyna = 0,
            verbose = False
            )
 	 		  		  		    	 		 		   		 		  
    # this method creates a QLearner, and train it for trading  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def fit(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Train the Q-learning policy over a historical time window.
    
        The agent loops through the training period for multiple epochs. Each day,
        it:
          1) encodes the current state (position + binned indicators),
          2) selects an action (with exploration),
          3) simulates the trade (including costs),
          4) computes a reward based on portfolio value change,
          5) updates the Q-table.
    
        Parameters
        ----------
        symbol : str
            Ticker symbol to train on.
        sd : datetime
            Start date (inclusive).
        ed : datetime
            End date (inclusive).
        sv : float
            Starting cash for the simulated portfolio.
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
 		    	 		 		   		 		  
        # example usage of the old backward compatible util function  		  	   		 	 	 		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		 	 	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            print(prices)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # example use with new colname  		  	   		 	 	 		  		  		    	 		 		   		 		  
        volume_all = ut.get_data(  		  	   		 	 	 		  		  		    	 		 		   		 		  
            syms, dates, colname="Volume"  		  	   		 	 	 		  		  		    	 		 		   		 		  
        )  # automatically adds SPY  		  	   		 	 	 		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		  	   		 	 	 		  		  		    	 		 		   		 		  
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            print(volume)

        # compute indicators (continuous)
        bbp_series, rsi_series, macd_series = self.compute_indicators(prices, volume, symbol)

        # discretize indicators (0, 1, 2)
        bbp_bins = self.discretize_BBP(bbp_series)
        rsi_bins = self.discretize_RSI(rsi_series)
        macd_bins = self.discretize_MACD(macd_series)

        # print(bbp_series.head(10))
        # print(bbp_bins.head(10))
        # print(rsi_series.head(10))
        # print(rsi_bins.head(10))
        # print(macd_series.head(10))
        # print(macd_bins.head(10))

        dates_list = list(prices.index)
        price_arr = prices[symbol].values
        bbp_arr = bbp_bins.values
        rsi_arr = rsi_bins.values
        macd_arr = macd_bins.values

        # training loop (multiple epochs over same data)
        num_days = len(dates_list)
        epochs = 200

        for epoch in range(epochs):
            # portfolio state at start of epoch
            holdings = 0 # shares
            holdings_bin = 1
            cash = sv
            prev_port_val = cash

            # initial state & initial action
            bbp0 = bbp_arr[0]
            rsi0 = rsi_arr[0]
            macd0 = macd_arr[0]
            state = self.encode_state(holdings_bin, bbp0, rsi0, macd0)
            
            # querysetstate: set s, get initial action a
            action = self.learner.querysetstate(state)

            for day in range(1, num_days):
                price_today = price_arr[day]
                price_yesterday = price_arr[day - 1]

                # convert previous action → target holdings bin / shares
                # (0 -> -1000, 1 -> 0, 2 -> +1000)
                target_holdings = self.action_to_shares(action)

                # compute trade needed and apply transaction costs
                trade_shares = target_holdings - holdings
                trade_penalty = 0.0
                if trade_shares != 0:
                    # execution price moves against us because of impact
                    if trade_shares > 0: # buy
                        exec_price = price_yesterday * (1 + self.impact)
                        cash -= exec_price * trade_shares
                        cash -= self.commission
                    else: # sell
                        exec_price = price_yesterday * (1 - self.impact)
                        cash -= exec_price * trade_shares
                        cash -= self.commission

                trade_penalty = 0.995

                # update holdings
                holdings = target_holdings

                # discretize holdings
                holdings_bin = self.shares_to_bin(holdings)

                # compute new portfolio value and reward
                new_port_val = cash + holdings * price_today

                if prev_port_val != 0:
                    reward = ((new_port_val / prev_port_val) - 1.0) * 100 - trade_penalty

                prev_port_val = new_port_val

                # build new state from today's indicators
                bbp_today = bbp_arr[day]
                rsi_today = rsi_arr[day]
                macd_today = macd_arr[day]
                state_prime = self.encode_state(holdings_bin, bbp_today, rsi_today, macd_today)

                # let QLearner update own Q-table and choose next action
                action = self.learner.query(state_prime, reward)

  
    # this method uses the existing policy and tests it against new data
    def predict_trades(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Generate trades using the learned (greedy) policy over a new time window.
    
        Exploration is disabled and actions are chosen greedily from the learned
        Q-table. The output is a daily trades DataFrame (positive = buy, negative = sell).
    
        Parameters
        ----------
        symbol : str
            Ticker symbol to trade.
        sd : datetime
            Start date (inclusive).
        ed : datetime
            End date (inclusive).
        sv : float
            Starting cash (only used here if you also track portfolio value internally).
    
        Returns
        -------
        pd.DataFrame
            DataFrame of daily share trades indexed by date (one column: `symbol`).
        """ 		  	   		 	 	 		  		  		    	 		 		   		 		  

        dates = pd.date_range(sd, ed)
        syms = [symbol]
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]

        volume_all = ut.get_data(  		  	   		 	 	 		  		  		    	 		 		   		 		  
            syms, dates, colname="Volume"  		  	   		 	 	 		  		  		    	 		 		   		 		  
        )  # automatically adds SPY  		  	   		 	 	 		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		  	   		 	 	 		  		  		    	 		 		   		 		  
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later

        # compute indicators (continuous)
        bbp_series, rsi_series, macd_series = self.compute_indicators(prices, volume, symbol)

        # discretize indicators (0, 1, 2)
        bbp_bins = self.discretize_BBP(bbp_series)
        rsi_bins = self.discretize_RSI(rsi_series)
        macd_bins = self.discretize_MACD(macd_series)

        # convert to arrays for fast indexing by day
        dates_list = list(prices.index)
        price_arr = prices[symbol].values
        bbp_arr = bbp_bins.values
        rsi_arr = rsi_bins.values
        macd_arr = macd_bins.values

        num_days = len(dates_list)

        # trades has same index/column as prices
        trades = prices.copy()

        # start with no trades
        trades.values[:,:] = 0.0

        # portfolio state
        holdings = 0 # num shares currently held
        holdings_bin = 1 # start at a neutral/flat
        cash = sv # starting cash
        prev_port_val = sv 

        # in test, we want no exploration (greedy)
        self.learner.rar = 0.0

        # initial state & initial action
        bbp0 = bbp_arr[0]
        rsi0 = rsi_arr[0]
        macd0 = macd_arr[0]
        state = self.encode_state(holdings_bin, bbp0, rsi0, macd0)
        
        # choose best action
        action = self.best_action(state)

        for day in range(1, num_days):
                price_today = price_arr[day]
                price_yesterday = price_arr[day - 1]

                 # convert previous action → target holdings bin / shares
                 # (0 -> -1000, 1 -> 0, 2 -> +1000)
                target_holdings = self.action_to_shares(action)

                # compute trade needed and apply transaction costs
                trade_shares = target_holdings - holdings

                trades.iloc[day, 0] = trade_shares

                if trade_shares != 0:
                    # execution price moves against us because of impact
                    if trade_shares > 0: # buy
                        exec_price = price_yesterday * (1 + self.impact)
                        cash -= exec_price * trade_shares
                        cash -= self.commission
                    else: # sell
                        exec_price = price_yesterday * (1 - self.impact)
                        cash -= exec_price * trade_shares
                        cash -= self.commission

                holdings = target_holdings
                holdings_bin = self.shares_to_bin(holdings)
                new_port_val = cash + holdings * price_today
                prev_port_val = new_port_val

                # build next state
                bbp_today = bbp_arr[day]
                rsi_today = rsi_arr[day]
                macd_today = macd_arr[day]
                state = self.encode_state(holdings_bin, bbp_today, rsi_today, macd_today)
                action = self.best_action(state)

        return trades

    def compute_indicators(self, prices, volumes, symbol):
        """
        Compute the continuous indicator series used as features.
    
        Returns
        -------
        tuple(pd.Series, pd.Series, pd.Series)
            (BBP, RSI, MACD) for the requested symbol.
        """

        sma, ub, lb, bbp = indicator.bollinger_bands(prices)
        price_series = prices[symbol]      # this is a Series
        rsi = indicator.relative_strength_index(price_series)
        rsi = rsi.to_frame(name=symbol) # convert back to DF

        macd, macd_signal = indicator.moving_average_convergence_divergence(prices)

        # return as series for symbol
        return (bbp[symbol], rsi[symbol], macd[symbol])
    
    def discretize_BBP(self, bbp_series):
        """
        Convert continuous RSI values into 3 bins: oversold / neutral / overbought.
        NaNs are forward-filled; if no prior value exists, a neutral default is used.
        """
      
        # create empty bin (series)
        bins = pd.Series(index=bbp_series.index, dtype=int)
        last_valid = None

        for date in bbp_series.index:
            val = bbp_series.loc[date]

            # ref -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
            if pd.isna(val):
                if last_valid is not None:
                    val = last_valid
                else:
                    val = 0.5 # set to some arbitrary "neutral" value

            # determine which bin
            if val < 0.2: # cheap/near lower band
                bin_val = 0

            elif val > 0.8: # expensive/near upper band
                bin_val = 2

            else: # middle
                bin_val = 1

            bins.loc[date] = bin_val
            last_valid = val

        return bins.astype(int)
    
    def discretize_RSI(self, rsi_series):
        """
        Convert continuous RSI values into 3 bins: oversold / neutral / overbought.
        NaNs are forward-filled; if no prior value exists, a neutral default is used.
        """
        # create empty bin (series)
        bins = pd.Series(index=rsi_series.index, dtype=int)
        last_valid = None

        for date in rsi_series.index:
            val = rsi_series.loc[date]

            # ref -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
            if pd.isna(val):
                if last_valid is not None:
                    val = last_valid
                else:
                    val = 50 # set to some arbitrary "neutral" value

            # determine which bin
            if val < 30: # oversold
                bin_val = 0

            elif val > 70: # overbought
                bin_val = 2

            else: # neutral
                bin_val = 1

            bins.loc[date] = bin_val
            last_valid = val

        return bins.astype(int)
    
    def discretize_MACD(self, macd_series):
        """
        Convert continuous RSI values into 3 bins: oversold / neutral / overbought.
        NaNs are forward-filled; if no prior value exists, a neutral default is used.
        """
        # create empty bin (series)
        bins = pd.Series(index=macd_series.index, dtype=int)
        last_valid = None

        for date in macd_series.index:
            val = macd_series.loc[date]

            # ref -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
            if pd.isna(val):
                if last_valid is not None:
                    val = last_valid
                else:
                    val = 0.0 # set to some arbitrary "neutral" value

            # determine which bin
            if val < 0: # bearish momentum
                bin_val = 0

            elif val > 0: # bullish momentum
                bin_val = 2

            else: # neutral
                bin_val = 1

            bins.loc[date] = bin_val
            last_valid = val

        return bins.astype(int)

    def encode_state(self, holdings_bin, bbp_bin, rsi_bin, macd_bin):
        """
        Map (position bin + indicator bins) to a unique integer state ID.
    
        This creates a compact tabular state representation for Q-learning.
        """
        bins = self.num_bins
        
        # deterministic mapping:
        # state = H * (num_bins^3) + bbp_bin * (num_bins^2) + rsi_bin * (num_bins^1) + macd_bin

        # returns unique int between [0, num_states - 1] -> [0, 80]
        return (holdings_bin * (bins ** 3) + bbp_bin * (bins ** 2) + rsi_bin * (bins ** 1) + macd_bin)
    
    def action_to_shares(self, action):
        """Converts actions to respective long/short/flat positions"""
        if action == 0:
            # go short
            return -1000
        
        elif action == 1:
            # go flat -> neutral
            return 0
        
        else: # action == 2
            # go long
            return 1000

    def shares_to_bin(self, shares):
      """Converts respective long/short/flat positions to bins"""
        if shares < 0: # negative = short
            return 0
        
        elif shares == 0: # zero = flat/neutral
            return 1
        
        else: # positive = long
            return 2
        
    def best_action(self, state):
        """Return the greedy action (argmax Q) for a given state."""
        # q_val is a vector with length = num_actions
        q_vals = self.learner.Q[state, :]
        return np.argmax(q_vals)


if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("RL Strategy")
    # sl = RLTradingStrategy(verbose=False)

    # symbol="JPM"
    # sd=dt.datetime(2008, 1, 1)
    # ed=dt.datetime(2009, 1, 1)
    # sv=10000

    # sl.fit(symbol, sd, ed, sv)
