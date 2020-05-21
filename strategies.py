from pandas_datareader import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TIME = {"year": 260, "month": 21.75}

class Strategy:

	def __init__(self, data, start, end):
		# indexed by date
		# columns represent stock tickers
		self.data = data
		self.start_date = start
		self.end_date = end

		business_days = pd.date_range(start=start, end=end, freq="B")
		self.data = data.reindex(business_days).fillna(method="ffill")
		self.num_assets = len(data.columns)
		self.tickers = data.columns

		# basic transformations all strategies use
		self.log_returns = np.log(self.data).diff()

	def initialize_with_strategy(self, weights_matrix):
		"""
		Secondary initialization function that takes a WEIGHTS_MATRIX which is ultimately
		multiplied with log_returns
		"""

		# multiply our log returns with our weight matrix
		self.strategy_log_returns = self.log_returns * weights_matrix 

		# Cumulative Sum of our strategy (both log and relative)
		# These are used to plot all assets and their cumulative sums
		self.cum_strategy_log_returns = self.strategy_log_returns.cumsum()
		self.cum_strategy_relative_returns = np.exp(self.strategy_log_returns.cumsum()) - 1


		# EXACT
		# Converts each asset's relative return
		cum_strategy_relative_returns_total = np.exp(self.cum_strategy_log_returns) - 1

		# Sums over each row, so all asset's relative return get totaled
		# Used to plot entire portfolio's performance
		self.cum_relative_returns_exact = cum_strategy_relative_returns_total.sum(axis=1)

		# APPROX
		# Sum over each row, so all asset's relative return get totaled
		cum_strategy_log_returns_total = self.cum_strategy_log_returns.sum(axis=1)

		# Converts each asset's relative return
		# Used to plot entire portfolio's performance
		self.cum_relative_returns_approx = np.exp(cum_strategy_log_returns_total) - 1


	def compute_portfolio_return(self, unit="year"):
		# Calculating the time-related parameters of the simulation
		data = self.cum_relative_returns_exact
		days_per_unit = TIME[unit]
		total_days_in_simulation = data.shape[0]
		number_of_units = total_days_in_simulation / days_per_unit

		# The last data point will give us the total portfolio return
		total_portfolio_return = data[-1]
		# Average portfolio return assuming compunding of returns
		# (1 + r)^12 = 1 + total_portfolio_return
		# (1 + r) = (1 + total_portfolio_return)^1/12
		# r = (1 + total_portfolio_return)^1/12 - 1
		# basically, at what rate are we compounding monthly to get our computed final growth
		average_unit_return = (1 + total_portfolio_return)**(1 / number_of_units) - 1
		print("Total portfolio return is {:.2f}%".format(100 * total_portfolio_return))
		print("Average {}ly return is {:.2f}%".format(unit, 100 * average_unit_return))
		return (total_portfolio_return, 100 * average_unit_return)

	def plot_cum_log_returns(self):
		"""
		Plots the cumulative log returns over time according to our weights strategy
		"""
		fig, ax = plt.subplots(figsize=(9,6))

		# plots log returns over time, recall this is r(t) = log(p(t) / p(t - 1))
		for c in self.tickers:
		    ax.plot(self.cum_strategy_log_returns.index, self.cum_strategy_log_returns[c], label=str(c))

		ax.set_ylabel('Cumulative log returns')
		ax.legend(loc='best')

		return ax

	def plot_cum_relative_returns(self):
		"""
		Plots the cumulative relative returns over time according to our weights strategy
		"""
		fig, ax = plt.subplots(figsize=(9,6))
		# converts log returns to relative returns (i.e. readable)
		# this is simply the cummulative sum of all logged prices, raised to e
		for c in self.tickers:
		    ax.plot(self.cum_strategy_relative_returns.index, self.cum_strategy_relative_returns[c] * 100, label=str(c))

		ax.set_ylabel('Total relative returns (%)')
		ax.legend(loc='best')

		return ax

	def plot_all_assets_behavior(self):
		fig, ax = plt.subplots(figsize=(9,6))

		ax.plot(self.cum_relative_returns_exact.index, 100 * self.cum_relative_returns_exact, label='Exact')
		ax.plot(self.cum_relative_returns_approx.index, 100 * self.cum_relative_returns_approx, label='Approximation')

		ax.set_ylabel('Total cumulative relative returns (%)')
		ax.legend(loc='best')

		return ax


class RelativeReturns(Strategy):
	"""
	Returns a strategy object with the following returns function
	- r(t) is the relative return at timestep t
	- p(t) is the price of a certain stock at timestep t

	r(t) = p(t) - p(t - 1) / p(t - 1)
	or 
	r_log(t) = log(p(t) / p(t-1)) = log(p(t)) - log(p(t-1))
	"""
	name = "RR"
	implemented = True

	def __init__(self, data, start, end):
		super().__init__(data, start, end)
		# Preliminaries
		self.cumulative_log_returns = self.log_returns.cumsum()
		self.relative_returns = data.pct_change(1)
		self.cumulative_relative_returns = (np.exp(self.cumulative_log_returns) - 1) 
		self.weights_matrix = pd.DataFrame(1/self.num_assets, self.log_returns.index, self.log_returns.columns)
		self.initialize_with_strategy(self.weights_matrix)


class ExponentialMovingAverage(Strategy):
	"""
	This is a strategy that that determines weights (buy or sell) based on if
	the price curve is above or below the exponential moving average curve
	"""
	name = "EMA"
	implemented = True

	def __init__(self, data, start, end):
		super().__init__(data, start, end)
		# generates ema curve with a 20 time window
		self.ema_short = data.ewm(span=20, adjust=False).mean()
		# difference between prices and ema
		self.price_ema = data - self.ema_short
		# if negative, assign negative, else positive
		self.weights_matrix = (self.price_ema.apply(np.sign) * 1/self.num_assets).shift(1)
		self.initialize_with_strategy(self.weights_matrix)
		
	







