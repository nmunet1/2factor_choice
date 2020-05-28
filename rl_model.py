import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import bhv_analysis as bhv
from bhv_model import softmax

class RescorlaWagnerPlus(object):
	'''
	Generalized variant of Rescorla-Wagner reinforcement learning model with side bias and decaying learning rates
	'''

	def __init__(self, alpha=0.01, tau=0, gamma=0.01, c=0.1, beta=-0.1, delta=0.01, d=0.1, d_0=0.1):
		'''
		alpha: 		value learning rate
		tau:		learning rate decay constant
		gamma_c: 	stickiness learning rate
		c:			stickiness feedback
		beta: 		softmax inverse temperature
		delta: 		softmax side bias
		d:			left/right side bias feedback
		d_0:		initial left/right side bias
		'''
		self.params_init = [alpha, tau, gamma, c, beta, delta, d, d_0] # initial guess for parameter values
		self.bounds = [(0,1), (0,1), (0,1), (0,None), (None,None), (None,None), (0,None), (None,None)] # parameter bounds
		self.params_fit = self.params_init # fitted parameters

		# filter out Nonetype initial parameters and their bounds
		filt = [x is None for x in self.params_init]
		self.params_init = self.params_init[filt]
		self.bounds = self.bounds[filt]

		self.params_fit = self.params_init # fitted parameters
		self.aic = None # Akaike Information Criterion of best fit model

		self.levels2prob = {1: 0.7, 2: 0.4, 3: 0.1} # conversion from probability levels to probabilities of reward
		self.levels2amnt = {1: 0.5, 2: 0.3, 3: 0.1} # conversion from amount levels to reward amounts in L

	def learningRule(self, value_old, feedback, rate):
		'''
		Basic incremental learning rule
		
		value_old: 	value to be updated
		feedback:	learning signal
		rate:     	learning rate
		'''
		return value_old + rate * (feedback - value_old)

	def simulate(self, data, sim_choice, alpha=None, tau=None, gamma=None, c=None, beta=None, delta=None, d=None, d_0=None):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		DataFrame containing experimental dataset
		sim_choice: if True, simulate choices as well as hidden states;
					if False, simulate hidden states only and calculate likelihood of actual (non-simulated) choices
		alpha:		value learning rate
		tau:		learning rate exponential decay
		gamma:		stickiness learning rate
		c:			stickiness feedback
		beta: 		softmax weight of value difference
		d: 			left/right side bias
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		if alpha is None:
			alpha = self.params_fit[0]

		if tau is None:
			tau = self.params_fit[1]

		if gamma is None:
			gamma = self.params_fit[2]

		if c is None:
			c = self.params_fit[3]

		if beta is None:
			beta = self.params_fit[4]

		if delta is None:
			delta = self.params_fit[5]

		if d is None:
			d = self.params_fit[6]

		if d_0 is None:
			d_0 = self.params_fit[7]

		sim_results = pd.DataFrame()
		if sim_choice:
			cols = ['v1','v2','v3','v4','v5','v6','v7','v8','v9', \
				'c1','c2','c3','c4','c5','c6','c7','c8','c9', \
				'side_bias', 'choice','outcome']
		else:
			cols = ['v1','v2','v3','v4','v5','v6','v7','v8','v9', \
				'c1','c2','c3','c4','c5','c6','c7','c8','c9', \
				'side_bias', 'choice','likelihood']

		# iterate through valid blocks of each session
		dates = data['date'].unique()
		for date in dates:
			block = data[data['date']==date]

			# initialize state values for block
			block_sim = np.zeros((block.shape[0],len(cols)))
			block_sim[0,18] = d_0

			for ii in range(block_sim.shape[0]):
				trial = block.iloc[ii] # trial data

				idx_l = trial['left_image'] - 1 # left image index
				idx_r = trial['right_image'] - 1 # right image index

				# simulated probability of choosing left
				p_l = np.nan
				if np.isnan(idx_l):
					idx_r = int(idx_r)
					if sim_choice:
						p_l = 0
				elif np.isnan(idx_r):
					idx_l = int(idx_l)
					if sim_choice:
						p_l = 1
				else:
					idx_l = int(idx_l)
					idx_r = int(idx_r)

					q_l = block_sim[ii, idx_l] + block_sim[ii, idx_l+9]
					q_r = block_sim[ii, idx_r] + block_sim[ii, idx_r+9]

					p_l = softmax(q_l, q_r, beta, block_sim[ii,18])

				if sim_choice:
					# simulate choice and reward outcome
					if stats.bernoulli.rvs(p_l):
						choice = -1
						chosen = idx_l
						p_rwd = self.levels2prob[trial['left_prob_level']]
						amnt_rwd = self.levels2amnt[trial['left_amnt_level']]
					else:
						choice = 1
						chosen = idx_r
						p_rwd = self.levels2prob[trial['right_prob_level']]
						amnt_rwd = self.levels2amnt[trial['right_amnt_level']]

					if trial['lever'] == choice:
						outcome = amnt_rwd * trial['if_reward']
					else:
						outcome = amnt_rwd * stats.bernoulli.rvs(p_rwd)

					# log choice and outcome
					block_sim[ii, -2] = choice
					block_sim[ii, -1] = outcome

				else:
					# compute single-trial choice likelihood
					if trial['lever'] == -1:
						block_sim[ii, -1] = p_l # likelihood of left choice, assuming Bernoulli Distribution
						chosen = idx_l
						unchosen = idx_r
						outcome = self.levels2amnt[trial['left_amnt_level']] * trial['if_reward'] # amount rewarded

					elif trial['lever'] == 1:
						block_sim[ii, -1] = 1 - p_l # likelihood of right choice, assuming Bernoulli Distribution
						chosen = idx_r
						unchosen = idx_l
						outcome = self.levels2amnt[trial['right_amnt_level']] * trial['if_reward']

				# update hidden state values
				if ii < block_sim.shape[0]-1:
					for img in range(9):
						if img == chosen:
							block_sim[ii+1, img] = self.learningRule(block_sim[ii, img], \
								outcome, alpha*np.exp(-tau*ii)) # update value
							block_sim[ii+1, img+9] = self.learningRule(block_sim[ii, img+9], \
								c, gamma*np.exp(-tau*ii)) # increment choice bias
						else:
							block_sim[ii+1, img] = block_sim[ii, img] # maintain value
							if img == unchosen:
								block_sim[ii+1, img+9] = self.learningRule(block_sim[ii+1, img+9], \
									0, gamma*np.exp(-tau*ii)) # decrement choice bias
							else:
								block_sim[ii+1, img+9] = block_sim[ii, img] # maintain choice bias

					# update left/right side bias
					if trial['lever'] < 0:
						block_sim[ii+1, 18] = self.learningRule(block_sim[ii+1, 18], -d, delta*np.exp(-tau*ii))
					else:
						block_sim[ii+1, 18] = self.learningRule(block_sim[ii+1, 18], d, delta*np.exp(-tau*ii))

			block_sim = pd.DataFrame(block_sim, index=block.index, columns=cols)
			sim_results = sim_results.append(block_sim)

		return sim_results

	def negLogLikelihood(self, params, data):
		'''
		Calculate negative log-likelihood of choice behavior given model parameters
		
		params:	free parameters
		data: 	DataFrame containing experimental dataset
		'''
		params = list(params)
		sim_results = self.simulate(data, False, *params)

		return -np.log(sim_results['likelihood']).sum()

	def fit(self, data, params_init=None, verbose=False, **kwargs):
		'''
		Fit model free parameters

		data: 			DataFrame containing experimental dataset
		params_init: 	initial guess for free parameter values
		verbose: 		if True, display optimization results
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks

		if params_init is None:
			params = self.params_init
		else:
			params = params_init

		# fit data by minimizing negative log-likelihood of choice behavior given model and parameters
		opt = minimize(self.negLogLikelihood, params, args=(data), tol=1e-4, bounds=self.bounds, **kwargs)

		if verbose or not opt.success:
			print(opt)

		self.params_fit = list(opt.x) # update fitted parameters
		self.aic = 2*len(opt.x) + 2*opt.fun # Akaike Information Criterion

		sim_results = self.simulate(data, False, *self.params_fit)

		return sim_results, self.aic

class RescorlaWagner(RescorlaWagnerPlus):
	'''
	Basic Rescorla-Wagner model with stochastic (softmax) choice and fixed left-right bias
	'''

	def __init__(self, alpha=0.01, beta=-0.1, d_0=0.1):
		super().__init__(self, alpha=alpha, tau=None, gamma=None, c=None, \
			beta=beta, delta=delta, d=None, d_0=None)

	def simulate(self, data, sim_choice, alpha=None, beta=None, d_0=None):
		return super().simulate(alpha=alpha, beta=beta, d_0=d_0, gamma=0, delta=0)
