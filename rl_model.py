import numpy as np
import pandas as pd
from scipy.optimize import minimize

import bhv_analysis as bhv
from bhv_model import softmax

class RescorlaWagnerModel(object):
	'''
	Basic reinforcement learning model using Rescorla-Wagner learning rule
	'''

	def __init__(self, alpha=0.01, w2=-0.1, w3=0.1):
		'''
		alpha: 	learning rate
		w2: 	softmax weight of value difference
		w3: 	softmax left/right bias
		'''
		self.params_init = [alpha, w2, w3] # initial guess for parameter values
		self.bounds = [(0,1), (None,None), (None,None)] # parmeter bounds

		self.params_fit = self.params_init # fitted parameters
		self.aic = None # Akaike Information Criterion of best fit model

		self.levels2amnt = {1: 0.5, 2: 0.3, 3: 0.1} # conversion from amount levels to reward amounts in L

	def learningRule(self, value_old, outcome, alpha):
		'''
		Basic Rescorla-Wagner learning rule
		
		value_old: value to be updated
		outcome:   trial outcome (i.e. amount rewarded)
		alpha:     learning rate
		'''
		return value_old + alpha * (outcome - value_old)

	def estHiddenStates(self, data, sim_choice = False, alpha=None, w2=None, w3=None):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		DataFrame containing experimental dataset
		sim_choice: if True, simulate choices as well as hidden states;
					if False, simulate hidden states only and calculate likelihood of actual (non-simulated) choices
		alpha:		learning rate
		w2: 		softmax weight of value difference
		w3: 		softmax left/right bias
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		estimates = pd.DataFrame()

		if alpha is None:
			alpha = self.params_fit[0]

		if w2 is None:
			w2 = self.params_fit[1]

		if w3 is None:
			w3 = self.params_fit[2]

		# iterate through valid blocks of each session
		dates = data['date'].unique()
		for date in dates:
			block = data[data['date']==date]

			# initialize state values for block
			block_est = np.zeros((block.shape[0],10))

			for ii in range(block_est.shape[0]):
				trial = block.iloc[ii] # trial data

				idx_l = trial['left_image'] - 1 # left image index
				if not np.isnan(idx_l):
					idx_l = int(idx_l)

				idx_r = trial['right_image'] - 1 # right image index
				if not np.isnan(idx_r):
					idx_r = int(idx_r)

				# probability of choosing left
				if np.isnan(idx_l) or np.isnan(idx_r):
					p_l = np.nan
				else:
					p_l = softmax(block_est[ii, idx_l], block_est[ii, idx_r], w2, w3)

				# compute single-trial choice likelihood and update hidden state estimates
				if trial['lever'] == -1:
					block_est[ii, -1] = p_l # likelihood of left choice, assuming Bernoulli Distribution
					chosen_img = idx_l
					outcome = self.levels2amnt[trial['left_amnt_level']] * trial['if_reward'] # amount rewarded

				elif trial['lever'] == 1:
					block_est[ii, -1] = 1 - p_l # likelihood of right choice, assuming Bernoulli Distribution
					chosen_img = idx_r
					outcome = self.levels2amnt[trial['right_amnt_level']] * trial['if_reward']

				# update hidden state values
				if ii < block_est.shape[0]-1:
					for img in range(9):
						if img == chosen_img:
							block_est[ii+1, img] = self.learningRule(block_est[ii, img], outcome, alpha)
						else:
							block_est[ii+1, img] = block_est[ii, img]

			block_est = pd.DataFrame(block_est, columns=['v1','v2','v3','v4','v5','v6','v7','v8','v9','likelihood'])
			estimates = estimates.append(block_est)

		return estimates

	def negLogLikelihood(self, params, data):
		'''
		Calculate negative log-likelihood of choice behavior given model parameters
		
		params:	free parameters
		data: 	DataFrame containing experimental dataset
		'''
		params = list(params)
		estimates = self.estHiddenStates(data, *params)

		return -np.log(estimates['likelihood']).sum()

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

		estimates = self.estHiddenStates(data, *self.params_fit)

		return estimates, self.aic
