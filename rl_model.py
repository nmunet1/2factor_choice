import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, shgo

import bhv_analysis as bhv
from bhv_model import softmax, fitSubjValues

class RescorlaWagnerModel(object):
	'''
	Basic Rescorla-Wagner model with stochastic (softmax) choice and fixed left-right bias
	'''

	def __init__(self, fname=None, alpha=0.01, beta=-0.1, lr_bias=0.1):
		'''
		alpha: 		(float) value learning rate
		beta: 		(float)softmax inverse temperature
		lr_bias:	(float) initial left/right side bias
		'''
		self.params_init = {'alpha': alpha, 'beta': beta, 'lr_bias': lr_bias} # initial parameter values
		self.bounds = {'alpha': (0,1), 'beta': (None,None), 'lr_bias': (None,None)} # parameter bounds
		#self.params_fit = self.params_init.copy() # fitted parameter values
		self.params_fit = pd.DataFrame(columns=['alpha','beta','lr_bias'])

		self.aic = None # Akaike Information Criterion of best fit model
		self.sims = None # Simulation results from bootstrapped simulations

		self.levels2prob = {1: 0.7, 2: 0.4, 3: 0.1} # conversion from probability levels to probabilities of reward
		self.levels2amnt = {1: 0.5, 2: 0.3, 3: 0.1} # conversion from amount levels to reward amounts in L

	def learningRule(self, value_old, rate, feedback):
		'''
		Basic incremental learning rule
		
		value_old: 	(float) value to be updated
		rate:		(float) learning rate
		feedback:	(float) learning signal
		'''
		return value_old + rate * (feedback - value_old)

	def simulate(self, data, params=None, sim_choice=True, track_values=True, merge_data=False):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		sim_choice: (bool) if True, simulate choices as well as hidden states;
					if False, simulate hidden states only and calculate likelihood of actual (non-simulated) choices
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		if params is None:
			set_params = True
			# params = self.params_fit
		else:
			set_params = False

		sim_results = pd.DataFrame()
		if track_values:
			cols = ['value1','value2','value3','value4','value5','value6','value7','value8','value9','choice']
		else:
			cols = ['choice']
		if sim_choice:
			cols.append('outcome')
		else:
			cols.append('likelihood')

		# iterate through valid blocks of each session
		dates = data['date'].unique()
		for date in dates:
			block = data[data['date']==date]

			if set_params:
				try:
					params = self.params_fit.loc[date].to_dict()
				except KeyError:
					params = None
					raise

			# initialize state values for block
			block_sim = np.full((block.shape[0],len(cols)), np.nan)
			values = np.zeros(9)

			if not params is None:
				for ii in range(block_sim.shape[0]):
					trial = block.iloc[ii] # trial data

					if track_values:
						block_sim[ii, :9] = values # record values at start of trial

					idx_l = trial['left_image'] - 1
					if not np.isnan(idx_l):
						idx_l = int(idx_l)

					idx_r = trial['right_image'] - 1
					if not np.isnan(idx_r):
						idx_r = int(idx_r)

					# simulated probability of choosing left
					if (not np.isnan(idx_l)) and (not np.isnan(idx_r)):
						p_l = softmax(values[idx_l], values[idx_r], params['beta'], params['lr_bias'])
					elif sim_choice:
						p_l = float(np.isnan(idx_r))
					else:
						p_l = np.nan

					if sim_choice:
						# simulate choice and reward outcome
						if stats.bernoulli.rvs(p_l):
							choice = -1
							chosen = idx_l # chosen image index
							p_rwd = self.levels2prob[trial['left_prob_level']] # probability of reward
							amnt_rwd = self.levels2amnt[trial['left_amnt_level']] # amount of reward
						else:
							choice = 1
							chosen = idx_r
							p_rwd = self.levels2prob[trial['right_prob_level']]
							amnt_rwd = self.levels2amnt[trial['right_amnt_level']]

						if trial['lever'] == choice:
							outcome = amnt_rwd * trial['if_reward']
						else:
							outcome = amnt_rwd * stats.bernoulli.rvs(p_rwd)

						block_sim[ii, -2] = choice # record choice
						block_sim[ii, -1] = outcome # record outcome

					else:
						# compute single-trial choice likelihood
						block_sim[ii, -2] = trial['lever']

						if trial['lever'] == -1:
							block_sim[ii, -1] = p_l # likelihood of left choice, assuming Bernoulli Distribution
							chosen = idx_l # chosen image index
							outcome = self.levels2amnt[trial['left_amnt_level']] * trial['if_reward']
						elif trial['lever'] == 1:
							block_sim[ii, -1] = 1 - p_l # likelihood of right choice, assuming Bernoulli Distribution
							chosen = idx_r
							outcome = self.levels2amnt[trial['right_amnt_level']] * trial['if_reward']

					values[chosen] = self.learningRule(values[chosen], params['alpha'], outcome) # value update

			block_sim = pd.DataFrame(block_sim, index=block.index, columns=cols)
			sim_results = sim_results.append(block_sim)

		if merge_data:
			sim_results = pd.concat([data, sim_results], axis=1, sort=False)

		return sim_results

	def bootstrap(self, data, n_iter=100, track_values=True, merge_data=True, verbose=False):
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		if merge_data:
			results = data.copy()
		else:
			results = pd.DataFrame()

		for n in range(n_iter):
			if n % 10 == 0 and verbose:
				print('simulation', n)
			res_n = self.simulate(data, sim_choice=True, track_values=track_values, merge_data=False)
			results = pd.concat([results, res_n], axis=1)
		self.sims = results

		return results

	def negLogLikelihood(self, params, data, param_labels):
		'''
		Calculate negative log-likelihood of choice behavior given model parameters
		
		params:			(sequence) free parameters
		data: 			(DataFrame) experimental dataset
		param_labels: 	(list) parameter labels
		'''
		param_dict = {}
		for key, value in zip(param_labels, params):
			param_dict[key] = value
		sim_results = self.simulate(data, param_dict, sim_choice=False, track_values=False)

		return -np.log(sim_results['likelihood']).sum()

	def fit(self, data, params_init=None, verbose=False, min_type='local', **kwargs):
		'''
		Fit model free parameters

		data: 			(DataFrame) experimental dataset
		params_init: 	(dict) initial guess for free parameter values
		verbose: 		(bool) if True, display optimization results
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		dates = data['date'].unique()

		for date in dates:
			sess_data = data[data['date']==date]

			if params_init is None:
				param_labels = list(self.params_init.keys())
				params = [self.params_init[label] for label in param_labels]
			else:
				param_labels = list(params_init.keys())
				params = [params_init[label] for label in param_labels]

			bounds = [self.bounds[label] for label in param_labels]

			# fit data by minimizing negative log-likelihood of choice behavior given model and parameters
			if min_type == 'local':
				opt = minimize(self.negLogLikelihood, params, args=(sess_data, param_labels), \
					tol=1e-4, bounds=bounds, **kwargs)
				if verbose or not opt.success:
					print(opt)
			elif min_type == 'global':
				opt = shgo(self.negLogLikelihood, bounds, args=(sess_data, param_labels), \
					minimizer_kwargs={'method':'L-BFGS-B'}, options={'f_tol':1e-4, 'max_iter': 100, 'disp':verbose})
			else:
				raise ValueError

			self.params_fit.loc[date, param_labels] = list(opt.x)

		sim_results = self.simulate(data, sim_choice=False, merge_data=True)
		negLL = -np.log(sim_results['likelihood']).sum()
		self.aic = 2*len(params)*len(dates) + 2*negLL # Update Akaike Information Criterion

		return sim_results, self.aic

	def save(self, fname):
		pass
		# for key in self.__dict__.keys():
		# 	attr = self.__dict__[key]
		# 	if type(attr) == pd.core.frame.DataFrame:
		# 		pass
		# 	elif type(attr) == dict:
		# 		pass
		# 	else:
		# 		pass

class FixedSoftmaxRescorlaWagnerModel(RescorlaWagnerModel):
	def __init__(self, alpha=0.01):
		super().__init__()

		self.params_init = {'alpha': alpha}
		self.bounds = {'alpha': (0,1)}

	def negLogLikelihood(self, params, data, param_labels):
		'''
		Calculate negative log-likelihood of choice behavior given model parameters
		
		params:			(sequence) free parameters
		data: 			(DataFrame) experimental dataset
		param_labels: 	(list) parameter labels
		'''
		date = data['date'].unique()[0]

		param_dict = {}
		for key, value in zip(param_labels, params):
			param_dict[key] = value

		if not date in self.params_fit.index:
			fits = fitSubjValues(data, 'ev')[0].iloc[0]
			self.params_fit.loc[date, ['beta','lr_bias']] = np.array(fits[['w2','w3']])

		param_dict['beta'] = self.params_fit.loc[date]['beta']
		param_dict['lr_bias'] = self.params_fit.loc[date]['lr_bias']

		sim_results = self.simulate(data, param_dict, sim_choice=False, track_values=False)

		return -np.log(sim_results['likelihood']).sum()

class LRDecayRescorlaWagnerModel(RescorlaWagnerModel):
	def __init__(self, tau=0.01, **kwargs):
		super().__init__(**kwargs)

		self.params_init['tau'] = tau
		self.bounds['tau'] = (0,None)
		self.params_fit['tau'] = []

	def simulate(self, data, params=None, sim_choice=True, track_values=True, merge_data=False):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		sim_choice: (bool) if True, simulate choices as well as hidden states;
					if False, simulate hidden states only and calculate likelihood of actual (non-simulated) choices
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		if params is None:
			set_params = True
			# params = self.params_fit
		else:
			set_params = False

		sim_results = pd.DataFrame()
		if track_values:
			cols = ['value1','value2','value3','value4','value5','value6','value7','value8','value9','choice']
		else:
			cols = ['choice']
		if sim_choice:
			cols.append('outcome')
		else:
			cols.append('likelihood')

		# iterate through valid blocks of each session
		dates = data['date'].unique()
		for date in dates:
			block = data[data['date']==date]

			if set_params:
				try:
					params = self.params_fit.loc[date].to_dict()
				except KeyError:
					params = None
					raise

			# initialize state values for block
			block_sim = np.full((block.shape[0],len(cols)), np.nan)
			values = np.zeros(9)

			if not params is None:
				for ii in range(block_sim.shape[0]):
					trial = block.iloc[ii] # trial data

					if track_values:
						block_sim[ii, :9] = values # record values at start of trial

					idx_l = trial['left_image'] - 1
					if not np.isnan(idx_l):
						idx_l = int(idx_l)

					idx_r = trial['right_image'] - 1
					if not np.isnan(idx_r):
						idx_r = int(idx_r)

					# simulated probability of choosing left
					if (not np.isnan(idx_l)) and (not np.isnan(idx_r)):
						p_l = softmax(values[idx_l], values[idx_r], params['beta'], params['lr_bias'])
					elif sim_choice:
						p_l = float(np.isnan(idx_r))
					else:
						p_l = np.nan

					if sim_choice:
						# simulate choice and reward outcome
						if stats.bernoulli.rvs(p_l):
							choice = -1
							chosen = idx_l # chosen image index
							p_rwd = self.levels2prob[trial['left_prob_level']] # probability of reward
							amnt_rwd = self.levels2amnt[trial['left_amnt_level']] # amount of reward
						else:
							choice = 1
							chosen = idx_r
							p_rwd = self.levels2prob[trial['right_prob_level']]
							amnt_rwd = self.levels2amnt[trial['right_amnt_level']]

						if trial['lever'] == choice:
							outcome = amnt_rwd * trial['if_reward']
						else:
							outcome = amnt_rwd * stats.bernoulli.rvs(p_rwd)

						block_sim[ii, -2] = choice # record choice
						block_sim[ii, -1] = outcome # record outcome

					else:
						# compute single-trial choice likelihood
						block_sim[ii, -2] = trial['lever']

						if trial['lever'] == -1:
							block_sim[ii, -1] = p_l # likelihood of left choice, assuming Bernoulli Distribution
							chosen = idx_l # chosen image index
							outcome = self.levels2amnt[trial['left_amnt_level']] * trial['if_reward']
						elif trial['lever'] == 1:
							block_sim[ii, -1] = 1 - p_l # likelihood of right choice, assuming Bernoulli Distribution
							chosen = idx_r
							outcome = self.levels2amnt[trial['right_amnt_level']] * trial['if_reward']

					alpha = params['alpha']*np.exp(-params['tau']*ii)
					if alpha < 1e-3:
						alpha = 0
					values[chosen] = self.learningRule(values[chosen], alpha, outcome) # value update

			block_sim = pd.DataFrame(block_sim, index=block.index, columns=cols)
			sim_results = sim_results.append(block_sim)

		if merge_data:
			sim_results = pd.concat([data, sim_results], axis=1, sort=False)

		return sim_results

class WinStayLoseShiftRescorlaWagnerModel(RescorlaWagnerModel):
	def __init__(self, wsls_bias=0.1, **kwargs):
		super().__init__(**kwargs)

		self.params_init['wsls_bias'] = wsls_bias
		self.bounds['wsls_bias'] = (0,None)
		self.params_fit['wsls_bias'] = []

	def simulate(self, data, params=None, sim_choice=True, track_values=True, merge_data=False):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		sim_choice: (bool) if True, simulate choices as well as hidden states;
					if False, simulate hidden states only and calculate likelihood of actual (non-simulated) choices
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		data = data.replace({'if_reward': {np.nan: 0}}) # replace if_reward nan values with 0s

		if params is None:
			set_params = True
			# params = self.params_fit
		else:
			set_params = False

		sim_results = pd.DataFrame()
		if track_values:
			cols = ['value1','value2','value3','value4','value5','value6','value7','value8','value9','choice']
		else:
			cols = ['choice']
		if sim_choice:
			cols.append('outcome')
		else:
			cols.append('likelihood')

		# iterate through valid blocks of each session
		dates = data['date'].unique()
		for date in dates:
			block = data[data['date']==date]

			if set_params:
				try:
					params = self.params_fit.loc[date].to_dict()
				except KeyError:
					params = None
					raise

			# initialize state values for block
			block_sim = np.full((block.shape[0],len(cols)), np.nan)
			values = np.zeros(9)

			if not params is None:
				for ii in range(block_sim.shape[0]):
					trial = block.iloc[ii] # trial data

					if track_values:
						block_sim[ii, :9] = values # record values at start of trial

					idx_l = trial['left_image'] - 1
					if not np.isnan(idx_l):
						idx_l = int(idx_l)

					idx_r = trial['right_image'] - 1
					if not np.isnan(idx_r):
						idx_r = int(idx_r)

					# simulated probability of choosing left
					if (not np.isnan(idx_l)) and (not np.isnan(idx_r)):
						p_l = softmax(values[idx_l], values[idx_r], params['beta'], params['lr_bias'])
					elif sim_choice:
						p_l = float(np.isnan(idx_r))
					else:
						p_l = np.nan

					if sim_choice:
						# simulate choice and reward outcome
						if stats.bernoulli.rvs(p_l):
							choice = -1
							chosen = idx_l # chosen image index
							p_rwd = self.levels2prob[trial['left_prob_level']] # probability of reward
							amnt_rwd = self.levels2amnt[trial['left_amnt_level']] # amount of reward
						else:
							choice = 1
							chosen = idx_r
							p_rwd = self.levels2prob[trial['right_prob_level']]
							amnt_rwd = self.levels2amnt[trial['right_amnt_level']]

						if trial['lever'] == choice:
							outcome = amnt_rwd * trial['if_reward']
						else:
							outcome = amnt_rwd * stats.bernoulli.rvs(p_rwd)

						block_sim[ii, -2] = choice # record choice
						block_sim[ii, -1] = outcome # record outcome

					else:
						# compute single-trial choice likelihood
						block_sim[ii, -2] = trial['lever']

						if trial['lever'] == -1:
							block_sim[ii, -1] = p_l # likelihood of left choice, assuming Bernoulli Distribution
							chosen = idx_l # chosen image index
							outcome = self.levels2amnt[trial['left_amnt_level']] * trial['if_reward']
						elif trial['lever'] == 1:
							block_sim[ii, -1] = 1 - p_l # likelihood of right choice, assuming Bernoulli Distribution
							chosen = idx_r
							outcome = self.levels2amnt[trial['right_amnt_level']] * trial['if_reward']

					values[chosen] = self.learningRule(values[chosen], params['alpha'], outcome) # value update

			block_sim = pd.DataFrame(block_sim, index=block.index, columns=cols)
			sim_results = sim_results.append(block_sim)

		if merge_data:
			sim_results = pd.concat([data, sim_results], axis=1, sort=False)

		return sim_results
