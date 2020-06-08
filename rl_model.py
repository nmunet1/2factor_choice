import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, basinhopping
import seaborn as sns
import time

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
		self.bounds = {'alpha': (0,1), 'beta': (None,0), 'lr_bias': (None,None)} # parameter bounds
		#self.params_fit = self.params_init.copy() # fitted parameter values
		self.params_fit = pd.DataFrame(columns=['alpha','beta','lr_bias'])

		self.aic = None # Akaike Information Criterion of best fit model
		self.sims = None # Simulation results from bootstrapped simulations

		self.levels2prob = {1: 0.7, 2: 0.4, 3: 0.1} # conversion from probability levels to probabilities of reward
		self.levels2amnt = {1: 0.5, 2: 0.3, 3: 0.1} # conversion from amount levels to reward amounts in L

	def data2numpy(self, data):
		data = data[bhv.isvalid(data, forced=True, sets='new')]

		img_l = data['left_image'].to_numpy()
		img_r = data['right_image'].to_numpy()
		lever = data['lever'].to_numpy()
		reward = data['if_reward'].replace({np.nan:0}).to_numpy()

		return img_l, img_r, lever, reward

	def learningRule(self, value_old, rate, feedback):
		'''
		Basic incremental learning rule
		
		value_old: 	(float) value to be updated
		rate:		(float) learning rate
		feedback:	(float) learning signal
		'''
		return value_old + rate * (feedback - value_old)

	def simulate(self, data, mode='sim', merge_data=False):
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		sim_results = pd.DataFrame()

		cols = ['value1','value2','value3','value4','value5','value6','value7','value8','value9']
		if mode == 'sim':
			cols += ['sim_choice','reward']
		elif mode == 'est':
			cols.append('log-likelihood')
		else:
			raise ValueError

		# iterate through valid blocks of each session
		for date in data['date'].unique():
			block = data[data['date']==date]
			params = self.params_fit.loc[date].to_dict()

			sim_res, values = self.simSess(*self.data2numpy(block), mode=mode, **params)

			block_sim = pd.DataFrame(np.concatenate((values, sim_res), axis=1), \
				index=block.index, columns=cols)
			sim_results = sim_results.append(block_sim)

		if merge_data:
			sim_results = pd.concat((data, sim_results), axis=1, sort=False)

		return sim_results

	def simSess(self, img_l, img_r, lever, reward, alpha=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.4, 0.1, 0.7, 0.4, 0.1, 0.7, 0.4, 0.1])

		err_ct = 0
		for ii in range(lever.size):
			# simulated probability of choosing left
			if np.isnan(img_l[ii]):
				q_l = -np.inf
			else:
				q_l = values[ii, int(img_l[ii])-1]

			if np.isnan(img_r[ii]):
				q_r = -np.inf
			else:
				q_r = values[ii, int(img_r[ii])-1]

			p_l = softmax(q_l, q_r, beta, lr_bias)

			if mode == 'sim':
				# simulate choice and reward outcome
				if stats.bernoulli.rvs(p_l):
					choice = -1
					chosen = int(img_l[ii]) # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])

				if lever[ii] == choice:
					outcome = amnt_map[chosen-1] * reward[ii]
				else:
					outcome = amnt_map[chosen-1] * stats.bernoulli.rvs(prob_map[chosen-1])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])
				
				outcome = amnt_map[chosen-1] * reward[ii]

			# value update
			if ii+1 < lever.size:
				values[ii+1,:] = values[ii,:]
				values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha, outcome)

		return result, values

	def bootstrap(self, data, n_iter=100, merge_data=True, verbose=False):
		if merge_data:
			results = data.copy()
		else:
			results = pd.DataFrame()

		for n in range(n_iter):
			if n % 10 == 0 and verbose:
				print('simulation', n)
			res_n = self.simulate(data, mode='sim', merge_data=True)
			res_n['sim'] = n
			results = pd.concat([results, res_n], axis=1)
		self.sim_results = results

		return results

	def negLogLikelihood(self, params, param_labels, img_l, img_r, lever, reward, fixed_params={}):
		'''
		Calculate negative log-likelihood of choice behavior given model parameters
		
		params:			(list) free parameters
		param_labels:	(list) parameter labels
		block: 			(DataFrame) block data
		'''
		param_dict = fixed_params
		for key, value in zip(param_labels, params):
			param_dict[key] = value

		return -self.simSess(img_l, img_r, lever, reward, mode='est', **param_dict)[0].sum()

	def fit(self, data, params_init=None, verbose=False, min_type='local', **kwargs):
		'''
		Fit model free parameters

		data: 			(DataFrame) experimental dataset
		params_init: 	(dict) initial guess for free parameter values
		verbose: 		(bool) if True, display optimization results
		'''
		t0 = time.time()
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		dates = data['date'].unique()

		for date in dates:
			block = data[data['date']==date]

			if params_init is None:
				params_init = self.params_init

			param_labels = list(params_init.keys())
			params = [params_init[label] for label in param_labels]
			bounds = [self.bounds[label] for label in param_labels]

			# fit data by minimizing negative log-likelihood of choice behavior given model and parameters
			if min_type == 'local':
				opt = minimize(self.negLogLikelihood, params, args=(param_labels, *self.data2numpy(block)), \
					tol=1e-4, bounds=bounds, **kwargs)

				if verbose or not opt.success:
					print(date, 'fitting failed:')
					print(opt, '\n')

				if opt.success:
					self.params_fit.loc[date, param_labels] = list(opt.x)

			elif min_type == 'global':
				opt = basinhopping(self.negLogLikelihood, params, minimizer_kwargs={'method':'L-BFGS-B', \
					'args':(param_labels, *self.data2numpy(block)), 'tol':1e-4, 'bounds':bounds})

				if verbose or not opt.lowest_optimization_result.success:
					print(date, 'fitting failed:')
					print(opt, '\n')

				if opt.lowest_optimization_result.success:
					self.params_fit.loc[date, param_labels] = list(opt.x)
					
			else:
				raise ValueError

		sim_results = self.simulate(data, mode='est', merge_data=True)
		negLL = -sim_results['log-likelihood'].sum()
		self.aic = 2*len(params)*len(dates) + 2*negLL # Update Akaike Information Criterion

		print('Fit time:', time.time()-t0)

		return sim_results, self.aic, opt

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

	def plotValueDiff(self, data, date, redo_sim=False, n_iter=100):
		if redo_sim or (self.sims is None or not any(self.sims['date']==date)):
			self.sims = self.bootstrap(data, n_iter=n_iter)
		sess_sims = self.sims[self.sims['date']==date]

		colors = ['blue','blue','blue','grey','grey','grey','red','red','red']
		ls = ['-','--',':','-','--',':','-','--',':']


		sess_data = data[data['date']==date]
		est = self.simulate(sess_data, sim_choice=False)

		for img_i in range(1,10):
			value_diff = sess_sims['value%i' % img_i]
			value_diff = value_diff.apply(lambda x: est['value%i' % img_i] - x)
			orig_size = value_diff.shape

			value_diff = value_diff.to_numpy().T.reshape(np.prod(orig_size))
			trials = np.tile(np.arange(orig_size[0])+1, orig_size[1])

			sns.lineplot(trials, value_diff, color=colors[img_i-1], ls=ls[img_i-1])
		plt.xlabel('Trial')
		plt.ylabel('Estimated Value - Simulation Value')

class FixedSoftmaxRescorlaWagnerModel(RescorlaWagnerModel):
	def __init__(self, alpha=0.01):
		super().__init__()

		self.params_init = {'alpha': alpha}
		self.bounds = {'alpha': (0,1)}

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
			block = data[data['date']==date]

			if params_init is None:
				params_init = self.params_init

			param_labels = list(params_init.keys())
			params = [params_init[label] for label in param_labels]
			bounds = [self.bounds[label] for label in param_labels]

			softmax_fits = fitSubjValues(block, model='ev', min_type=min_type)[0].iloc[0][['beta', 'lr_bias']].to_dict()

			# fit data by minimizing negative log-likelihood of choice behavior given model and parameters
			if min_type == 'local':
				opt = minimize(self.negLogLikelihood, params, args=(param_labels, *self.data2numpy(block), softmax_fits), \
					tol=1e-4, bounds=bounds, **kwargs)

				if verbose or not opt.success:
					print(date, 'fitting failed:')
					print(opt, '\n')

				if opt.success:
					self.params_fit.loc[date, param_labels] = list(opt.x)

			elif min_type == 'global':
				opt = basinhopping(self.negLogLikelihood, params, minimizer_kwargs={'method':'L-BFGS-B', \
					'args':(param_labels, *self.data2numpy(block), softmax_fits), 'tol':1e-4, 'bounds':bounds})

				if verbose or not opt.lowest_optimization_result.success:
					print(date, 'fitting failed:')
					print(opt, '\n')

				if opt.lowest_optimization_result.success:
					self.params_fit.loc[date, param_labels] = list(opt.x)

			else:
				raise ValueError

		sim_results = self.simulate(data, mode='est', merge_data=True)
		negLL = -sim_results['log-likelihood'].sum()
		self.aic = 2*len(params)*len(dates) + 2*negLL # Update Akaike Information Criterion

		return sim_results, self.aic

class FSAlphaDecayRWModel(FixedSoftmaxRescorlaWagnerModel):
	def __init__(self, tau=0.001, **kwargs):
		super().__init__(**kwargs)

		self.params_init['tau'] = tau
		self.bounds['tau'] = (0,None)
		self.params_fit['tau'] = []

	def simSess(self, img_l, img_r, lever, reward, alpha=0.01, tau=0.001, beta=-0.1, lr_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.4, 0.1, 0.7, 0.4, 0.1, 0.7, 0.4, 0.1])

		err_ct = 0
		for ii in range(lever.size):
			# simulated probability of choosing left
			if np.isnan(img_l[ii]):
				q_l = -np.inf
			else:
				q_l = values[ii, int(img_l[ii])-1]

			if np.isnan(img_r[ii]):
				q_r = -np.inf
			else:
				q_r = values[ii, int(img_r[ii])-1]

			p_l = softmax(q_l, q_r, beta, lr_bias)

			if mode == 'sim':
				# simulate choice and reward outcome
				if stats.bernoulli.rvs(p_l):
					choice = -1
					chosen = int(img_l[ii]) # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])

				if lever[ii] == choice:
					outcome = amnt_map[chosen-1] * reward[ii]
				else:
					outcome = amnt_map[chosen-1] * stats.bernoulli.rvs(prob_map[chosen-1])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])
				
				outcome = amnt_map[chosen-1] * reward[ii]

			# value update
			if ii+1 < lever.size:
				alpha_dis = alpha*np.exp(-tau*ii)
				values[ii+1,:] = values[ii,:]
				values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_dis, outcome)

		return result, values

class FSWinStayLoseShiftRWModel(FixedSoftmaxRescorlaWagnerModel):
	def __init__(self, wsls_bias=0.1, **kwargs):
		super().__init__(**kwargs)

		self.params_init['wsls_bias'] = wsls_bias
		self.bounds['wsls_bias'] = (0,None)
		self.params_fit['wsls_bias'] = []

	def simSess(self, img_l, img_r, lever, reward, alpha=0.01, beta=-0.1, lr_bias=0.1, wsls_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		last_outcome = np.ones(9)*0.5
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.4, 0.1, 0.7, 0.4, 0.1, 0.7, 0.4, 0.1])

		err_ct = 0
		for ii in range(lever.size):
			# simulated probability of choosing left
			if np.isnan(img_l[ii]):
				q_l = -np.inf
			else:
				q_l = values[ii, int(img_l[ii])-1]

			if np.isnan(img_r[ii]):
				q_r = -np.inf
			else:
				q_r = values[ii, int(img_r[ii])-1]

			bias = lr_bias
			if not (np.isnan(img_l[ii]) or np.isnan(img_r[ii])):
				if last_outcome[int(img_l[ii])-1] > last_outcome[int(img_r[ii])-1]:
					bias -= wsls_bias
				elif last_outcome[int(img_l[ii])-1] < last_outcome[int(img_r[ii])-1]:
					bias += wsls_bias

			p_l = softmax(q_l, q_r, beta, bias)

			if mode == 'sim':
				# simulate choice and reward outcome
				if stats.bernoulli.rvs(p_l):
					choice = -1
					chosen = int(img_l[ii]) # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])

				if lever[ii] == choice:
					outcome = amnt_map[chosen-1] * reward[ii]
				else:
					outcome = amnt_map[chosen-1] * stats.bernoulli.rvs(prob_map[chosen-1])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])
				
				outcome = amnt_map[chosen-1] * reward[ii]

			# value update
			if ii+1 < lever.size:
				values[ii+1,:] = values[ii,:]
				values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha, outcome)
				last_outcome[chosen-1] = outcome > 0

		return result, values

class FSAlphaFixedAlphaForcedRWModel(FixedSoftmaxRescorlaWagnerModel):
	def __init__(self, alpha_fixed=0.01, alpha_forced=0.01, **kwargs):
		super().__init__(**kwargs)

		self.params_init = {'alpha_fixed': alpha_fixed, 'alpha_forced': alpha_forced}
		self.bounds = {'alpha_fixed': (0,1), 'alpha_forced': (0,1)}
		self.params_fit = pd.DataFrame(columns=['alpha_fixed','alpha_forced','beta','lr_bias'])

	def simSess(self, img_l, img_r, lever, reward, alpha_fixed=0.01, alpha_forced=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.4, 0.1, 0.7, 0.4, 0.1, 0.7, 0.4, 0.1])

		err_ct = 0
		for ii in range(lever.size):
			# simulated probability of choosing left
			if np.isnan(img_l[ii]):
				q_l = -np.inf
			else:
				q_l = values[ii, int(img_l[ii])-1]

			if np.isnan(img_r[ii]):
				q_r = -np.inf
			else:
				q_r = values[ii, int(img_r[ii])-1]

			p_l = softmax(q_l, q_r, beta, lr_bias)

			if mode == 'sim':
				# simulate choice and reward outcome
				if stats.bernoulli.rvs(p_l):
					choice = -1
					chosen = int(img_l[ii]) # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])

				if lever[ii] == choice:
					outcome = amnt_map[chosen-1] * reward[ii]
				else:
					outcome = amnt_map[chosen-1] * stats.bernoulli.rvs(prob_map[chosen-1])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])
				
				outcome = amnt_map[chosen-1] * reward[ii]

			# value update
			if ii+1 < lever.size:
				values[ii+1,:] = values[ii,:]
				if np.isnan(img_l[ii]) or np.isnan(img_r[ii]):
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_forced, outcome)
				else:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_fixed, outcome)

		return result, values
