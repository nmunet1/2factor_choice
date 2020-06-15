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
		self.sim_results = None

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
		values = np.ones((lever.size,9))*0.12
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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
			results = data[bhv.isvalid(data, forced=True, sets='new')]
		else:
			results = pd.DataFrame()

		for n in range(n_iter):
			if n % 10 == 0 and verbose:
				print('simulation', n)
			res_n = self.simulate(data, mode='sim', merge_data=False)
			# res_n['sim'] = n
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

		return sim_results, self.aic

	def assessModel(self, data, aic=True, bic=True, n_trials=None, selection='first'):
		'''
		Returns DataFrame of AICs for each session
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')]
		if not n_trials is None:
			if selection == 'first':
				data = data.groupby('date').head(n_trials)
			elif selection == 'last':
				data = data.groupby('date').tail(n_trials)
			else:
				raise ValueError

		dates = data['date'].unique()
		results = pd.DataFrame()

		for date in dates:
			sess_data = data[data['date']==date]
			sess_est = self.simulate(sess_data, mode='est')

			k = len(self.params_init) # number of free params
			ll = sess_est['log-likelihood'].sum() # max log-likelihood

			if aic:
				results.loc[date,'AIC'] = 2*k - 2*ll
			if bic:
				n = sess_est.shape[0] # number of observations
				results.loc[date,'BIC'] = k*np.log(n) - 2*ll

		return results

	def plotValueLearning(self, date, mode='sim'):
		sess_sims = self.sim_results[self.sim_results['date']==date]
		if mode == 'sim':
			data = sess_sims
		elif mode == 'est':
			data = self.simulate(sess_sims, mode='est')
		else:
			raise ValueError

		ev = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1]) * \
			np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])

		for img in range(1,10):
			plt.figure()
			plt.plot(data['value%i' % img].to_numpy())
			plt.gca().axhline(ev[img-1],ls='--',color='k')
			plt.xlabel('Trial')
			plt.ylabel('Subjective Value')
			if mode == 'sim':
				plt.title('Image %i Simulation' % img)
			else:
				plt.title('Image %i Estimate' % img)

	def plotValueDiff(self, date):
		sess_sims = self.sim_results[self.sim_results['date']==date]

		colors = ['blue','grey','red']
		prob_labels = ['High','Medium','Low']

		est = self.simulate(sess_sims, mode='est')

		for ii, prob_group in enumerate([[1,2,3],[4,5,6],[7,8,9]]):
			plt.figure()
			for img in prob_group:
				value_diff = sess_sims['value%i' % img]
				value_diff = value_diff.apply(lambda x: est['value%i' % img] - x)
				n_trials, n_sims = value_diff.shape

				value_diff = value_diff.to_numpy().reshape(n_trials*n_sims, order='F')
				trials = np.tile(np.arange(1,n_trials+1), n_sims)

				sns.lineplot(trials, value_diff, color=colors[(img%3)-1])
			plt.xlabel('Trial')
			plt.ylabel('Estimated Value - Simulation Value')
			plt.legend(['High Reward','Medium Reward','Low Reward'])
			plt.title('%s Probability Images' % prob_labels[ii])

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

			softmax_fits = fitSubjValues(block, model='ev', min_type=min_type).loc[date,['beta','lr_bias']].to_dict()
			self.params_fit.loc[date, 'beta'] = softmax_fits['beta']
			self.params_fit.loc[date, 'lr_bias'] = softmax_fits['lr_bias']

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

class AlphaDecayRWModel(RescorlaWagnerModel):
	def __init__(self, tau=0.001, **kwargs):
		super().__init__(**kwargs)

		self.params_init['tau'] = tau
		self.bounds['tau'] = (0,None)
		self.params_fit['tau'] = np.nan

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
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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

class WinStayLoseShiftRWModel(RescorlaWagnerModel):
	def __init__(self, wsls_bias=0.1, **kwargs):
		super().__init__(**kwargs)

		self.params_init['wsls_bias'] = wsls_bias
		self.bounds['wsls_bias'] = (0,None)
		self.params_fit['wsls_bias'] = np.nan

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
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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

class ChoiceKernelRWModel(RescorlaWagnerModel):
	def __init__(self, alpha_choice=0.01, choice_bias=0.1, **kwargs):
		super().__init__(**kwargs)

		self.params_init.update({'alpha_choice': alpha_choice, 'choice_bias': choice_bias})

		self.bounds['alpha_choice'] = (0,1)
		self.bounds['choice_bias'] = (0,None)

		self.params_fit['alpha_choice'] = np.nan
		self.params_fit['choice_bias'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha=0.01, beta=-0.1, lr_bias=0.1, alpha_choice=0.01, choice_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		choice_kernel = np.zeros(9)
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

		err_ct = 0
		for ii in range(lever.size):
			# simulated probability of choosing left
			if np.isnan(img_l[ii]):
				idx_l = None
				q_l = -np.inf
				c_l = -np.inf
			else:
				idx_l = int(img_l[ii])-1
				q_l = values[ii, idx_l]
				c_l = choice_kernel[idx_l]

			if np.isnan(img_r[ii]):
				idx_r = None
				q_r = -np.inf
				c_r = -np.inf
			else:
				idx_r = int(img_r[ii])-1
				q_r = values[ii, idx_r]
				c_r = choice_kernel[idx_r]

			p_l = softmax(q_l, q_r, beta, c_r-c_l)

			if mode == 'sim':
				# simulate choice and reward outcome
				if stats.bernoulli.rvs(p_l):
					choice = -1
					chosen = idx_l # chosen image index
					unchosen = idx_r
				else:
					choice = 1
					chosen = idx_r
					unchosen = idx_l

				if lever[ii] == choice:
					outcome = amnt_map[chosen] * reward[ii]
				else:
					outcome = amnt_map[chosen] * stats.bernoulli.rvs(prob_map[chosen])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = idx_l
					unchosen = idx_r
				else:
					result[ii] = np.log(1-p_l)
					chosen = idx_r
					unchosen = idx_l
				
				outcome = amnt_map[chosen] * reward[ii]

			# value update
			if ii+1 < lever.size:
				values[ii+1,:] = values[ii,:]
				values[ii+1, chosen] = self.learningRule(values[ii+1, chosen], alpha, outcome)
				choice_kernel[chosen] = self.learningRule(choice_kernel[chosen], alpha_choice, choice_bias)
				if not unchosen is None:
					choice_kernel[unchosen] = self.learningRule(choice_kernel[unchosen], alpha_choice, 0)

		return result, values

class AlphaFreeAlphaForcedRWModel(RescorlaWagnerModel):
	def __init__(self, alpha_free=0.01, alpha_forced=0.01, **kwargs):
		super().__init__(**kwargs)

		del self.params_init['alpha']
		self.params_init.update({'alpha_free': alpha_free, 'alpha_forced': alpha_forced})

		del self.bounds['alpha']
		self.bounds.update({'alpha_free': (0,1), 'alpha_forced': (0,1)})

		self.params_fit = self.params_fit.drop('alpha',1)
		self.params_fit['alpha_free'] = np.nan
		self.params_fit['alpha_forced'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha_free=0.01, alpha_forced=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
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
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_free, outcome)

		return result, values

class AlphaAmountRWModel(RescorlaWagnerModel):
	def __init__(self, alpha_low=0.01, alpha_med=0.01, alpha_high=0.01, **kwargs):
		super().__init__(**kwargs)

		del self.params_init['alpha']
		self.params_init.update({'alpha_low': alpha_low, 'alpha_med': alpha_med, 'alpha_high': alpha_high})

		del self.bounds['alpha']
		self.bounds.update({'alpha_low': (0,1), 'alpha_med': (0,1), 'alpha_high': (0,1)})

		self.params_fit = self.params_fit.drop('alpha',1)
		self.params_fit['alpha_low'] = np.nan
		self.params_fit['alpha_med'] = np.nan
		self.params_fit['alpha_high'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha_low=0.01, alpha_med=0.01, alpha_high=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
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
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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
				if chosen % 3 == 1:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_high, outcome)
				elif chosen % 3 == 2:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_med, outcome)
				elif chosen % 3 == 0:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_low, outcome)

		return result, values

class AlphaOutcomeRWModel(RescorlaWagnerModel):
	def __init__(self, alpha_win=0.01, alpha_lose=0.01, **kwargs):
		super().__init__(**kwargs)

		del self.params_init['alpha']
		self.params_init.update({'alpha_win': alpha_win, 'alpha_lose': alpha_lose})

		del self.bounds['alpha']
		self.bounds.update({'alpha_win': (0,1), 'alpha_lose': (0,1)})

		self.params_fit = self.params_fit.drop('alpha',1)
		self.params_fit['alpha_win'] = np.nan
		self.params_fit['alpha_lose'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha_win=0.01, alpha_lose=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
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
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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
				if outcome == 0:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_lose, outcome)
				else:
					values[ii+1, chosen-1] = self.learningRule(values[ii+1, chosen-1], alpha_win, outcome)

		return result, values

class AmountLearningRWModel(RescorlaWagnerModel):
	def __init__(self, alpha_prob=0.01, alpha_amnt=0.01, **kwargs):
		super().__init__(**kwargs)

		del self.params_init['alpha']
		self.params_init.update({'alpha_prob': alpha_prob, 'alpha_amnt': alpha_amnt})

		del self.bounds['alpha']
		self.bounds.update({'alpha_prob': (0,1), 'alpha_amnt': (0,1)})

		self.params_fit = self.params_fit.drop('alpha',1)
		self.params_fit['alpha_prob'] = np.nan
		self.params_fit['alpha_amnt'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha_prob=0.01, alpha_amnt=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.zeros((lever.size,9))
		probs = np.zeros(9)
		amnts = np.zeros(9)
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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
				if outcome == 0:
					probs[chosen-1] = self.learningRule(probs[chosen-1], alpha_prob, 0)
				else:
					probs[chosen-1] = self.learningRule(probs[chosen-1], alpha_prob, 1)
					amnts[chosen-1] = self.learningRule(amnts[chosen-1], alpha_amnt, outcome)

				values[ii+1,:] = probs*amnts

		return result, values
