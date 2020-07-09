import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, basinhopping, NonlinearConstraint
import seaborn as sns
import time

import bhv_analysis as bhv
from bhv_model import softmax, fitSubjValues

class RescorlaWagnerModel(object):
	'''
	Basic Rescorla-Wagner model with stochastic (softmax) choice and fixed left-right bias
	'''

	def __init__(self, alpha=0.01, beta=-0.1, lr_bias=0.1):
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
		self.data = None
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
		values = np.zeros((lever.size,9))
		if mode == 'sim':
			result = np.zeros((lever.size,2)) # row: [simulated choice, outcome]
		elif mode == 'est':
			result = np.zeros((lever.size,1)) # log-likelihoods

		amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
		prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])

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

				# if lever[ii] == choice:
				# 	outcome = amnt_map[chosen-1] * reward[ii]
				# else:
				# 	outcome = amnt_map[chosen-1] * stats.bernoulli.rvs(prob_map[chosen-1])
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

	def bootstrap(self, data, n_iter=100, verbose=False):
		data = data[bhv.isvalid(data, forced=True, sets='new')]
		results = pd.DataFrame()

		for n in range(n_iter):
			if n % 10 == 0 and verbose:
				print('simulation', n)
			sim_n = self.simulate(data, mode='sim', merge_data=False)
			sim_n.insert(0,'iter',n)
			results = pd.concat([results, sim_n], axis=0)

		self.data = data
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

	def fit(self, data, params_init=None, verbose=False, method='L-BFGS-B', min_type='local', \
			cost_fn = None, cons=(), **kwargs):
		'''
		Fit model free parameters

		data: 			(DataFrame) experimental dataset
		params_init: 	(dict) initial guess for free parameter values
		verbose: 		(bool) if True, display optimization results
		'''
		data = data[bhv.isvalid(data, forced=True, sets='new')] # filter out invalid trials and unwanted blocks
		dates = data['date'].unique()

		if cost_fn is None:
			cost_fn = self.negLogLikelihood

		for date in dates:
			block = data[data['date']==date]

			if params_init is None:
				params_init = self.params_init

			param_labels = list(params_init.keys())
			params = [params_init[label] for label in param_labels]
			bounds = [self.bounds[label] for label in param_labels]

			# fit data by minimizing negative log-likelihood of choice behavior given model and parameters
			if min_type == 'local':
				opt = minimize(cost_fn, params, args=(param_labels, *self.data2numpy(block)), \
					method=method, tol=1e-4, bounds=bounds, constraints=cons, **kwargs)

				if verbose or not opt.success:
					print(date, 'fitting failed:')
					print(opt, '\n')

				if opt.success:
					self.params_fit.loc[date, param_labels] = list(opt.x)

			elif min_type == 'global':
				opt = basinhopping(cost_fn, params, minimizer_kwargs={'method':method, \
					'args':(param_labels, *self.data2numpy(block)), 'tol':1e-4, 'bounds':bounds, \
					'constraints': cons})

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

	def plotEVLearning(self, date, win_size=50, min_trials=10, win_step=10):
		block_data = self.data[self.data['date']==date]
		block_data = block_data[bhv.isvalid(block_data, sets='all')]
		block_sims = self.sim_results.loc[block_data.index]

		left_amnt = block_data['left_amnt_level'].replace({1: 0.5, 2: 0.3, 3: 0.1})
		left_prob = block_data['left_prob_level'].replace({1: 0.7, 2: 0.4, 3: 0.1})
		right_amnt = block_data['right_amnt_level'].replace({1: 0.5, 2: 0.3, 3: 0.1})
		right_prob = block_data['right_prob_level'].replace({1: 0.7, 2: 0.4, 3: 0.1})

		optimal = (left_amnt*left_prob > right_amnt*right_prob).replace({True: -1.0, False: 1.0})

		sim_opt = block_sims.set_index('iter',append=True)['sim_choice'].unstack('iter')
		sim_opt = sim_opt.apply(lambda x: x == optimal, axis=0).reset_index(drop=True)
		sim_opt = sim_opt.rolling(window=win_size, min_periods=min_trials).mean()[win_step-1::win_step]

		real_opt = (block_data['lever'] == optimal).reset_index(drop=True)
		real_opt = real_opt.rolling(window=win_size, min_periods=min_trials).mean()[win_step-1::win_step]

		all_opt = pd.concat((real_opt, sim_opt.stack().reset_index('iter',drop=True)), \
			axis=1).set_axis(['Monkey C', 'Model'], axis=1)

		plt.figure()
		plt.axhline(0.5, color=[0.75, 0.75, 0.75], ls='--')
		plt.axhline(0.8, color=[0.75, 0.75, 0.75], ls='--')

		sns.lineplot(data=all_opt, palette={'Monkey C': 'grey', 'Model': 'red'}, dashes=False)

		plt.title(date)
		plt.xlabel('Free Trial')
		plt.ylabel('P(Optimal)')
		plt.ylim(0.4, 1.01)

	def plotValueLearning(self, date, x_label='Updates'):
		sess_data = self.data[self.data['date']==date]
		sims = self.sim_results.loc[sess_data.index]
		est = self.simulate(sess_data, mode='est')

		amnt_label = ['High','Medium','Low','High','Medium','Low','High','Medium','Low']
		prob_label = ['High','High','High','Medium','Medium','Medium','Low','Low','Low']
		x_max = 0

		ev = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1]) * \
			np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])

		for img in range(1,10):
			plt.figure()
			val_label = 'value%i' % img

			if x_label == 'Updates':
				for n in sims['iter'].unique():
					sim_vals = sims[(((sess_data['left_image'] == img) & (sims['sim_choice'] == -1)) | \
							((sess_data['right_image'] == img) & (sims['sim_choice'] == 1))) & \
							(sims['iter'] == n)][val_label]
					x = np.arange(sim_vals.size)
					# x_max = max(x_max, sim_vals.size)
					plt.plot(x, sim_vals, color=[0.8, 0.8, 1])

				est_vals = est[((sess_data['lever']==-1) & (sess_data['left_image']==img)) | ((sess_data['lever']==1) & (sess_data['right_image']==img))][val_label]
				x = np.arange(est_vals.size)
				x_max = max(x_max, est_vals.size)

			elif x_label == 'Trials':
				for n in sims['iter'].unique():
					sim_vals = sims[sims['iter']==n][val_label]
					x = np.arange(sim_vals.size)
					plt.plot(x, sim_vals, color=[0.8, 0.8, 1])

				est_vals = est[val_label]
				x = np.arange(est_vals.size)

			else:
				raise ValueError
			
			plt.plot(x, est_vals, color='k')
			plt.gca().axhline(ev[img-1],ls='--',color='k')
			if x_label == 'Updates':
				plt.xlim([-10, x_max+10])
			plt.xlabel(x_label)
			plt.ylabel('Subjective Value')
			plt.title('%s Probability, %s Reward' % (prob_label[img-1], amnt_label[img-1]))

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

class NonzeroInitValuesRWModel(RescorlaWagnerModel):
	def __init__(self, v0=0.001, **kwargs):
		super().__init__(**kwargs)

		self.params_init['v0'] = v0
		self.bounds['v0'] = (0,0.35)
		self.params_fit['v0'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, alpha=0.01, v0=0.001, beta=-0.1, lr_bias=0.1, mode='sim'):
		'''
		Estimates learned subjective values for each trial, given experimental data
		
		data: 		(DataFrame) experimental dataset
		params:		(dict) free parameters
		'''
		values = np.ones((lever.size,9))*v0
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

class BayesianModel(RescorlaWagnerModel):
	def __init__(self, beta=-0.1, lr_bias=0.1):
		super().__init__(beta=beta, lr_bias=lr_bias)

		del self.params_init['alpha']
		del self.bounds['alpha']
		self.params_fit = self.params_fit.drop('alpha',1)

	def simSess(self, img_l, img_r, lever, reward, beta=-0.1, lr_bias=0.1, mode='sim'):
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

		prior = np.ones((3,9))/3
		p_win = np.array([0.7, 0.4, 0.1])

		# expected reward contingencies for each image
		probs_est = np.ones(9)*prob_map.mean()
		amnts_est = np.ones(9)*amnt_map.mean()

		for ii in range(lever.size):
			values[ii,:] = amnts_est*probs_est

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
					chosen = int(img_l[ii])-1 # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])-1

				if lever[ii] == choice:
					outcome = amnt_map[chosen] * reward[ii]
				else:
					outcome = amnt_map[chosen] * stats.bernoulli.rvs(prob_map[chosen])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])-1
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])-1
				
				outcome = amnt_map[chosen] * reward[ii]

			# value update
			if ii+1 < lever.size:
				if outcome == 0:
					post = (1-p_win) * prior[:,chosen]
				else:
					amnts_est[chosen] = outcome
					post = p_win * prior[:,chosen]

				probs_est[chosen] = np.dot(p_win, post/post.sum())
				prior[:, chosen] = post

		return result, values

class LimitedMemoryBayesianModel(BayesianModel):
	def __init__(self, omega=0.01, **kwargs):
		super().__init__(**kwargs)
		self.params_init['omega'] = omega
		self.bounds['omega'] = (0,1)
		self.params_fit['omega'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, omega=0.01, beta=-0.1, lr_bias=0.1, mode='sim'):
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

		prob_prior = np.ones((3,9))/3
		p_win = np.array([0.7, 0.4, 0.1])

		amnt_prior = np.ones((3,9))/3
		amnts = np.array([0.5, 0.3, 0.1])

		# expected reward contingencies for each image
		probs_est = np.dot(p_win, prob_prior)
		amnts_est = np.dot(amnts, amnt_prior)

		for ii in range(lever.size):
			values[ii,:] = amnts_est*probs_est

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
					chosen = int(img_l[ii])-1 # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])-1

				if lever[ii] == choice:
					outcome = amnt_map[chosen] * reward[ii]
				else:
					outcome = amnt_map[chosen] * stats.bernoulli.rvs(prob_map[chosen])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])-1
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])-1
				
				outcome = amnt_map[chosen] * reward[ii]

			# value update
			if ii+1 < lever.size:
				if outcome == 0:
					post = (1-p_win) * prob_prior[:,chosen]**omega
				else:
					amnts_est[chosen] = outcome
					post = p_win * prob_prior[:,chosen]**omega

				probs_est[chosen] = np.dot(p_win, post/post.sum())
				prior[:, chosen] = post

		return result, values

class BiasedPriorsLMBModel(BayesianModel):
	def __init__(self, omega=0.01, rho=1, **kwargs):
		super().__init__(**kwargs)
		self.params_init['omega'] = omega
		self.bounds['omega'] = (0,1)
		self.params_fit['omega'] = np.nan

		self.params_init['rho'] = rho
		self.bounds['rho'] = (0,None)
		self.params_fit['rho'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, omega=0.01, rho=1, beta=-0.1, lr_bias=0.1, mode='sim'):
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

		prob_prior = np.ones((3,9))/3
		prob_prior[0,:] *= 2*rho/(rho+1)
		prob_prior[2,:] *= 2/(rho+1)
		p_win = np.array([0.7, 0.4, 0.1])

		amnt_prior = np.ones((3,9))/3
		amnts = np.array([0.5, 0.3, 0.1])

		# expected reward contingencies for each image
		probs_est = np.dot(p_win, prob_prior)
		amnts_est = np.dot(amnts, amnt_prior)

		for ii in range(lever.size):
			values[ii,:] = amnts_est*probs_est

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
					chosen = int(img_l[ii])-1 # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])-1

				if lever[ii] == choice:
					outcome = amnt_map[chosen] * reward[ii]
				else:
					outcome = amnt_map[chosen] * stats.bernoulli.rvs(prob_map[chosen])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])-1
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])-1
				
				outcome = amnt_map[chosen] * reward[ii]

			# value update
			if ii+1 < lever.size:
				if outcome == 0:
					post = (1-p_win) * prob_prior[:,chosen]**omega
				else:
					amnts_est[chosen] = outcome
					post = p_win * prob_prior[:,chosen]**omega

				probs_est[chosen] = np.dot(p_win, post/post.sum())
				prior[:, chosen] = post

		return result, values

class SigmoidalMemoryBayesianModel(BayesianModel):
	def __init__(self, tau=1, d=4, **kwargs):
		super().__init__(**kwargs)
		self.params_init.update({'tau': tau, 'd': d, 'w_lim':w_lim})
		self.bounds.update({'tau': (0, None), 'd': (1,None), 'w_lim':(0,0.95)})
		self.params_fit['tau'] = np.nan
		self.params_fit['d'] = np.nan

	def simSess(self, img_l, img_r, lever, reward, tau=1, d=4, beta=-0.1, lr_bias=0.1, mode='sim'):
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

		# expected amount of reward for each image
		ll_history = [[[] for c in range(9)] for r in range(3)]
		probs_est = np.ones(9)*prob_map.mean()
		amnts_est = np.ones(9)*amnt_map.mean()

		for ii in range(lever.size):
			values[ii,:] = amnts_est*probs_est

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
					chosen = int(img_l[ii])-1 # chosen image index
				else:
					choice = 1
					chosen = int(img_r[ii])-1

				if lever[ii] == choice:
					outcome = amnt_map[chosen] * reward[ii]
				else:
					outcome = amnt_map[chosen] * stats.bernoulli.rvs(prob_map[chosen])

				result[ii,:] = [choice, outcome]

			else:
				# compute single-trial choice likelihood
				if lever[ii] == -1:
					result[ii] = np.log(p_l)
					chosen = int(img_l[ii])-1
				else:
					result[ii] = np.log(1-p_l)
					chosen = int(img_r[ii])-1
				
				outcome = amnt_map[chosen] * reward[ii]

			# value update
			if ii+1 < lever.size:
				if outcome == 0:
					ll_history[0][chosen].append(0.3)
					ll_history[1][chosen].append(0.6)
					ll_history[2][chosen].append(0.9)
				else:
					ll_history[0][chosen].append(0.7)
					ll_history[1][chosen].append(0.4)
					ll_history[2][chosen].append(0.1)
					amnts_est[chosen] = outcome

				t = np.flip(np.arange(len(ll_history[0][chosen])))
				weights = (1+np.exp(tau*(t-d)))**-1

				ll_high = (np.array(ll_history[0][chosen])**weights).prod()
				ll_med = (np.array(ll_history[1][chosen])**weights).prod()
				ll_low = (np.array(ll_history[2][chosen])**weights).prod()

				probs_est[chosen] = (0.7*ll_high + 0.4*ll_med + 0.1*ll_low) / (ll_high + ll_med + ll_low)

		return result, values

	def smallMemoryBiasedNegLL(self, params, param_labels, *args):
		d = params[param_labels.index('d')]
		return self.negLogLikelihood(params, param_labels, *args) + 5*(d-1)
    
	def fit(self, data, **kwargs):
		lim_upper = {'type': 'ineq', 'fun': lambda x: ((1+np.exp(-x[2]*x[3]))**-1) - 0.95}
		return super().fit(data, method='SLSQP', cost_fn = self.smallMemoryBiasedNegLL, \
			cons=(lim_upper), **kwargs)
