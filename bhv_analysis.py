import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.io import loadmat
import seaborn as sns
import bhv_model

def loadData(subject='Chap', filt_sess=True):
	'''
	Reads .mat files containing single-session data into a multi-session pandas DataFrame
	'''
	root = './Data/%s' % subject
	if subject == 'Chap':
		excluded_dates = ['2019-08-02','2019-08-15','2019-08-18','2019-08-20','2019-08-23','2019-08-25','2019-08-27', \
			'2019-08-29','2019-09-02','2019-09-05','2019-09-08','2019-09-09','2019-09-10','2019-09-11','2019-09-12', \
			'2019-09-16','2019-11-19']
		folders = ['bhv_ABA', 'bhv_allB']
	elif subject == 'Tango':
		excluded_dates = []
		folders = ['Training']
	else:
		raise ValueError
	
	data = pd.DataFrame()
	
	for folder in folders:
		path = os.path.join(root,folder)
		with os.scandir(path) as it:
			for entry in it:
				if entry.name.endswith('.mat') and entry.is_file():
					date = entry.name[-25:-15] # get session date from file name
					if not filt_sess or not date in excluded_dates:
						# session type: ABA or B (image set structure)
						if folder == 'bhv_ABA':
							sess_type = 'ABA'
						else:
							sess_type = 'B'
							
						mat_data = loadmat(entry.path) # dictionary of all data values and labels

						# start with behavioral timeseries data and labels
						bhv_labels = [h[0] for h in mat_data['bhv_headers'].flatten()]
						sess_data = pd.DataFrame(mat_data['bhv_info'],columns=bhv_labels)

						sess_data.insert(0,'date',date) # add session date
						sess_data.insert(1,'sesstype',sess_type) # add session type

						if subject == 'Chap':
							# add fitted free parameters from discounted value and choice functions
							for w_name, w_fit in zip(mat_data['w_names'].flatten(), mat_data['w_fit'].flatten()):
								sess_data.insert(sess_data.shape[1], w_name[0][:2], w_fit)
						
						data = pd.concat([data,sess_data],ignore_index=True) # join session data with master dataset
					
	return data.sort_values(by=['date', 'trial']).reset_index(drop=True)

def fakeData(dates, images, ntrials=800):
	img2prob_level = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3}
	img2amnt_level = {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:1, 8:2, 9:3}
	img2prob = {1:0.7, 2:0.7, 3:0.7, 4:0.4, 5:0.4, 6:0.4, 7:0.1, 8:0.1, 9:0.1}

	fake_data = pd.DataFrame()
	for date in dates:
		trial = np.arange(ntrials)+1

		trialtype = np.zeros(ntrials)

		left_prob_level = np.full(ntrials, np.nan)
		right_prob_level = np.full(ntrials, np.nan)
		left_amnt_level = np.full(ntrials, np.nan)
		right_amnt_level = np.full(ntrials, np.nan)

		left_image = np.full(ntrials, np.nan)
		right_image = np.full(ntrials, np.nan)

		lever = np.zeros(ntrials)
		if_reward = np.zeros(ntrials)

		for ii in range(800):
			a, b = np.random.choice(images, 2, replace=False)

			if stats.bernoulli.rvs(0.67):
				trialtype[ii] = 2

				left_image[ii] = a
				left_prob_level[ii] = img2prob_level[a]
				left_amnt_level[ii] = img2amnt_level[a]

				right_image[ii] = b
				right_prob_level[ii] = img2prob_level[b]
				right_amnt_level[ii] = img2amnt_level[b]

				if stats.bernoulli.rvs(0.5):
					lever[ii] = -1
					if_reward[ii] = stats.bernoulli.rvs(img2prob[a])
				else:
					lever[ii] = 1
					if_reward[ii] = stats.bernoulli.rvs(img2prob[b])

			else:
				trialtype[ii] = 1
				if_reward[ii] = stats.bernoulli.rvs(img2prob[a])

				if stats.bernoulli.rvs(0.5):
					left_image[ii] = a
					left_prob_level[ii] = img2prob_level[a]
					left_amnt_level[ii] = img2amnt_level[a]

					lever[ii] = -1

				else:
					right_image[ii] = a
					right_prob_level[ii] = img2prob_level[a]
					right_amnt_level[ii] = img2amnt_level[a]

					lever[ii] = 1

		fake_data = pd.concat((fake_data, pd.DataFrame({'date': date, 'sesstype': 'B', 'trial': trial, 'block': 1, \
			'trialtype': trialtype, 'left_prob_level': left_prob_level, 'right_prob_level': right_prob_level, \
			'left_amnt_level': left_amnt_level, 'right_amnt_level': right_amnt_level, 'left_image': left_image, \
			'right_image': right_image, 'set': 1, 'lever': lever, 'if_reward': if_reward})), ignore_index=True)

	fake_data['if_reward'] = fake_data['if_reward'].replace({0: np.nan})

	return fake_data

def mergeSim(data, sim, n=None):
	data = data.copy()
	data = data.drop(columns=['lever', 'if_reward'])

	if not n is None:
		sim = sim[sim['iter']==n]

	new_lever = sim['sim_choice'].rename('lever')
	new_reward = sim['reward'].mask(sim['reward'] > 0, 1).replace({0: np.nan}).rename('if_reward')

	return pd.concat((data, new_lever, new_reward), axis=1)

def isvalid(data, forced=False, sets='new', return_data=False):
	'''
	Sequence of Booleans denoting whether each trial in data is valid for analysis

	data:   (DataFrame) all trials to be assessed
	forced: (Boolean) include forced trials, otherwise only
	sets:   (string) sets to include
				'all': include all sets of all sessions
				'new': include only set B of each session
	'''
	data = data.copy()
	valid = (data['lever'].isin([-1,1])) # a left or right choice must be made for trial to be valid
	
	# if forced trials included, exclude forced trials where wrong choice is made
	# else, only include free trials
	if forced:
		valid &= ((data['trialtype']==2) | ((data['trialtype']==1) & \
					((np.isnan(data['left_image']) & (data['lever'] == 1)) | \
					 (np.isnan(data['right_image']) & (data['lever'] == -1)))))
	else:
		valid &= (data['trialtype']==2)
	
	# filter out unwanted sessions
	if sets == 'all':
		valid &= (((data['sesstype']=='B') & (data['set']==1)) | (data['sesstype']=='ABA'))
	elif sets == 'new':
		valid &= (((data['sesstype']=='B') & (data['set']==1)) | ((data['sesstype']=='ABA') & (data['set']==2)))
	else:
		raise ValueError
	
	if return_data:
		return data[valid]
	else:
		return valid

def simRollingAvg(block_data, block_sims, win_size=50, min_trials=10):
	'''
	Calculates a rolling average of simulation performance

	block_data: (DataFrame) real data from block of trials
	block_sims: (DataFrame) simulations of block of trials
	win_size: (int) rolling window size, in trials
	min_trials: (int) minimum number of trials to include in rolling average
	'''
	block_data = block_data.copy()
	block_data = block_data[isvalid(block_data, sets='all')]

	left_amnt = block_data['left_amnt_level'].replace({1: 0.5, 2: 0.3, 3: 0.1})
	left_prob = block_data['left_prob_level'].replace({1: 0.7, 2: 0.4, 3: 0.1})
	right_amnt = block_data['right_amnt_level'].replace({1: 0.5, 2: 0.3, 3: 0.1})
	right_prob = block_data['right_prob_level'].replace({1: 0.7, 2: 0.4, 3: 0.1})

	perf = (left_amnt*left_prob > right_amnt*right_prob).replace({True: -1.0, False: 1.0})
	perf = (perf == block_sims[''])

def rollingAvg(block, output = 'perf', win_size=50, min_trials=10, amnt_map={1:0.5, 2:0.3, 3:0.1}, prob_map={1:0.7, 2:0.4, 3:0.1}):
	'''
	Calculates a rolling average of a given timeseries for the input block data

	block:      (DataFrame) block of trials
	output:     (string) timeseries data returned
					'perf': ratio of trials higher value is chosen over lower value
					'rt':   reaction time
	win_size:   (int) rolling window size, in trials
	min_trials: (int) minimum number of trials to include in rolling average
	'''
	block = isvalid(block.copy(), sets='all', return_data=True)
	if output == 'perf':
		left_amnt = block['left_amnt_level'].replace(amnt_map)
		left_prob = block['left_prob_level'].replace(prob_map)
		right_amnt = block['right_amnt_level'].replace(amnt_map)
		right_prob = block['right_prob_level'].replace(prob_map)

		ts = (~((left_amnt*left_prob > right_amnt*right_prob) ^ (block['lever'] == -1)) | \
			(left_amnt*left_prob == right_amnt*right_prob))

	elif output == 'model_accuracy':
		ts = block['sim_choice'].apply(lambda x: x == block['lever'])

	elif output == 'rt':
		ts = block['rt']
	
	return ts.rolling(window=win_size, min_periods=min_trials).mean()

def choiceMat(data, compare='lr', sesstype=None, win_size=200, start=None, end='back', model_params={}):
	'''
	Calculate choice matrix for all possible contingency pairs

	data:     (DataFrame) single- or multi-session dataset
	compare:  (string) values to be compared
					'lr': probability of choosing left image over right image
					'ab': probability of choosing contingency A over contingency B
					'lr_diff: mean difference in fitted value between left and right images
					'lr_model': predicted probability of left over right, using fitted values and softmax
	sesstype: (string or None) type of session to include (i.e. ABA or B); if None, includes all sessions
	win_size: (int) size of analysis window, in trials
	start:    (int) number of indices from end to start from
	end:      (string) end of each block to start from
					'back': start from last trial, going backward
					'front': start from first trial, going forward
	'''
	data = data.copy()
	nstim = int(data[['left_image', 'right_image']].max().max())
	if sesstype is not None:
		data = data[data['sesstype']==sesstype] # filter session type if applicable

	# find and select trials that go in specified window
	if end == 'front':
		if start is None:
			start = 0
		trials = list(range(start,start+win_size))
	elif end == 'back':
		if start is None:
			start = win_size

		if start >= win_size:
			trials = list(range(-start,-(start-win_size)))
		else:
			raise ValueError
	else:
		raise ValueError

	if compare == 'lr_model':
		choice_mat = np.full((nstim,nstim), np.nan)
		model_fits = bhv_model.fitSubjValues(data, **model_params)[0]

		for img_l in range(1,nstim+1):
			for img_r in range(1,nstim+1):
				if img_l != img_r:
					choice_mat[img_r-1, img_l-1] = bhv_model.softmax(model_fits['value%i' % img_l], \
						model_fits['value%i' % img_r], model_fits['beta'], model_fits['lr_bias']).mean()

	else:
		data = data[isvalid(data,sets='new')].groupby('date').nth(trials)

		# initialize confusion matrix and sample counts
		choice_mat = np.zeros((nstim,nstim))
		n = np.zeros((nstim,nstim))

		# iterate through all contingency pairs and calculate confusions
		for img_l in range(1,nstim+1):
			for img_r in range(1,nstim+1):
				pair = data[(data['left_image']==img_l) & (data['right_image']==img_r)]
				
				if compare in ['ab','lr']:
					l_over_r = pair['lever'].replace({1:0}).replace({-1:1}).sum()
					# total = pair['lever'].count()
				elif compare in ['lr_sim']:
					l_over_r = pair['sim_choice'].replace({1:0}).replace({-1:1}).mean(axis=1).sum()
					# total = pair['sim_choice'].count()
				elif compare in ['acc_sim']:
					acc = pair['sim_choice'].apply(lambda x: x == pair['lever']).mean(axis=1).sum()
				elif compare == 'lr_diff':
					lr_value_diff = pair['left_fit_value'].mean() - pair['right_fit_value'].mean()
				total = pair['lever'].count()
				
				if compare in ['ab','lr','lr_sim']:
					choice_mat[img_r-1, img_l-1] += l_over_r
					n[img_r-1, img_l-1] += total
				elif compare == 'acc_sim':
					choice_mat[img_r-1, img_l-1] += acc
					n[img_r-1, img_l-1] += total
				elif compare == 'lr_diff':
					choice_mat[img_r-1, img_l-1] += lr_value_diff
				else:
					raise ValueError
				
				if compare=='ab':
					choice_mat[img_l-1, img_r-1] += total - l_over_r
					n[img_l-1, img_r-1] += total

		if compare in ['ab','lr','lr_sim','acc_sim']:
			n[n==0]=np.nan
			choice_mat = choice_mat / n

	return choice_mat

def subdivide(choice_mat, by):
	'''
	Subdivides confusion matrix into submatrices grouped by reward probability or amount

	choice_mat: (numpy Array) confusion matrix
	by:       (string) grouping dimension
					'prob': group by reward probability
					'amnt': group by reward amount
	'''
	groups = np.zeros((9,9))
	labels = []

	if by == 'prob':
		levels = ['High P', 'Med P', 'Low P']
	elif by == 'amnt':
		levels = ['High A', 'Med A', 'Low A']
		# re-sort matrix by amount level
		choice_mat = choice_mat[[0,3,6,1,4,7,2,5,8],:]
		choice_mat = choice_mat[:,[0,3,6,1,4,7,2,5,8]]
	else:
		raise ValueError

	ii = 0
	for b in range(3):
		for a in range(3):
			groups[ii,:] = choice_mat[b*3:(b*3)+3, a*3:(a*3)+3].flatten()
			labels.append('%s v %s' % (levels[a], levels[b]))
			ii += 1

	return groups, labels
	
def plotSession(data, date, series1='perf', series2=None, win_step=10, amnt_map={1:0.5, 2:0.3, 3:0.1}, prob_map={1:0.7, 2:0.4, 3:0.1}, **kwargs):
	'''
	Plots rolling average of performance over time for a given session

	data:       (DataFrame) master experimental dataset
	date:       (string) session date, in the form 'YYYY-MM-DD'
	series1:    (string) primary timeseries from data
	series2:    (string) secondary timeseries from data
	win_step:   (int) sliding window step size, in trials
	plotRT:     (Boolean) if True, superimposes reaction time over performance
	'''
	sess = data[data['date']==date] # select data specific to session date
	cscheme = ['blue','crimson'] # custom color scheme
	
	fig, ax1 = plt.subplots(figsize=(12,5))
	ax2 = None

	# series 1 plot settings
	if series1 == 'perf':
		ylabel1 = 'P(optimal)'
		ylim1 = [0.4, 1.01]
		colors1 = ['blue', 'crimson']
	elif series1 == 'model':
		ylabel1 = 'P(optimal)'
		ylim1 = [0.4, 1.01]
		colors1 = ['red','red']
	elif series1 == 'model_accuracy':
		ylabel1 = '% Correct'
		ylim1 = [0.4, 1.01]
		colors1 = ['blue','blue']
	elif series1 == 'rt':
		ylabel1 = 'Reaction Time (ms)'
		ylim1 = [500, 1000]
		colors1 = ['g', 'g']
	else:
		raise ValueError

	# series2 plot settings
	if not series2 is None:
		if series2 == 'perf':
			if series1 in ['model_accuracy','rt']:
				ax2 = ax1.twinx() # add 2nd axis
				ylabel2 = 'P(high > low)'
				ylim2 = [.4, 1.01]
			colors2 = 'grey'
		elif series2 == 'model':
			if series1 in ['model_accuracy','rt']:
				ax2 = ax1.twinx()
				ylabel2 = 'P(high > low)'
				ylim2 = [.4, 1.01]
			colors2 = 'red'
		elif series2 == 'rt':
			ax2 = ax1.twinx()
			ylabel2 = 'Reaction Time (ms)'
			ylim2 = [500, 1000]
			colors2 = 'g'
		else:
			raise ValueError

	if ax2 is None:
		ax2 = ax1
	
	# chance and criterion performance lines
	ax1.axhline(.5,color='k',lw=.75)
	ax1.axhline(.8,color='k',lw=.75)
	
	# if session has an ABA structure, subdivide into A and B blocks
	if sess['sesstype'].iloc[0] == 'ABA' and sess['block'].iloc[0] == 1:
		last_trial = 0
		for ii, block_set in enumerate([[1,2],[3,4],[5]]):
			block = sess[sess['block'].isin(block_set)] # combine adjacent same-condition mini-blocks
			if block.shape[0] > 0:
				data1 = rollingAvg(block, output=series1, amnt_map=amnt_map, prob_map=prob_map, **kwargs) # calculate rolling average

				# align samples to trial numbers
				trials = np.arange(data1.shape[0])+1+last_trial
				last_trial = trials[-1]

				# downsample rolling average to sliding window step size
				data1 = data1[win_step-1::win_step]
				trials = trials[win_step-1::win_step]

				if ii < 2:
					ax1.axvline(last_trial+1,color='grey',lw=.5) # dividing boundary between blocks

				# repeat for series2 data
				if not series2 is None:
					data2 = rollingAvg(block, output=series2, amnt_map=amnt_map, prob_map=prob_map, **kwargs)
					data2 = data2[win_step-1::win_step]

					if len(data2.shape) > 1:
						data2 = data2.groupby(axis=1, level=0)
						ax2.errorbar(trials,data2.mean().to_numpy(),yerr=data2.sem().to_numpy().squeeze(), \
							color='red',ls='--',ecolor='pink')
					else:
						ax2.plot(trials,data2,color=colors2,ls='--')

				if len(data1.shape) > 1:
					orig_shape = data1.shape
					data1 = data1.to_numpy().T.reshape((np.prod(orig_shape)))
					sns.lineplot(np.tile(trials, orig_shape[1]), data1, color=colors1[ii%2], ax=ax1)
				else:
					ax1.plot(trials,data1,color=colors1[ii%2])
			
	else:
		data1 = rollingAvg(sess, output=series1, amnt_map=amnt_map, prob_map=prob_map, **kwargs) # calculate rolling average
		trials = np.arange(data1.shape[0])+1 # align samples to trial numbers

		# downsample rolling average to sliding window step size
		data1 = data1[win_step-1::win_step]
		trials = trials[win_step-1::win_step]
		
		# repeat for series2 data
		if not series2 is None:
			data2 = rollingAvg(sess, output=series2, amnt_map=amnt_map, prob_map=prob_map, **kwargs)
			data2 = data2[win_step-1::win_step]

			if len(data2.shape) > 1:
				data2 = data2.groupby(axis=1, level=0)
				ax2.errorbar(trials,data2.mean().to_numpy(),yerr=data2.sem().to_numpy().squeeze(), \
					color='red',ls='--',ecolor='pink')
			else:
				ax2.plot(trials,data2,color=colors2,ls='--')

		if len(data1.shape) > 1:
			orig_shape = data1.shape
			data1 = data1.to_numpy().T.reshape((np.prod(orig_shape,)))
			sns.lineplot(np.tile(trials, orig_shape[1]), data1, color=colors1[1], ax=ax1)
			# sns.lineplot(trials, data1.to_numpy(), ci='sd')
		else:
			ax1.plot(trials,data1,color=colors1[1])
	
	ax1.set_title(date)
	ax1.set_xlabel('Free Trial')
	ax1.set_ylabel(ylabel1)
	ax1.set_ylim(ylim1)
	
	# label and put limits on reaction time y-axis, if applicable
	if ax2 != ax1:
		ax2.set_ylabel(ylabel2,color=colors2,rotation=270)
		ax2.set_ylim(ylim2)

def plotImageLearningCurves(data, dates=None, win_size=20, min_trials=1, win_step=1, trials='rel', x_cutoff=None):
	data = isvalid(data, forced=True, sets='new', return_data=True)
	if dates is None:
		dates = data['date'].unique()

	amnt_map = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])
	prob_map = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])  

	# df = pd.DataFrame(columns=['date', 'trial', 'p(reward)', 'amount', 'EV', 'p(choose)'])
	df = pd.DataFrame(columns=['date', 'update', 'p_reward', 'amount', 'EV', 'p_choose'])

	for date in dates:
		for img in np.arange(1,10):
			# trials on date where img is offered
			sess_img_data = data[(data['date']==date) & \
				((data['left_image']==img) | (data['right_image']==img))]

			a = amnt_map[int(img)-1] # amount of reward
			p = prob_map[int(img)-1] # probability of reward

			# image location: Left = -1, Right = 1
			img_loc = (sess_img_data['left_image']==img).replace({True:-1, False:1})

			# series of Bool indicating whether image location matches choice
			update = (img_loc==sess_img_data['lever'])

			trials = sess_img_data[sess_img_data['trialtype']==2]['trial']

			update_trials = sess_img_data[update]['trial'].to_list()
			update_trials.insert(0, -1)
			if update_trials[-1] <= trials.iloc[-1]:
				update_trials.append(trials.iloc[-1]+1)

			# pstable = choose.tail(100).mean() # stable p_choose
			pchoose = update[sess_img_data['trialtype']==2]
			pchoose = pchoose.rolling(window=20, min_periods=1).mean().rename('p_choose')

			# bin pchoose estimates by most recent update and average within bin
			pchoose = pd.concat((pchoose, pd.cut(trials, update_trials, right=False, \
									   labels=np.arange(len(update_trials)-1)).rename('update')), axis=1)
			pchoose = pchoose.groupby('update').mean().interpolate()

			df2 = pd.DataFrame({'date': date, 'update': pchoose.index, \
				'p_reward': p, 'amount': a, 'EV': round(p*a, 2), 'p_choose': pchoose['p_choose']})
			df = pd.concat((df, df2), axis=0, ignore_index=True)

	if not x_cutoff is None:
		df = df[df['update'] < x_cutoff]

	plt.figure()
	sns.lineplot(x='update', y='p_choose', hue='EV', data=df, palette=sns.color_palette('coolwarm_r',9))
	plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
	plt.xlabel('Value Updates')
	plt.ylabel('p(Choose)')

	# plt.figure()
	# sns.lineplot(x='trial', y='pct_stable', hue='EV', data=df, palette=sns.color_palette('coolwarm_r',9), ci=None)
	# plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)
	# plt.xlabel('Trial Offered')
	# plt.ylabel('Log Odds Rel. to Stable')

	return df

def plotChoiceMat(data, compare='lr', sesstype=None, win_size=200, start=200, end='back', \
	annot=True, model_params={}):
	'''
	Plots a confusion matrix for all possible contingency pairs

	data:     (DataFrame) single- or multi-session dataset
	compare:  (string) values to be compared
					'lr': probability of choosing left image over right image
					'ab': probability of choosing contingency A over contingency B
					'lr_diff: mean difference in fitted value between left and right images
					'lr_model': predicted probability of left over right, using fitted values and softmax
	sesstype: (string or None) type of session to include (i.e. ABA or B); if None, includes all sessions
	win_size: (int) size of analysis window, in trials
	start:    (int) number of indices from end to start from
	end:      (string) end of each block to start from
					'back': start from last trial, going backward
					'front': start from first trial, going forward
	annot:    (Boolean) if True, annotates each box of matrix with its value
	'''
	nstim = int(data[['left_image', 'right_image']].max().max())
	nprob = int(data[['left_prob_level', 'right_prob_level']].max().max())

	choice_mat = choiceMat(data, compare=compare, sesstype=sesstype, win_size=win_size, start=start, end=end, \
		model_params=model_params)
	
	# set colorbar label and limits according to specified comparison
	if compare == 'lr':
		cbar_label = 'P(left > right)'
		cmap='plasma'
		vmin = 0
		vmax = 1
	elif compare == 'ab':
		cbar_label = 'P(A > B)'
		cmap='plasma'
		vmin = 0
		vmax= 1
	elif compare == 'lr_diff':
		cbar_label = 'Left - Right (Mean Fitted Value)'
		cmap='plasma'
		vmin = -4.6
		vmax = 4.6
	elif compare == 'lr_model':
		cbar_label = 'Predicted P(left > right)'
		cmap='plasma'
		vmin = 0
		vmax = 1
	elif compare == 'lr_sim':
		cbar_label = 'Simulated P(left > right)'
		cmap='plasma'
		vmin = 0
		vmax = 1
	elif compare == 'acc_sim':
		cbar_label = 'Simulation Accuracy'
		cmap='inferno'
		vmin = 0.5
		vmax = 1
	
	# plot heatmap and subdividing gridlines
	sns.heatmap(np.round(choice_mat,2), vmin=vmin, vmax=vmax, cmap=cmap, annot=annot, cbar_kws={'label': cbar_label})
	for ii in range(int(nstim/nprob)-1):
		plt.axvline((ii+1)*nprob, color='w', lw=2)
		plt.axhline((ii+1)*nprob, color='w', lw=2)
	
	# add axis labels
	if compare in ['lr','lr_diff','lr_model','lr_sim','acc_sim']:
		plt.xlabel('Left Image')
		plt.ylabel('Right Image')
	elif compare == 'ab':
		plt.xlabel('Image A')
		plt.ylabel('Image B')
	
	if nstim == 9:
		labels = ['High P/High A', 'High P/Med A', 'High P/Low A', \
				 'Med P/High A', 'Med P/Med A', 'Med P/Low A', \
				 'Low P/High A', 'Low P/Med A', 'Low P/Low A']
	elif nstim == 4:
		labels = ['High P/High A', 'High P/Low A', 'Low P/High A', 'Low P/Low A']
	plt.xticks(np.arange(nstim)+.5, labels, rotation=45, ha='right')
	plt.yticks(np.arange(nstim)+.5, labels, rotation='horizontal')
	plt.tick_params(length=0)
	
	# add title
	if end == 'back':
		if win_size == start:
			plt.title('Last %i trials' % win_size)
		else:
			plt.title('%i-%i trials from end' % (start-win_size, start))
	elif end == 'front':
		if 0 == start:
			plt.title('First %i trials' % win_size)
		else:
			plt.title('Trials %i-%i' % (start, start+win_size))
	
	return choice_mat

def plotChoiceEvolution(data, compare='lr', sesstype=None, by=None, epoch_size=50, extent=500, end = 'front'):
	'''
	Plot evolution of choice preferences over time. Specifically, looks at correlation between choice probabilities
	in epochs at different stages of learning with choices probabilities in the final (stable) epoch

	data:      (DataFrame) master dataset
	compare:   (string) values to be compared
					'lr': probability of choosing left image over right image
					'ab': probability of choosing contingency A over contingency B
	sesstype:  (string or None) type of session to include (i.e. ABA or B); if None, includes all sessions
	by:        (string or None) correlate contingencies grouped by reward probability or amount
					None: correlate matrix containing all (ungrouped) contingencies
					'prob': group contingencies by probability
					'amnt': group contingencies by amount
	win_size:  (int) size of each epoch, in trials
	extent:    (int) farthest number of trials from chosen end of block to track choice evolution
	step_size: (int) number of trials to step analysis window over block progression
	end:       (string) end of each block to start from
					'back': start from last trial, going backward
					'front': start from first trial, going forward 
	'''
	step_size = epoch_size

	if compare not in ['lr','ab','lr_sim']:
		raise ValueError

	# ensure extent is compatible with epoch_size
	if extent < epoch_size:
		raise ValueError
	extent = (extent//epoch_size) * epoch_size

	# get final epoch choice matrix to use as standard for comparison
	final_epoch = choiceMat(data, compare=compare, sesstype=sesstype, win_size=epoch_size, end='back')
	if by is None:
		final_epoch = final_epoch.flatten()
		final_epoch = final_epoch[~np.isnan(final_epoch)]
		corr_w_final = np.zeros(extent//epoch_size) # timeseries of epoch correlations w final epoch over learning
		labels = None # group labels (here None, because choice matrix isn't subdivided into groups)
	else:
		# subdivide choice matrix into specified groups
		final_epoch, labels = subdivide(final_epoch, by)
		corr_w_final = np.zeros((9,extent//epoch_size))

	# sample correlation with final choice matrix at different time points across learning
	for ii, start in enumerate(range(0,extent,epoch_size)):
		if end == 'back':
			start += epoch_size

		# sample epoch
		epoch = choiceMat(data, compare=compare, sesstype=sesstype, win_size=epoch_size, start=start, end=end)

		# subdivide choice matrices, if applicable
		if by is None:
			epoch = epoch.flatten()
			epoch = epoch[~np.isnan(epoch)]

			corr_w_final[ii] = np.corrcoef(epoch, final_epoch)[0,1]
		else:
			epoch, _ = subdivide(epoch, by)
			# calculate correlations between similar groups (e.g. sample 'High P/Low P' vs final 'High P/Low P')
			for jj, e_grp, fe_grp in zip(range(len(epoch)), epoch, final_epoch):
				corr_w_final[jj,ii] = np.corrcoef(e_grp[~np.isnan(e_grp)], fe_grp[~np.isnan(fe_grp)])[0,1]

	plt.plot(range(0,extent,step_size), corr_w_final.T) # plot correlation(s) over time

	xtick_labels = []
	for start in range(0, extent, step_size):
		xtick_labels.append('%i-%i' % (start, start+epoch_size))

	plt.xticks(np.arange(0, extent, step_size), xtick_labels, rotation=45, ha='right')
	if end == 'back':
		plt.xlabel('Trials from end of block')
	elif end == 'front':
		plt.xlabel('Trials from start of block')

	plt.ylabel('Correlation with Final Epoch')
	if by is None:
		plt.ylim([.85,1.005])

	# if subdividing matrix, create margin in x-axis and add legend in the margin
	if by is not None:
		plt.legend(labels, loc='upper right')
		plt.xlim([0, extent+200])

	return corr_w_final, labels

def winStayLoseShift(data, n_trials=400, epoch='early'):
	data = data[isvalid(data,forced=True,sets='new')]
	if not n_trials is None:
		if epoch == 'early':
			data = data.groupby('date').head(n_trials)
		elif epoch == 'late':
			data = data.groupby('date').tail(n_trials)

	# Running logs
	date_log = [] # dates, in the form yyyy-mm-dd
	trial_log = [] # trial numbers
	img_log = [] # image ids
	stay_log = [] # stay = 1, shift = 0
	win_log = []  # win = 1, loss = 0
	gap_log = [] # sizes of gaps since image was last chosen

	dates = data['date'].unique()
	for date in dates:
		sess = data[data['date']==date]

		last_outcome = -np.ones(9) # win = 1, loss = 0, unchosen = -1
		last_t = -np.ones(9) # last chosen trial

		for t in range(sess.shape[0]):
			trial = sess.iloc[t]

			if trial['lever'] == -1:
				chosen = int(trial['left_image']-1)
				unchosen = trial['right_image']-1
			else:
				chosen = int(trial['right_image']-1)
				unchosen = trial['left_image']-1

			if trial['trialtype']==2:
				# Update wsls history for chosen image
				if last_outcome[chosen] >= 0:
					date_log.append(date)
					trial_log.append(t)
					img_log.append(chosen+1)
					stay_log.append(1)
					win_log.append(last_outcome[chosen])
					gap_log.append(t-last_t[chosen])

				# update wsls history for unchosen image
				unchosen = int(unchosen)
				if last_outcome[unchosen] >= 0:
					date_log.append(date)
					trial_log.append(t)
					img_log.append(unchosen+1)
					stay_log.append(0)
					win_log.append(last_outcome[unchosen])
					gap_log.append(t-last_t[unchosen])

				last_outcome[unchosen] = -1

			# update chosen outcome (even if forced)
			last_outcome[chosen] = ~np.isnan(trial['if_reward'])
			last_t[chosen] = t
	
	wsls_results = pd.DataFrame({'date': date_log, 'trial': trial_log, 'image': img_log, \
		'stay': stay_log, 'last trial': win_log, 'gap_size': gap_log})

	return wsls_results

def histBiasPP(data):
	data = isvalid(data, forced=True, sets='new', return_data=True)
	data = data.replace({'left_prob_level': {1: 0.7, 2: 0.4, 3: 0.1}, \
		'right_prob_level': {1: 0.7, 2: 0.4, 3: 0.1}, \
		'left_amnt_level': {1: 0.5, 2: 0.3, 3: 0.1}, \
		'right_amnt_level': {1: 0.5, 2: 0.3, 3: 0.1}, \
		'if_reward': {np.nan: 0}})

	max_lag = 10
	max_prev_choice = 8

	col_labels = ['date', 'trial', 'epoch', 'prob', 'amnt', 'alt_prob', 'alt_amnt', 'right_side', 'lag1_gap']
	col_labels += ['lag%s_outcome' % (lag+1) for lag in range(max_lag)]
	col_labels += ['prev_choice'] + ['prev%s_choice' % (n+1) for n in range(1,max_prev_choice)]
	col_labels += ['prev_outcome', 'choice']

	data_pp = np.ndarray(((data['trialtype']==2).sum()*2, len(col_labels)), dtype=object)

	ii = 0
	for date in data['date'].unique():
		sess_data = data[data['date']==date]

		outcome_hist = np.full((9, max_lag), np.nan)
		lag1_trial = np.full(9, np.nan)
		prev_outcome = np.full(9, np.nan)
		prev_choice = np.full((9,max_prev_choice), np.nan)

		for jj in range(sess_data.shape[0]):
			row = sess_data.iloc[jj,:]
			if row['trialtype'] == 2:
				# add left image info
				img_idx = int(row['left_image']-1)

				data_pp[ii,0] = date
				data_pp[ii,1] = row['trial']

				if jj < 400:
					data_pp[ii,2] = 'early'
				elif sess_data.shape[0] - jj < 400:
					data_pp[ii,2] = 'late'

				data_pp[ii,3] = row['left_prob_level']
				data_pp[ii,4] = row['left_amnt_level']
				# data_pp[ii,4] = round(row['left_prob_level']*row['left_amnt_level'],2)
				# data_pp[ii,5] = round(row['right_prob_level']*row['right_amnt_level'],2)
				data_pp[ii,5] = row['right_prob_level']
				data_pp[ii,6] = row['right_amnt_level']
				data_pp[ii,7] = 0
				data_pp[ii,8] = jj - lag1_trial[img_idx]
				data_pp[ii, 9:9+max_lag] = outcome_hist[img_idx,:]
				data_pp[ii,-(2+max_prev_choice):-2] = prev_choice[img_idx,:]
				data_pp[ii,-2] = prev_outcome[img_idx]

				prev_choice[img_idx,:] = np.roll(prev_choice[img_idx,:],1)

				if row['lever'] == -1:
					outcome_hist[img_idx,:] = np.roll(outcome_hist[img_idx,:],1)
					outcome_hist[img_idx,0] = row['if_reward']

					data_pp[ii,-1] = 1
					prev_choice[img_idx,0] = 1
				else:
					data_pp[ii,-1] = 0
					prev_choice[img_idx,0] = 0

				prev_outcome[img_idx] = row['if_reward']
				lag1_trial[img_idx] = jj
				ii += 1


				# add right image info
				img_idx = int(row['right_image']-1)

				data_pp[ii,0] = date
				data_pp[ii,1] = row['trial']

				if jj < 400:
					data_pp[ii,2] = 'early'
				elif sess_data.shape[0] - jj < 400:
					data_pp[ii,2] = 'late'

				data_pp[ii,3] = row['right_prob_level']
				data_pp[ii,4] = row['right_amnt_level']
				# data_pp[ii,4] = round(row['right_prob_level']*row['right_amnt_level'],2)
				# data_pp[ii,5] = round(row['left_prob_level']*row['left_amnt_level'],2)
				data_pp[ii,5] = row['left_prob_level']
				data_pp[ii,6] = row['left_amnt_level']
				data_pp[ii,7] = 1
				data_pp[ii,8] = jj - lag1_trial[img_idx]
				data_pp[ii, 9:9+max_lag] = outcome_hist[img_idx,:]
				data_pp[ii,-(2+max_prev_choice):-2] = prev_choice[img_idx,:]
				data_pp[ii,-2] = prev_outcome[img_idx]

				prev_choice[img_idx,:] = np.roll(prev_choice[img_idx,:],1)

				if row['lever'] == 1:
					outcome_hist[img_idx,:] = np.roll(outcome_hist[img_idx,:],1)
					outcome_hist[img_idx,0] = row['if_reward']

					data_pp[ii,-1] = 1
					prev_choice[img_idx,0] = 1
				else:
					data_pp[ii,-1] = 0
					prev_choice[img_idx,0] = 0

				prev_outcome[img_idx] = row['if_reward']
				lag1_trial[img_idx] = jj
				ii += 1

			else:
				if np.isfinite(row['left_image']):
					img_idx = int(row['left_image']-1)
				else:
					img_idx = int(row['right_image']-1)

				outcome_hist[img_idx,:] = np.roll(outcome_hist[img_idx,:],1)
				outcome_hist[img_idx,0] = row['if_reward']

				prev_choice[img_idx,:] = np.roll(prev_choice[img_idx,:],1)
				prev_choice[img_idx,0] = 1

				prev_outcome[img_idx] = row['if_reward']
				lag1_trial[img_idx] = jj

	# calculate EV
	# data_pp[:,5] = -np.diff(data_pp[:,4:6]).flatten()

	return pd.DataFrame(data_pp, columns=col_labels).infer_objects()
