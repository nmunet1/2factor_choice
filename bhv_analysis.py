import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy.io import loadmat
import seaborn as sns
import bhv_model

def loadData():
    '''
    Reads .mat files containing single-session data into a multi-session pandas DataFrame
    '''
    root = os.getcwd()
    folders = ['bhv_allB','bhv_ABA'] # folders containing data for different session structures
    
    data = pd.DataFrame()
    
    for folder in folders:
        path = os.path.join(root,folder)
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.endswith('.mat') and entry.is_file():
                    date = entry.name[-25:-15] # get session date from file name
                    
                    # session type: ABA or B (image set structure)
                    sess_type = np.nan
                    if folder == 'bhv_allB':
                        sess_type = 'B'
                    elif folder == 'bhv_ABA':
                        sess_type = 'ABA'
                        
                    mat_data = loadmat(entry.path) # dictionary of all data values and labels

                    # start with behavioral timeseries data and labels
                    bhv_labels = [h[0] for h in mat_data['bhv_headers'].flatten()]
                    sess_data = pd.DataFrame(mat_data['bhv_info'],columns=bhv_labels)

                    sess_data.insert(0,'date',date) # add session date
                    sess_data.insert(1,'sesstype',sess_type) # add session type

                    # add fitted free parameters from discounted value and choice functions
                    for w_name, w_fit in zip(mat_data['w_names'].flatten(), mat_data['w_fit'].flatten()):
                        sess_data.insert(sess_data.shape[1], w_name[0][:2], w_fit)
                    
                    data = pd.concat([data,sess_data],ignore_index=True) # join session data with master dataset
                        
    return data

def isvalid(data, forced=False, sets='new'):
    '''
    Sequence of Booleans denoting whether each trial in data is valid for analysis

    data:   (DataFrame) all trials to be assessed
    forced: (Boolean) include forced trials, otherwise only
    sets:   (string) sets to include
                'all': include all sets of all sessions
                'new': include only set B of each session
    '''  
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
    
    return valid

def rollingAvg(block, output = 'perf', win_size=50, min_trials=10):
    '''
    Calculates a rolling average of a given timeseries for the input block data

    block:      (DataFrame) block of trials
    output:     (string) timeseries data returned
                    'perf': ratio of trials higher value is chosen over lower value
                    'rt':   reaction time
    win_size:   (int) rolling window size, in trials
    min_trials: (int) minimum number of trials to include in rolling average
    '''
    block = block[isvalid(block,sets='all')]
    if output == 'perf':
        ts = (block['left_fit_value'] > block['right_fit_value']).replace({True: -1.0, False: 1.0})
        ts = (ts == block['lever']).replace({True:1, False:0})
    elif output == 'rt':
        ts = block['rt']
    else:
        raise ValueError
    
    return ts.rolling(window=win_size, min_periods=min_trials).mean()

def confusions(data, compare='lr', sesstype=None, win_size=200, start=None, end='back', model='dv_lin'):
    '''
    Calculate confusion matrix for all possible contingency pairs

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
        conf_mat = np.full((9,9), np.nan)
        model_fits = bhv_model.fitSubjValues(data, model=model)[0]

        for l_prob in range(3):
            for l_amnt in range(3):
                for r_prob in range(3):
                    for r_amnt in range(3):
                        try:
                            conf_mat[(r_prob*3)+r_amnt, (l_prob*3)+l_amnt] = \
                                model_fits.loc[(1+l_prob, 1+l_amnt, 1+r_prob, 1+r_amnt)]['prob_choose_left']
                        except:
                            conf_mat[(r_prob*3)+r_amnt, (l_prob*3)+l_amnt] = np.nan

    else:
        data = data[isvalid(data,sets='new')].groupby('date').nth(trials)

        # initialize confusion matrix and sample counts
        conf_mat = np.zeros((9,9))
        n = np.zeros((9,9))

        # iterate through all contingency pairs and calculate confusions
        for l_prob in range(3):
            for l_amnt in range(3):
                for r_prob in range(3):
                    for r_amnt in range(3):
                        pair = data[(data['left_prob_level']==(l_prob+1)) & \
                                (data['left_amnt_level']==(l_amnt+1)) & \
                                (data['right_prob_level']==(r_prob+1)) & \
                                (data['right_amnt_level']==(r_amnt+1))]
                        
                        if compare in ['ab','lr']:
                            l_over_r = pair['lever'].replace({1:0}).replace({-1:1}).sum()
                        total = pair['lever'].count()
                        lr_value_diff = pair['left_fit_value'].mean() - pair['right_fit_value'].mean()
                        
                        if compare in ['ab','lr']:
                            conf_mat[(r_prob*3)+r_amnt, (l_prob*3)+l_amnt] += l_over_r
                            n[(r_prob*3)+r_amnt, (l_prob*3)+l_amnt] += total
                        elif compare == 'lr_diff':
                            conf_mat[(r_prob*3)+r_amnt, (l_prob*3)+l_amnt] += lr_value_diff
                        else:
                            raise ValueError
                        
                        if compare=='ab':
                            conf_mat[(l_prob*3)+l_amnt, (r_prob*3)+r_amnt] += total - l_over_r
                            n[(l_prob*3)+l_amnt, (r_prob*3)+r_amnt] += total

        if compare in ['ab','lr']:
            conf_mat = np.divide(conf_mat,n)

    return conf_mat

def subdivide(conf_mat, by):
    '''
    Subdivides confusion matrix into submatrices grouped by reward probability or amount

    conf_mat: (numpy Array) confusion matrix
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
        conf_mat = conf_mat[[0,3,6,1,4,7,2,5,8],:]
        conf_mat = conf_mat[:,[0,3,6,1,4,7,2,5,8]]
    else:
        raise ValueError

    ii = 0
    for b in range(3):
        for a in range(3):
            groups[ii,:] = conf_mat[b*3:(b*3)+3, a*3:(a*3)+3].flatten()
            labels.append('%s v %s' % (levels[a], levels[b]))
            ii += 1

    return groups, labels

def plotSession(data, date, win_step=5, plotRT=False, **kwargs):
    '''
    Plots rolling average of performance over time for a given session

    data:     (DataFrame) master experimental dataset
    date:     (string) session date, in the form 'YYYY-MM-DD'
    win_step: (int) sliding window step size, in trials
    plotRT:   (Boolean) if True, superimposes reaction time over performance
    '''
    sess = data[data['date']==date] # select data specific to session date
    cscheme = ['blue','crimson'] # custom color scheme
    
    fig, ax1 = plt.subplots(figsize=(12,5))
    if plotRT:
        ax2 = ax1.twinx() # add 2nd axis for reaction time, if applicable
    
    # chance and criterion performance lines
    ax1.axhline(.5,color='k',lw=.75)
    ax1.axhline(.8,color='k',lw=.75)
    
    # if session has an ABA structure, subdivide into A and B blocks
    if sess['sesstype'].iloc[0] == 'ABA':
        last_trial = 0
        for ii, block_set in enumerate([[1,2],[3,4],[5]]):
            block = sess[sess['block'].isin(block_set)] # combine adjacent same-condition mini-blocks
            if block.shape[0] > 0:
                # calculate rolling averages
                perf = rollingAvg(block, output='perf', **kwargs)
                rt = rollingAvg(block, output='rt')

                # align samples to trial numbers
                trials = np.arange(perf.shape[0])+1+last_trial
                last_trial = trials[-1]

                # downsample rolling average to sliding window step size
                perf = perf[win_step-1::win_step]
                rt = rt[win_step-1::win_step]
                trials = trials[win_step-1::win_step]

                # add dividing boundaries between blocks
                if ii < 2:
                    ax1.axvline(last_trial+1,color='grey',lw=.5)
                
                # plot rolling average(s) according to color scheme
                ax1.plot(trials,perf,color=cscheme[ii%2])
                if plotRT:
                    ax2.plot(trials,rt,color='g')
            
    elif sess['sesstype'].iloc[0] == 'B':
        # calculate rolling averages
        perf = rollingAvg(sess, output='perf', **kwargs)
        rt = rollingAvg(sess, output='rt')
        
        # align samples to trial numbers
        trials = np.arange(perf.shape[0])+1

        # downsample rolling average to sliding window step size
        perf = perf[win_step-1::win_step]
        rt = rt[win_step-1::win_step]
        trials = trials[win_step-1::win_step]

        # plot rolling average(s) according to color scheme
        ax1.plot(trials,perf,color=cscheme[1])
        if plotRT:
            ax2.plot(trials,rt,color='g')
        
    else:
        raise ValueError
    
    ax1.set_title(date)
    ax1.set_xlabel('Free Trial')
    ax1.set_ylabel('P(high > low)')
    ax1.set_ylim([.4,1.01])
    
    # label and put limits on reaction time y-axis, if applicable
    if plotRT:
        ax2.set_ylabel('Reaction Time (ms)',color='g',rotation=270)
        ax2.set_ylim([500,1000])

def plotConfusions(data, compare='lr', sesstype=None, win_size=200, start=200, end='back', annot=True, model='lin'):
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
    conf_mat = confusions(data, compare=compare, sesstype=sesstype, win_size=win_size, start=start, end=end, model=model)
    
    # set colorbar label and limits according to specified comparison
    if compare == 'lr':
        cbar_label = 'P(left > right)'
        vmin = 0
        vmax = 1
    elif compare == 'ab':
        cbar_label = 'P(A > B)'
        vmin = 0
        vmax= 1
    elif compare == 'lr_diff':
        cbar_label = 'Left - Right (Mean Fitted Value)'
        vmin = -4.6
        vmax = 4.6
    elif compare == 'lr_model':
        cbar_label = 'Predicted P(left > right)'
        vmin = 0
        vmax = 1
    
    # plot heatmap and subdividing gridlines
    sns.heatmap(np.round(conf_mat,2), vmin=vmin, vmax=vmax, cmap='plasma', annot=annot, cbar_kws={'label': cbar_label})
    plt.axvline(3,color='w', lw=2)
    plt.axvline(6,color='w', lw=2)
    plt.axhline(3,color='w', lw=2)
    plt.axhline(6,color='w', lw=2)
    
    # add axis labels
    if compare in ['lr','lr_diff','lr_model']:
        plt.xlabel('Left Image')
        plt.ylabel('Right Image')
    elif compare == 'ab':
        plt.xlabel('Image A')
        plt.ylabel('Image B')
        
    labels = ['High P/High A', 'High P/Med A', 'High P/Low A', \
             'Med P/High A', 'Med P/Med A', 'Med P/Low A', \
             'Low P/High A', 'Low P/Med A', 'Low P/Low A']
    plt.xticks(np.arange(9)+.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(9)+.5, labels, rotation='horizontal')
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
    
    return conf_mat

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

    if compare not in ['lr','ab']:
        raise ValueError

    # ensure extent is compatible with epoch_size
    if extent < epoch_size:
        raise ValueError
    extent = (extent//epoch_size) * epoch_size

    # get final epoch choice matrix to use as standard for comparison
    final_epoch = confusions(data, compare=compare, sesstype=sesstype, win_size=epoch_size, end='back')
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
        epoch = confusions(data, compare=compare, sesstype=sesstype, win_size=epoch_size, start=start, end=end)

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
        plt.ylim([.92,1.005])

    # if subdividing matrix, create margin in x-axis and add legend in the margin
    if by is not None:
        plt.legend(labels, loc='upper right')
        plt.xlim([0, extent+200])

    return corr_w_final, labels