import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, basinhopping

import behav_analyzer as bhv

def dv_exp(amnt, prob, w1):
    '''
    Exponential discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt*np.exp(w1*(1-prob))

def dv_hyp(amnt, prob, w1):
    '''
    Hyperbolic discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt / (1 + w1*(1-prob))

def dv_lin(amnt, prob, w1):
    '''
    Hyperbolic discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt - w1*(1-prob)

def prob_weight(p, gamma, delta):
    '''
    Weighting function for impact of outcome probability

    p: (float or numpy Array) prospect probability
    gamma: (float or numpy Array) curvature
    delta: (float or numpy Array) elevation
    '''
    return (delta*(p**gamma)) / (delta*(p**gamma) + (1-p)**gamma)

def q_prospect(x, p, alpha, gamma, delta):
    '''
    Subjective value equation for gains under risk, based on prospect theory

    x: (float or numpy Array) prospect gain
    p: (float or numpy Array) prospect probability
    alpha: (float or numpy Array) curvature of gain utility function
    gamma: (float or numpy Array) curvature of probability weight function
    delta: (float or numpy Array) elevation of probability weight function
    '''
    return (x**alpha) * prob_weight(p, gamma, delta)

def softmax(ql, qr, w2, w3):
    '''
    Softmax function for choice of left image over right image

    ql:   (float or numpy Array) left subjective value
    qr:   (float or numpy Array) right subjective value
    w2:     (float or numpy Array) value difference weight
    w3:     (float or numpy Array) left/right bias
    '''
    return (1 + np.exp(w2 * (ql - qr) + w3))**-1

def negLogLikelihood(params, amnt_l, prob_l, amnt_r, prob_r, k, n, q_fun):
    '''
    Negative log-likelihood function for a set of image pairs

    params: (list) free parameters to fit, [(value curve params), (softmax params)], len >= 3
    amnt_l:     (float or numpy Array) left benefit
    prob_l:     (float or numpy Array) left cost
    amnt_r:     (float or numpy Array) right benefit
    prob_r:     (float or numpy Array) right cost
    k:      (float or numpy Array) number of left choices
    n:      (float or numpy Array) total number of choices
    q_fun:  (function) discount value function
    '''
    params = list(params)
    val_params = params[:-2]
    choice_params = params[-2:]

    ql = q_fun(amnt_l, prob_l, *val_params) # left discounted value
    qr = q_fun(amnt_r, prob_r, *val_params) # right discounted value
    
    p_choose_l = softmax(ql, qr, *choice_params) # estimated probability of left choice
    
    negLL = -np.sum(stats.binom.logpmf(k, n, p_choose_l)) # negative log-likelihood
    
    return negLL

def fitSubjValues(all_data, model='dv_lin', split_sess=False, verbose=False, **kwargs):
    '''
    Fits free parameters for subjective value and softmax choice curves

    data: master dataset
    model: subjective value model
            'dv_lin': linear discounted value
            'dv_hyp': hyperbolic discounted value
            'dv_exp': exponential discounted value
            'prospect': prospect theory
    '''
    # select last 200 free trials of each valid session, convert choices to binary (1 = left, 0 = right)
    all_data = all_data[bhv.isvalid(all_data,sets='new')].groupby('date').tail(200)
    all_data = all_data.replace({'lever': {1: 0}}).replace({'lever': {-1: 1}})

    groups = ['left_prob_level', 'left_amnt_level', 'right_prob_level', 'right_amnt_level']
    levels2prob = {1.0: .7, 2.0: .4, 3.0: .1}
    levels2amnt = {1.0: 0.5, 2.0: 0.3, 3.0: 0.1}

    # pick appropriate discounted value function according to model and initial free parameter guesses
    if model == 'dv_lin':
        q_fun = dv_lin
        param_labels = ['w1','w2','w3']
        params = [.1,-.1,.1]
        bnds = [(None,None), (None,0), (None,None)]
    elif model == 'dv_hyp':
        q_fun = dv_hyp
        param_labels = ['w1','w2','w3']
        params = [.1,-.1,.1]
        bnds = [(0,None), (None,0), (None,None)]
    elif model == 'dv_exp':
        q_fun = dv_exp
        param_labels = ['w1','w2','w3']
        params = [.1,-.1,.1]
        bnds = [(None,None), (None,0), (None,None)]
    elif model == 'prospect':
        q_fun = q_prospect
        param_labels = ['alpha','gamma','delta','w2','w3']
        params = [.1,.1,.1,-.1,.1]
        bnds = [(0,1), (0,1), (0,1), (None,None), (None,None)]
    elif model == 'ev':
        q_fun = lambda p, x: p*x
        param_labels = ['w2','w3']
        params = [-.1,.1]
        bnds = [(None,0), (None,None)]
    else:
        raise ValueError

    if split_sess:
        dates = all_data['date'].unique()
        groups.insert(0,'date')
        result = pd.DataFrame()
        aic = []

        for date in dates:
            pairs = all_data[all_data['date']==date].groupby(groups)

            prob_l = np.array(pairs['left_prob_level'].mean().replace(levels2prob)) # left reward probability
            amnt_l = np.array(pairs['left_amnt_level'].mean().replace(levels2amnt)) # left reward amount
            prob_r = np.array(pairs['right_prob_level'].mean().replace(levels2prob)) # right reward probability
            amnt_r = np.array(pairs['right_amnt_level'].mean().replace(levels2amnt)) # right reward probability

            n = np.array(pairs['lever'].count()) # number of trials
            k = np.array(pairs['lever'].sum()) # number of left choices

            opt = minimize(negLogLikelihood, params, args=(amnt_l, prob_l, amnt_r, prob_r, k, n, q_fun), \
                tol=1e-4, bounds=bnds, **kwargs)
            if verbose or not opt.success:
                print(opt)

            param_fits = list(opt.x) # final fitted parameters
            q_fun_fits = param_fits[:-2]
            softmax_fits = param_fits[-2:]

            ql = q_fun(amnt_l, prob_l, *q_fun_fits) # fitted left image values
            qr = q_fun(amnt_r, prob_r, *q_fun_fits) # fitted right image values
            pl = softmax(ql, qr, *softmax_fits) # fitted propabilities of left choice over right

            # compile DataFrame of fitted values for each pair
            sess_res = pairs.mean().index.to_frame()
            sess_res.insert(len(sess_res.columns), 'k', k)
            sess_res.insert(len(sess_res.columns), 'n', n)
            sess_res.insert(len(sess_res.columns), 'prob_choose_left', pl)
            sess_res.insert(len(sess_res.columns), 'left_fit_value', ql)
            sess_res.insert(len(sess_res.columns), 'right_fit_value', qr)
            for fit_value, label in zip(param_fits, param_labels):
                sess_res.insert(len(sess_res.columns), label, fit_value)

            result = pd.concat([result, sess_res], ignore_index = True)
            aic.append(2*len(param_fits) + 2*opt.fun)
    else:
        # select last 200 free trials of each valid session, and convert lever values to binary (1 = left, 0 = right)
        pairs = all_data.groupby(groups)

        prob_l = np.array(pairs['left_prob_level'].mean().replace(levels2prob)) # left reward probability
        amnt_l = np.array(pairs['left_amnt_level'].mean().replace(levels2amnt)) # left reward amount
        prob_r = np.array(pairs['right_prob_level'].mean().replace(levels2prob)) # right reward probability
        amnt_r = np.array(pairs['right_amnt_level'].mean().replace(levels2amnt)) # right reward probability

        n = np.array(pairs['lever'].count()) # number of trials
        k = np.array(pairs['lever'].sum()) # number of left choices

        opt = minimize(negLogLikelihood, params, args=(amnt_l, prob_l, amnt_r, prob_r, k, n, q_fun), \
            tol=1e-4, bounds=bnds, **kwargs)
        if verbose or not opt.success:
            print(opt)

        param_fits = list(opt.x) # final fitted parameters
        q_fun_fits = param_fits[:-2]
        softmax_fits = param_fits[-2:]

        ql = q_fun(amnt_l, prob_l, *q_fun_fits) # fitted left image values
        qr = q_fun(amnt_r, prob_r, *q_fun_fits) # fitted right image values
        pl = softmax(ql, qr, *softmax_fits) # fitted propabilities of left choice over right

        # compile DataFrame of fitted values for each contingency pair
        result = pairs.mean().index.to_frame()
        result.insert(len(result.columns), 'k', k)
        result.insert(len(result.columns), 'n', n)
        result.insert(len(result.columns), 'prob_choose_left', pl)
        result.insert(len(result.columns), 'left_fit_value', ql)
        result.insert(len(result.columns), 'right_fit_value', qr)
        for fit_value, label in zip(param_fits, param_labels):
            result.insert(len(result.columns), label, fit_value)

        aic = 2*len(param_fits) + 2*opt.fun

    return result, aic
