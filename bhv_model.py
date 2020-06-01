import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, basinhopping

import bhv_analysis as bhv

def dv_exp(amnt, prob, w1=1):
    '''
    Exponential discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt*np.exp(w1*(1-prob))

def dv_hyp(amnt, prob, w1=1):
    '''
    Hyperbolic discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt / (1 + w1*(1-prob))

def dv_lin(amnt, prob, w1=1):
    '''
    Linear discounted value function

    amnt:   (float or numpy Array) reward amount
    prob:   (float or numpy Array) reward probability
    w1:     (float or numpy Array) risk aversion
    '''
    return amnt - w1*(1-prob)

def prob_weight(p, gamma):
    '''
    1-parameter weighting function for impact of outcome probability

    p: (float or numpy Array) prospect probability
    gamma: (float or numpy Array) amount and direction of distortion
    '''
    return np.exp(-(-np.log(p))**gamma)

def prob_weight2(p, gamma, delta):
    '''
    2-parameter weighting function for impact of outcome probability

    p: (float or numpy Array) prospect probability
    gamma: (float or numpy Array) curvature
    delta: (float or numpy Array) elevation
    '''
    return (delta*(p**gamma)) / (delta*(p**gamma) + (1-p)**gamma)

def su_mult(x, p, alpha=1, gamma=1, delta=None):
    '''
    Multiplicative subjective utility function for gains under risk, based on prospect theory

    x:      (float or numpy Array) prospect gain
    p:      (float or numpy Array) prospect probability
    alpha:  (float or numpy Array) curvature of gain utility function
    gamma:  (float or numpy Array) curvature of probability weight function
    delta:  (float or numpy Array) elevation of probability weight function
    '''
    if delta is None:
        return (x**alpha) * prob_weight(p, gamma)
    else:
        return (x**alpha) * prob_weight2(p, gamma, delta)

def su_add(x, p, alpha=1, beta_p=.5, gamma=1, delta=None):
    '''
    Additive subjective utility function for gains under risk

    x:      (float or numpy Array) prospect gain
    p:      (float or numpy Array) prospect probability
    alpha:  (float or numpy Array) curvature of gain utility function
    beta_p: (float or numpy Array) weight of probability relative to utility
    gamma:  (float or numpy Array) curvature of probability weight function
    delta:  (float or numpy Array) elevation of probability weight function
    '''
    if delta is None:
        return (1-beta_p) * (x**alpha) + beta_p * prob_weight(p, gamma)
    else:
        return (1-beta_p) * (x**alpha) + beta_p * prob_weight2(p, gamma, delta)

def su_hybrid(x, p, beta_mult=.5, alpha=1, beta_p=.5, gamma=1, delta=None):
    return (1 - beta_mult) * su_add(x, p, alpha, beta_p, gamma, delta) + beta_mult * su_mult(x, p, alpha, gamma, delta)

def softmax(ql, qr, w2=1, w3=0):
    '''
    Softmax function for choice of left image over right image

    ql:   (float or numpy Array) left subjective value
    qr:   (float or numpy Array) right subjective value
    w2:     (float or numpy Array) value difference weight
    w3:     (float or numpy Array) left/right bias
    '''
    return (1 + np.exp(w2 * (ql - qr) + w3))**-1

def estimateSubjValues(data, q_fun, params, param_labels):
    '''
    Estimate subjective values and log-likelihoods

    data:           (DataFrame) session data
    q_fun:          (function) subjective utility/value function
    params:         (list) free parameter values
    param_labels:   (list) free parameter labels
    '''
    levels2prob = {1.0: 0.7, 2.0: 0.4, 3.0: 0.1}
    levels2amnt = {1.0: 0.5, 2.0: 0.3, 3.0: 0.1}

    choice_params = {}
    val_params = {}
    for key, value in zip(param_labels, params):
        if key in ['w2','w3']:
            choice_params[key] = value
        else:
            val_params[key] = value

    pairs = data.groupby(['date', 'left_prob_level', 'left_amnt_level', 'right_prob_level', 'right_amnt_level'])

    prob_l = pairs['left_prob_level'].mean().replace(levels2prob).to_numpy() # left reward probability
    amnt_l = pairs['left_amnt_level'].mean().replace(levels2amnt).to_numpy() # left reward amount
    prob_r = pairs['right_prob_level'].mean().replace(levels2prob).to_numpy() # right reward probability
    amnt_r = pairs['right_amnt_level'].mean().replace(levels2amnt).to_numpy() # right reward probability

    n = pairs['lever'].count().to_numpy() # number of trials
    k = pairs['lever'].sum().to_numpy() # number of left choices

    ql = q_fun(amnt_l, prob_l, **val_params) # left discounted value
    qr = q_fun(amnt_r, prob_r, **val_params) # right discounted value
    
    p_l = softmax(ql, qr, **choice_params) # estimated probability of left choice

    ll = stats.binom.logpmf(k, n , p_l) # log-likelihood

    est = {'k':k, 'n':n, 'left_fit_value':ql, 'right_fit_value':qr, 'prob_choose_left':p_l, 'log-likelihood':ll}
    est.update(val_params)
    est.update(choice_params)

    return pd.DataFrame(est, index=pairs.mean().index)

def negLogLikelihood(params, data, q_fun, param_labels):
    '''
    Negative log-likelihood function for a set of image pairs

    params:         (list) free parameter values
    data:           (DataFrame) session data
    q_fun:          (function) subjective utility/value function
    param_labels:   (list) free parameter labels
    '''
    q_ests = estimateSubjValues(data, q_fun, params, param_labels)

    return -q_ests['log-likelihood'].sum()

def fitSubjValues(data, model='dv_lin', prelec=True, verbose=False, min_type='local', **kwargs):
    '''
    Fits free parameters for subjective value and softmax choice curves

    data:       (DataFrame) dataset or block data
    model:      (str) subjective value model
                    'dv_lin':   linear discounted value
                    'dv_hyp':   hyperbolic discounted value
                    'dv_exp':   exponential discounted value
                    'su_mult':  multiplicative subjective utility
                    'su_add':   additive subjective utility
                    'ev':       expected value
                    'ev_add':   additive "expected value" function
    prelec:     (bool) if True, use Prelec probability weighting function
    verbose:    (bool) if True, displays summary of fit results
    min_type:   (str) minimization type ('local' or 'global')
    '''
    # select last 200 free trials of each valid session, convert choices to binary (1 = left, 0 = right)
    data = data[bhv.isvalid(data, sets='new')].groupby('date').tail(200)
    data = data.replace({'lever': {1: 0}}).replace({'lever': {-1: 1}})

    # select model settings
    if model == 'dv_lin':
        q_fun = dv_lin
        params_init = {'w1':0.1,'w2':-0.1, 'w3':0.1}
        bounds = {'w1':(None,None), 'w2':(None,0), 'w3':(None,None)}
    elif model == 'dv_hyp':
        q_fun = dv_hyp
        params_init = {'w1':0.1,'w2':-0.1, 'w3':0.1}
        bounds = {'w1':(0,None), 'w2':(None,0), 'w3':(None,None)}
    elif model == 'dv_exp':
        q_fun = dv_exp
        params_init = {'w1':0.1, 'w2':-0.1, 'w3':0.1}
        bounds = {'w1':(None,None), 'w2':(None,0), 'w3':(None,None)}
    elif model == 'ev':
        q_fun = su_mult
        params_init = {'w2':-0.1, 'w3':0.1}
        bounds = {'w2':(None,0), 'w3':(None,None)}
    elif model == 'ev_add':
        q_fun = su_add
        params_init = {'beta_p':0.5, 'w2':-0.1, 'w3':0.1}
        bounds = {'beta_p':(0,1), 'w2':(None,0), 'w3':(None,None)}
    elif model == 'su_mult':
        q_fun = su_mult
        params_init = {'alpha':0.1, 'gamma':0.1, 'w2':-0.1, 'w3':0.1}
        bounds = {'alpha':(0,1), 'gamma':(0,1), 'w2':(None,None), 'w3':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)
    elif model == 'su_add':
        q_fun = su_add
        params_init = {'alpha':0.1, 'beta_p':0.5, 'gamma':0.1, 'w2':-0.1, 'w3':0.1}
        bounds = {'alpha':(0,1), 'beta_p':(0,1), 'gamma':(0,1), 'w2':(None,None), 'w3':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)
    elif model == 'su_hybrid':
        q_fun = su_hybrid
        params_init = {'beta_mult':0.5, 'alpha':0.1, 'beta_p':0.5, 'gamma':0.1, 'w2':-0.1, 'w3':0.1}
        bounds = {'beta_mult':(0,1), 'alpha':(0,1), 'beta_p':(0,1), 'gamma':(0,1), 'w2':(None,None), 'w3':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)
    else:
        raise ValueError

    fits = pd.DataFrame() # fitted parameters, estimated subjective values, and relevant data

    dates = data['date'].unique()
    for date in dates:
        sess_data = data[data['date']==date]

        param_labels = params_init.keys()
        params = [params_init[label] for label in param_labels]
        bnds = [bounds[label] for label in param_labels]

        if min_type == 'local':
            opt = minimize(negLogLikelihood, params, args=(sess_data, q_fun, param_labels), \
                tol=1e-4, bounds=bnds, **kwargs)
            if verbose or not opt.success:
                print(opt)

        elif min_type == 'global':
            opt = shgo(negLogLikelihood, bounds, args=(sess_data, q_fun, param_labels), \
                minimizer_kwargs={'method':'L-BFGS-B'}, options={'f_tol':1e-4, 'max_iter':100, 'disp':verbose})

        else:
            raise ValueError

        sess_fits = estimateSubjValues(sess_data, q_fun, opt.x, param_labels)
        fits = pd.concat([fits, sess_fits])

    aic = 2*len(params)*len(dates) - 2*fits['log-likelihood'].sum()

    return fits, aic
