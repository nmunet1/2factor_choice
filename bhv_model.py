import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, basinhopping, shgo

import bhv_analysis as bhv

def data2numpy(data):
    pairs = data.groupby(['date', 'left_image', 'right_image'])

    img_l = pairs['left_image'].mean().to_numpy() # left image IDs
    img_r = pairs['right_image'].mean().to_numpy() # right image IDs
    k = pairs['lever'].sum().to_numpy() # number of left choices
    n = pairs['lever'].count().to_numpy() # number of trials

    return img_l, img_r, k, n

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

def softmax(ql, qr, beta=-0.1, lr_bias=0):
    '''
    Softmax function for choice of left image over right image

    ql:   (float or numpy Array) left subjective value
    qr:   (float or numpy Array) right subjective value
    beta:     (float or numpy Array) value difference weight
    lr_bias:     (float or numpy Array) left/right bias
    '''
    return (1 + np.exp(beta * (ql - qr) + lr_bias))**-1

def estimateSubjValues(params, param_labels, q_fun, img_l, img_r, k, n):
    '''
    Estimate subjective values and log-likelihoods

    data:           (DataFrame) session data
    q_fun:          (function) subjective utility/value function
    params:         (list) free parameter values
    param_labels:   (list) free parameter labels
    '''
    amnt_l = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])[img_l.astype(int)-1]
    amnt_r = np.array([0.5, 0.3, 0.1, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1])[img_r.astype(int)-1]
    prob_l = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])[img_l.astype(int)-1]
    prob_r = np.array([0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1])[img_r.astype(int)-1]

    choice_params = {}
    val_params = {}
    for key, value in zip(param_labels, params):
        if key in ['beta','lr_bias']:
            choice_params[key] = value
        else:
            val_params[key] = value

    ql = q_fun(amnt_l, prob_l, **val_params) # left discounted value
    qr = q_fun(amnt_r, prob_r, **val_params) # right discounted value
    
    p_l = softmax(ql, qr, **choice_params) # estimated probability of left choice

    ll = stats.binom.logpmf(k, n, p_l) # log-likelihood

    est = {'left_image':img_l, 'right_image':img_r, 'left_fit_value':ql, 'right_fit_value':qr, \
        'prob_choose_left':p_l, 'log-likelihood':ll, 'k':k, 'n':n}
    est.update(val_params)
    est.update(choice_params)

    return est

def negLogLikelihood(params, param_labels, q_fun, img_l, img_r, k, n):
    '''
    Negative log-likelihood function for a set of image pairs

    params:         (list) free parameter values
    data:           (DataFrame) session data
    q_fun:          (function) subjective utility/value function
    param_labels:   (list) free parameter labels
    '''
    ll = estimateSubjValues(params, param_labels, q_fun, img_l, img_r, k, n)['log-likelihood']

    return -ll.sum()

def fitSubjValues(data, model='dv_lin', prelec=True, verbose=False, min_type='global', **kwargs):
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
        params_init = {'w1':0.1,'beta':-0.1, 'lr_bias':0.1}
        bounds = {'w1':(None,None), 'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'dv_hyp':
        q_fun = dv_hyp
        params_init = {'w1':0.1,'beta':-0.1, 'lr_bias':0.1}
        bounds = {'w1':(0,None), 'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'dv_exp':
        q_fun = dv_exp
        params_init = {'w1':0.1, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'w1':(None,None), 'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'ev':
        q_fun = su_mult
        params_init = {'beta':-0.1, 'lr_bias':0.1}
        bounds = {'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'ev_add':
        q_fun = su_add
        params_init = {'beta_p':0.5, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'beta_p':(0,1), 'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'ev_hybrid':
        q_fun = su_hybrid
        params_init = {'beta_mult':0.5, 'beta_p':0.5, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'beta_mult':(0,1), 'beta_p':(0,1), 'beta':(None,0), 'lr_bias':(None,None)}

    elif model == 'su_mult':
        q_fun = su_mult
        params_init = {'alpha':0.1, 'gamma':0.1, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'alpha':(0,1), 'gamma':(0,1), 'beta':(None,None), 'lr_bias':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)

    elif model == 'su_add':
        q_fun = su_add
        params_init = {'alpha':0.1, 'beta_p':0.5, 'gamma':0.1, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'alpha':(0,1), 'beta_p':(0,1), 'gamma':(0,1), 'beta':(None,None), 'lr_bias':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)
    elif model == 'su_hybrid':
        q_fun = su_hybrid
        params_init = {'beta_mult':0.5, 'alpha':0.1, 'beta_p':0.5, 'gamma':0.1, 'beta':-0.1, 'lr_bias':0.1}
        bounds = {'beta_mult':(0,1), 'alpha':(0,1), 'beta_p':(0,1), 'gamma':(0,1), 'beta':(None,None), 'lr_bias':(None,None)}
        if not prelec:
            params_init['delta'] = 0.1
            bounds['delta'] = (0,1)
    else:
        raise ValueError

    # fitted parameters, estimated subjective values, and relevant data
    fits = pd.DataFrame(columns=['value1','value2','value3','value4','value5','value6',\
        'value7', 'value8','value9','beta','lr_bias'])
    ll = 0

    dates = data['date'].unique()
    for date in dates:
        sess_data = data[data['date']==date]

        param_labels = params_init.keys()
        params = [params_init[label] for label in param_labels]
        bnds = [bounds[label] for label in param_labels]

        if min_type == 'local':
            opt = minimize(negLogLikelihood, params, args=(param_labels, q_fun, *data2numpy(sess_data)), \
                tol=1e-4, bounds=bnds, **kwargs)
            if verbose or not opt.success:
                print(opt)

        elif min_type == 'global':
            opt = basinhopping(negLogLikelihood, params, minimizer_kwargs={'method':'L-BFGS-B', \
                'args':(param_labels, q_fun, *data2numpy(sess_data)), 'bounds':bnds, 'tol':1e-4}, disp=verbose)

        else:
            raise ValueError

        sess_est = estimateSubjValues(opt.x, param_labels, q_fun, *data2numpy(sess_data))
        sess_fits = np.zeros(11)
        for ii in range(9):
            sess_fits[ii] = sess_est['left_fit_value'][sess_est['left_image']==ii+1][0]
        sess_fits[9] = sess_est['beta']
        sess_fits[10] = sess_est['lr_bias']

        fits.loc[date] = sess_fits
        ll += sess_est['log-likelihood'].sum()

    aic = 2*len(params)*len(dates) - 2*ll

    return fits, aic
