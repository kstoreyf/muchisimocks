import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

import chainconsumer
import dynesty
import emcee

import utils


def plot_loss(history):
    plt.figure(figsize=(4,3))
    plt.plot(history['loss'], color='blue', label='training set', alpha=0.6)
    plt.plot(history['val_loss'], color='limegreen', label='validation set', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    

def plot_hists_mean(fracdiffs_arr, param_labels, label_arr=None,
                    color_arr=['salmon'], bins=20, xlim_auto=True,
                    alpha=0.5, histtype='bar'):
    if fracdiffs_arr.ndim==2:
        fracdiffs_arr = np.array([fracdiffs_arr])
    n_params = fracdiffs_arr.shape[-1]
    
    for pp in range(n_params):
        plt.figure(figsize=(3,3))
        for i, fracdiffs in enumerate(fracdiffs_arr):
            #delta_param = (theta_train_pred[:,pp] - theta_train[:,pp])/theta_train[:,pp]
            label = None
            if label_arr is not None:
                label = label_arr[i]
            plt.hist(fracdiffs[:,pp], bins=bins, alpha=alpha, 
                     color=color_arr[i], label=label, histtype=histtype,
                     lw=2)
        plt.xlabel(rf'$\Delta${param_labels[pp]}/{param_labels[pp]}', fontsize=14)
        plt.ylabel(r'$N$ in bin', fontsize=14)
        plt.axvline(0, color='grey')
        if not xlim_auto:
            plt.xlim(-np.max(abs(fracdiffs_arr[:,:,pp])), np.max(abs(fracdiffs[:,:,pp])))
        if label_arr is not None:
            plt.legend(fontsize=8)


def plot_hists_cov(theta_true, theta_pred, covs_pred, param_labels):
    
    # compute stats
    n_params = theta_true.shape[1]
    diffs = theta_pred - theta_true
    sigmas_from_truth = []
    chi2s = []
    for i, cov_pred in enumerate(covs_pred):
        err = np.sqrt(np.diag(cov_pred))
        sigmas_from_truth.append( diffs[i]/err )
        
        cov_pred_inv = np.linalg.inv(cov_pred)
        chi2s.append( diffs[i].T @ cov_pred_inv @ diffs[i] )
        
    sigmas_from_truth = np.array(sigmas_from_truth)
    chi2s = np.array(chi2s)
        
    #plot sigmas (1d)
    x_normal = np.linspace(-3, 3, 30)
    mean, variance = 0, 1
    y_normal = np.exp(-np.square(x_normal-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    xmin, xmax = -3, 3
    for pp in range(n_params):
        plt.figure(figsize=(3,3))
        plt.title(rf'{param_labels[pp]}')
        plt.hist(sigmas_from_truth[:,pp], bins=np.linspace(xmin, xmax, 20),
                color='salmon', alpha=0.5, density=True)
        plt.xlabel(r'$\sigma_\text{MN}$')
        plt.ylabel(r'normalized density', fontsize=12)

        plt.axvline(0, color='grey')
        
        plt.xlim(xmin, xmax)
        plt.plot(x_normal, y_normal, color='black', lw=1, label=r'$\mathcal{N}(0,1)$')
        plt.legend(fontsize=12)

    # plot chi^2s
    from scipy.stats import chi2
    x_normal = np.linspace(0, 10, 30)
    mean, variance = 0, 1
    y_normal = np.exp(-np.square(x_normal-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    xmin, xmax = 0, 8

    plt.figure(figsize=(3,3))
    plt.xlabel(r'$\chi^2$')
    plt.ylabel(r'$N$ in bin')
    plt.hist(chi2s, bins=np.linspace(xmin, xmax, 20), color='salmon', alpha=0.5, density=True)
    df = n_params #TODO check ??
    plt.plot(x_normal, chi2.pdf(x_normal, df), color='black', lw=1, label=rf'$\chi^2$ PDF, df={df}')
    plt.xlim(xmin, xmax)
    plt.legend(fontsize=12)





def plot_contours(samples_arr, labels, colors, param_names, param_label_dict, 
                  smooth_arr=None, bins_arr=None,
                  truth_loc={}, title=None, extents={}, fn_save=None):

    c = chainconsumer.ChainConsumer()
    if smooth_arr is None:
        smooth_arr = [0]*len(param_names)
    if bins_arr is None:
        bins_arr = [None]*len(param_names)

    for i, samples in enumerate(samples_arr):
        c.add_chain(chainconsumer.Chain(
                    samples=pd.DataFrame(samples, columns=param_names),
                    name=labels[i], color=colors[i],
                    smooth=smooth_arr[i], bins=bins_arr[i])
                    )

    c.set_plot_config(
        chainconsumer.PlotConfig(
            flip=True,
            labels=param_label_dict,
            contour_label_font_size=12,
            extents=extents,
        )
    )

    c.add_truth(chainconsumer.Truth(location=truth_loc))

    fig = c.plotter.plot(figsize = (5,4) )
    #ax = fig.gca()
    #ax.set_title(title)
    fig.suptitle(title)
    if fn_save is not None:
        plt.savefig(fn_save)
        
        
smooth_dict = {'mn': 1, 'emcee': 2, 'dynesty': 2}
bins_dict = {'mn': None, 'emcee': 10, 'dynesty': 7}

def plot_contours_inf(param_names, idx_obs, theta_obs_true,
                      methods, tags_inf, 
                      colors=None, labels=None, labels_extra=None):
    title = f'test model {idx_obs}'
    truth_loc = dict(zip(param_names, theta_obs_true))

    samples_arr = []
    for i, method in enumerate(methods):
        if method=='mn':
            rng = np.random.default_rng(42)
            dir_mn = f'../data/results_moment_network/mn{tags_inf[i]}'
            theta_test_pred = np.load(f'{dir_mn}/theta_test_pred.npy')
            covs_test_pred = np.load(f'{dir_mn}/covs_test_pred.npy')
            try:
                samples_mn = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                                    covs_test_pred[idx_obs], int(1e6),
                                                    check_valid='raise')
            except ValueError:
                title += f' [$C$ not PSD!]'
                samples_mn = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                                    covs_test_pred[idx_obs], int(1e6),
                                                    check_valid='ignore')
            samples_arr.append(samples_mn)
            
        if method=='emcee':
            dir_emcee =  f'../data/results_emcee/samplers{tags_inf[i]}'
            fn_emcee = f'{dir_emcee}/sampler_idxtest{idx_obs}.npy'
            if not os.path.exists(fn_emcee):
                print(f'File {fn_emcee} not found')
                return
            reader = emcee.backends.HDFBackend(fn_emcee)

            tau = reader.get_autocorr_time()
            n_burn = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            #print(n_burn, thin)
            samples_emcee = reader.get_chain(discard=n_burn, flat=True, thin=thin)
            samples_arr.append(samples_emcee)
            
        if method=='dynesty':
            dir_dynesty =  f'../data/results_dynesty/samplers{tags_inf[i]}'
            fn_dynesty = f'{dir_dynesty}/sampler_results_idxtest{idx_obs}.npy'
            results_dynesty = np.load(fn_dynesty, allow_pickle=True).item()
            #samples_dynesty = results_dynesty.samples_equal()
            
            from dynesty.utils import resample_equal
            # draw posterior samples
            weights = np.exp(results_dynesty['logwt'] - results_dynesty['logz'][-1])
            samples_dynesty = resample_equal(results_dynesty.samples, weights)

            samples_arr.append(samples_dynesty)


    smooth_arr = [smooth_dict[method] for method in methods]
    bins_arr = [bins_dict[method] for method in methods]
    if colors is None:
        colors = [utils.color_dict_methods[meth] for meth in methods]
    if labels is None:
        labels = [utils.label_dict_methods[meth] for meth in methods]
    if labels_extra is not None:
        labels = [labels[i] + ' ' + labels_extra[i] for i in range(len(labels))]

    plot_contours(samples_arr, labels, colors, param_names, utils.param_label_dict, 
                        smooth_arr=smooth_arr, bins_arr=bins_arr,
                        truth_loc=truth_loc, title=title, extents={}, fn_save=None)
