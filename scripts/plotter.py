import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd

import chainconsumer


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