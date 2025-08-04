import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

import utils


def plot_loss(history):
    plt.figure(figsize=(4,3))
    plt.plot(history['loss'], color='blue', label='training set', alpha=0.6)
    plt.plot(history['val_loss'], color='limegreen', label='validation set', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    
    

def plot_hists_params(vals_arr, param_labels, xlabel_base='', label_arr=None,
                color_arr=['salmon'], bins=20, xlim_auto=True,
                alpha=0.5, histtype='bar'):
    if vals_arr.ndim==2:
        vals_arr = np.array([vals_arr])
    n_params = vals_arr.shape[-1]
    for pp in range(n_params):
        fig = plt.figure(figsize=(3,3))
        for i, val_arr in enumerate(vals_arr):
            label = None
            if label_arr is not None:
                label = label_arr[i]
            plt.hist(val_arr[:,pp], bins=bins, alpha=alpha, 
                        color=color_arr[i], label=label, histtype=histtype,
                        lw=2)
        plt.xlabel(rf'{xlabel_base}, {param_labels[pp]}', fontsize=14)
        plt.ylabel(r'$N$ in bin', fontsize=14)
        plt.axvline(0, color='grey')
        if not xlim_auto:
            plt.xlim(-np.max(abs(vals_arr[:,:,pp])), np.max(abs(vals_arr[:,:,pp])))
        if label_arr is not None:
            fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))


def plot_hists_mean(theta_pred_arr, theta_true_arr, param_labels, label_arr=None,
                    color_arr=['salmon'], n_bins=20, xlim_auto=True,
                    alpha=0.5, histtype='bar'):
    
    fracdiffs_arr = theta_pred_arr/theta_true_arr - 1
    if fracdiffs_arr.ndim==2:
        fracdiffs_arr = np.array([fracdiffs_arr])
    n_params = fracdiffs_arr.shape[-1]
        
    for pp in range(n_params):
        fig = plt.figure(figsize=(3,3))
        for i, fracdiffs in enumerate(fracdiffs_arr):
            label = None
            if label_arr is not None:
                label = label_arr[i]
                
            if np.all(np.isnan(fracdiffs[:,pp])):
                continue
            
            if not xlim_auto:
                mean = 0
                i_good = ~np.isnan(fracdiffs_arr[:,:,pp])
                #std = np.std(fracdiffs_arr[i_good,pp])
                p16 = np.percentile(fracdiffs_arr[i_good,pp], 16)
                p84 = np.percentile(fracdiffs_arr[i_good,pp], 84)
                psym = 0.5*(np.abs(p16) + np.abs(p84))
                n_std = 3
                xmin = mean - n_std * psym
                xmax = mean + n_std * psym
                bins = np.linspace(xmin, xmax, n_bins)
            else:
                bins = n_bins
            plt.hist(fracdiffs[:,pp], bins=bins, alpha=alpha, 
                     color=color_arr[i], label=label, histtype=histtype,
                     lw=2)
        plt.xlabel(rf'$\Delta${param_labels[pp]}/{param_labels[pp]}', fontsize=14)
        plt.ylabel(r'$N$ in bin', fontsize=14)
        plt.axvline(0, color='grey')

        if label_arr is not None:
            fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))
            

def plot_hists_mean_subplots(theta_pred_arr, theta_true_arr, param_names, param_names_plot=None, param_label_dict=None,
                    n_rows=None, n_cols=None, label_arr=None,
                    color_arr=['salmon'], n_bins=20, xlim_auto=True,
                    alpha=0.5, histtype='bar'):
    """
    Plot histograms of fractional differences for selected parameters in subplots.

    Args:
        theta_pred_arr: Predicted parameter values (array).
        theta_true_arr: True parameter values (array).
        param_labels: List of parameter labels (for axis labeling).
        param_names_show: List of parameter names to show (subset of param_labels).
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        label_arr: List of labels for each set of predictions.
        color_arr: List of colors for each set of predictions.
        n_bins: Number of bins for histograms.
        xlim_auto: Whether to automatically set x-limits.
        alpha: Alpha for histogram bars.
        histtype: Histogram type.
    """
    fracdiffs_arr = theta_pred_arr / theta_true_arr - 1
    if fracdiffs_arr.ndim == 2:
        fracdiffs_arr = np.array([fracdiffs_arr])
    n_params = fracdiffs_arr.shape[-1]

    # If param_names_show is None, show all
    if param_names_plot is None:
        param_names_plot = param_names
    # Map param_names_show to indices in param_labels
    idxs_plot = [param_names.index(pn) for pn in param_names_plot]
    param_labels = [param_label_dict[pn] for pn in param_names_plot]
    print(idxs_plot)
    print(param_labels)

    if n_cols is None and n_rows is None:
        n_cols = len(param_names_plot)
        n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

    for ax_idx, pp in enumerate(idxs_plot):
        ax = axes[ax_idx]
        ax.set_title(rf'{param_labels[ax_idx]}', fontsize=22)
        for i, fracdiffs in enumerate(fracdiffs_arr):
            label = None
            if label_arr is not None:
                label = label_arr[i]
            if np.all(np.isnan(fracdiffs[:, pp])):
                continue
            if not xlim_auto:
                mean = 0
                i_good = ~np.isnan(fracdiffs_arr[:, :, pp])
                p16 = np.percentile(fracdiffs_arr[i_good, pp], 16)
                p84 = np.percentile(fracdiffs_arr[i_good, pp], 84)
                psym = 0.5 * (np.abs(p16) + np.abs(p84))
                n_std = 3
                xmin = mean - n_std * psym
                xmax = mean + n_std * psym
                bins = np.linspace(xmin, xmax, n_bins)
            else:
                bins = n_bins
            ax.hist(fracdiffs[:, pp], bins=bins, alpha=alpha,
                    color=color_arr[i], label=label, histtype=histtype, lw=2)
        ax.set_xlabel(rf'$\Delta${param_labels[ax_idx]}/{param_labels[ax_idx]}', fontsize=14)
        ax.set_ylabel(r'$N$ in bin', fontsize=14)
        ax.axvline(0, color='grey')
        if label_arr is not None and ax_idx == 0:
            ax.legend(fontsize=10)

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_dists_mean_subplots(
    theta_pred_arr, theta_true_arr, param_names, param_names_plot=None, param_label_dict=None,
    n_rows=None, n_cols=None, label_arr=None,
    color_arr=['salmon'], n_bins=20, xlim_auto=True,
    alpha=0.5, histtype='bar', plot_cdf=False
):
    """
    Plot histograms or CDFs of fractional differences for selected parameters in subplots.

    Args:
        theta_pred_arr: Predicted parameter values (array).
        theta_true_arr: True parameter values (array).
        param_names: List of parameter names (for indexing).
        param_names_plot: List of parameter names to show (subset of param_names).
        param_label_dict: Dict mapping param_names to labels.
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        label_arr: List of labels for each set of predictions.
        color_arr: List of colors for each set of predictions.
        n_bins: Number of bins for histograms.
        xlim_auto: Whether to automatically set x-limits.
        alpha: Alpha for histogram bars.
        histtype: Histogram type.
        plot_cdf: If True, plot CDF instead of histogram.
    """
    fracdiffs_arr = theta_pred_arr / theta_true_arr - 1
    if fracdiffs_arr.ndim == 2:
        fracdiffs_arr = np.array([fracdiffs_arr])
    n_params = fracdiffs_arr.shape[-1]

    if param_names_plot is None:
        param_names_plot = param_names
    idxs_plot = [param_names.index(pn) for pn in param_names_plot]
    param_labels = [param_label_dict[pn] for pn in param_names_plot]

    if n_cols is None and n_rows is None:
        n_cols = len(param_names_plot)
        n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

    for ax_idx, pp in enumerate(idxs_plot):
        ax = axes[ax_idx]
        ax.set_title(rf'{param_labels[ax_idx]}', fontsize=22)
        for i, fracdiffs in enumerate(fracdiffs_arr):
            label = None
            if label_arr is not None:
                label = label_arr[i]
            if np.all(np.isnan(fracdiffs[:, pp])):
                continue
            data = fracdiffs[:, pp][~np.isnan(fracdiffs[:, pp])]
            if not xlim_auto:
                mean = 0
                i_good = ~np.isnan(fracdiffs_arr[:, :, pp])
                p16 = np.percentile(fracdiffs_arr[i_good, pp], 16)
                p84 = np.percentile(fracdiffs_arr[i_good, pp], 84)
                psym = 0.5 * (np.abs(p16) + np.abs(p84))
                n_std = 3
                xmin = mean - n_std * psym
                xmax = mean + n_std * psym
                bins = np.linspace(xmin, xmax, n_bins)
            else:
                bins = n_bins
            if plot_cdf:
                # Plot CDF
                sorted_data = np.sort(data)
                yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
                ax.plot(sorted_data, yvals, color=color_arr[i], label=label, lw=2)
                if not xlim_auto:
                    ax.set_xlim(xmin, xmax)
            else:
                # Plot histogram
                ax.hist(data, bins=bins, alpha=alpha,
                        color=color_arr[i], label=label, histtype=histtype, lw=2)
        ax.set_xlabel(rf'$\Delta${param_labels[ax_idx]}/{param_labels[ax_idx]}', fontsize=14)
        if plot_cdf:
            ax.set_ylabel('CDF', fontsize=14)
        else:
            ax.set_ylabel(r'$N$ in bin', fontsize=14)
        ax.axvline(0, color='grey')
        if label_arr is not None and ax_idx == 0:
            ax.legend(fontsize=10)

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_hists_var(theta_true_arr, theta_pred_arr, var_pred_arr, param_labels,
                   label_arr=None, color_arr=None, n_bins=20, xlim_auto=True,
                   alpha=0.5, histtype='bar', lw=2):
    
    if theta_true_arr.ndim==2:
        theta_true_arr = np.array([theta_true_arr])
        theta_pred_arr = np.array([theta_pred_arr])
        covs_pred_arr = np.array([covs_pred_arr])
    n_params = theta_true_arr.shape[-1]
    
    if label_arr is None:
        label_arr = [None] * len(theta_true_arr)
    if color_arr is None:
        color_arr = ['salmon'] * len(theta_true_arr)
    
    # compute stats
    diffs_arr = theta_pred_arr - theta_true_arr
    sigmas_from_truth_arr = []
    chi2s_arr = []
    for i, vars_pred in enumerate(var_pred_arr):
        sigmas_from_truth = []
        for cc, var_pred in enumerate(vars_pred):
            err = np.sqrt(var_pred)
            diffs = diffs_arr[i]
            sigmas_from_truth.append( diffs[cc]/err )
        sigmas_from_truth_arr.append(sigmas_from_truth)
    sigmas_from_truth_arr = np.array(sigmas_from_truth_arr)
    chi2s_arr = np.array(chi2s_arr)
        
    #plot sigmas (1d)
    x_normal = np.linspace(-3, 3, 30)
    mean, variance = 0, 1
    y_normal = np.exp(-np.square(x_normal-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    xmin, xmax = -3, 3
    for pp in range(n_params):
        fig = plt.figure(figsize=(3,3))
        plt.title(rf'{param_labels[pp]}')
        for i in range(sigmas_from_truth_arr.shape[0]):
            plt.hist(sigmas_from_truth_arr[i][:,pp], bins=np.linspace(xmin, xmax, n_bins),
                    color=color_arr[i], label=label_arr[i], alpha=alpha, density=True,
                    histtype=histtype, lw=lw)
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'normalized density', fontsize=12)

        plt.axvline(0, color='grey')
        plt.xlim(xmin, xmax)
        plt.plot(x_normal, y_normal, color='black', lw=1, label=r'$\mathcal{N}(0,1)$')
        if label_arr is not None:
            fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))


def plot_dists_cov_subplots(
    theta_true_arr, theta_pred_arr, covs_pred_arr, param_names, param_names_plot=None, param_label_dict=None,
    label_arr=None, color_arr=None, nbins=20, xlim_auto=True,
    alpha=0.5, histtype='bar', lw=2, n_rows=1, n_cols=None, plot_cdf=False
):
    """
    Plot histograms or CDFs of sigmas-from-truth for selected parameters in subplots.
    """
    from scipy.stats import norm

    if theta_true_arr.ndim == 2:
        theta_true_arr = np.array([theta_true_arr])
        theta_pred_arr = np.array([theta_pred_arr])
        covs_pred_arr = np.array([covs_pred_arr])
    n_params = theta_true_arr.shape[-1]

    if param_names_plot is None:
        param_names_plot = param_names
    idxs_plot = [param_names.index(pn) for pn in param_names_plot]
    param_labels = [param_label_dict[pn] for pn in param_names_plot]

    if label_arr is None:
        label_arr = [None] * len(theta_true_arr)
    if color_arr is None:
        color_arr = ['salmon'] * len(theta_true_arr)

    # compute stats
    diffs_arr = theta_pred_arr - theta_true_arr
    sigmas_from_truth_arr = []
    chi2s_arr = []
    for i, covs_pred in enumerate(covs_pred_arr):
        sigmas_from_truth = []
        chi2s = []
        for cc, cov_pred in enumerate(covs_pred):
            err = np.sqrt(np.diag(cov_pred))
            diffs = diffs_arr[i]
            sigmas_from_truth.append(diffs[cc] / err)
            cov_pred_inv = np.linalg.inv(cov_pred)
            chi2s.append(diffs[cc].T @ cov_pred_inv @ diffs[cc])
        sigmas_from_truth_arr.append(sigmas_from_truth)
        chi2s_arr.append(chi2s)
    sigmas_from_truth_arr = np.array(sigmas_from_truth_arr)
    chi2s_arr = np.array(chi2s_arr)

    # Plot sigmas (1d) for selected parameters
    if n_cols is None:
        n_cols = len(param_names_plot)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).reshape(-1)

    x_normal = np.linspace(-3, 3, 200)
    mean, variance = 0, 1
    y_normal = norm.pdf(x_normal, mean, np.sqrt(variance))
    cdf_normal = norm.cdf(x_normal, mean, np.sqrt(variance))
    xmin, xmax = -3, 3

    for ax_idx, pp in enumerate(idxs_plot):
        ax = axes[ax_idx]
        ax.set_title(rf'{param_labels[ax_idx]}', fontsize=22)
        handles = []
        labels = []
        for i in range(sigmas_from_truth_arr.shape[0]):
            data = sigmas_from_truth_arr[i][:, pp]
            if plot_cdf:
                sorted_data = np.sort(data)
                yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
                line, = ax.plot(sorted_data, yvals, color=color_arr[i], label=label_arr[i], lw=2)
                handles.append(line)
                labels.append(label_arr[i])
            else:
                n, bins, patches = ax.hist(data, bins=np.linspace(xmin, xmax, nbins),
                                           color=color_arr[i], label=label_arr[i], alpha=alpha, density=True,
                                           histtype=histtype, lw=lw)
                line, = ax.plot(x_normal, y_normal, color='black', lw=1, label=r'$\mathcal{N}(0,1)$' if i == 0 else None)
                if i == 0:
                    handles.append(line)
                    labels.append(r'$\mathcal{N}(0,1)$')
        if plot_cdf:
            # Add Gaussian CDF last in legend, only once
            line_cdf, = ax.plot(x_normal, cdf_normal, color='black', lw=1, label='Gaussian CDF', ls='--')
            handles.append(line_cdf)
            labels.append('Gaussian CDF')
            ax.set_ylabel('CDF', fontsize=12)
        else:
            ax.set_ylabel(r'normalized density', fontsize=12)
        ax.set_xlabel(rf'$\Delta${param_labels[ax_idx]}/$\sigma$({param_labels[ax_idx]})', fontsize=14)
        ax.axvline(0, color='grey')
        ax.set_xlim(xmin, xmax)
        if ax_idx == 0:
            ax.legend(handles, labels, fontsize=10)

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Return chi2s_arr and n_params for further plotting
    return sigmas_from_truth_arr, chi2s_arr


def plot_dists_chi2(
    chi2s_arr, n_params, color_arr=None, label_arr=None, nbins=20, alpha=0.5, histtype='bar', lw=2, plot_cdf=False
):
    """
    Plot histogram or CDF of chi^2 values for all parameters.
    """
    from scipy.stats import chi2

    x_normal = np.linspace(0, 10, 200)
    xmin, xmax = 0, 8
    fig = plt.figure(figsize=(3, 3))
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('CDF' if plot_cdf else r'$N$ in bin')
    handles = []
    labels = []
    for i in range(chi2s_arr.shape[0]):
        data = chi2s_arr[i]
        if plot_cdf:
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
            line, = plt.plot(sorted_data, yvals, color=color_arr[i], label=label_arr[i], lw=2)
            handles.append(line)
            labels.append(label_arr[i])
        else:
            n, bins, patches = plt.hist(data, bins=np.linspace(xmin, xmax, nbins),
                                        color=color_arr[i], label=label_arr[i],
                                        alpha=alpha, density=True, lw=lw, histtype=histtype)
            line, = plt.plot(x_normal, chi2.pdf(x_normal, n_params), color='black', lw=1, label=rf'$\chi^2$ PDF, df={n_params}' if i == 0 else None)
            if i == 0:
                handles.append(line)
                labels.append(rf'$\chi^2$ PDF, df={n_params}')
    if plot_cdf:
        # Add chi2 CDF last in legend, only once
        line_cdf, = plt.plot(x_normal, chi2.cdf(x_normal, n_params), color='black', lw=1, label=rf'$\chi^2$ CDF, df={n_params}', ls='--')
        handles.append(line_cdf)
        labels.append(rf'$\chi^2$ CDF, df={n_params}')
    plt.xlim(xmin, xmax)
    plt.legend(handles, labels, fontsize=10, loc='best')#, bbox_to_anchor=(1.7, 0.9))
    plt.show()
    
    

def plot_hists_cov(theta_true_arr, theta_pred_arr, covs_pred_arr, param_labels,
                   label_arr=None, color_arr=None, n_bins=20, xlim_auto=True,
                   alpha=0.5, histtype='bar', lw=2):
    
    if theta_true_arr.ndim==2:
        theta_true_arr = np.array([theta_true_arr])
        theta_pred_arr = np.array([theta_pred_arr])
        covs_pred_arr = np.array([covs_pred_arr])
    n_params = theta_true_arr.shape[-1]
    
    if label_arr is None:
        label_arr = [None] * len(theta_true_arr)
    if color_arr is None:
        color_arr = ['salmon'] * len(theta_true_arr)
    
    # compute stats
    diffs_arr = theta_pred_arr - theta_true_arr
    sigmas_from_truth_arr = []
    chi2s_arr = []
    for i, covs_pred in enumerate(covs_pred_arr):
        sigmas_from_truth = []
        chi2s = []
        for cc, cov_pred in enumerate(covs_pred):
            err = np.sqrt(np.diag(cov_pred))
            diffs = diffs_arr[i]
            sigmas_from_truth.append( diffs[cc]/err )
            cov_pred_inv = np.linalg.inv(cov_pred)
            chi2s.append( diffs[cc].T @ cov_pred_inv @ diffs[cc] )
        sigmas_from_truth_arr.append(sigmas_from_truth)
        chi2s_arr.append(chi2s)
    sigmas_from_truth_arr = np.array(sigmas_from_truth_arr)
    chi2s_arr = np.array(chi2s_arr)
        
    #plot sigmas (1d)
    x_normal = np.linspace(-3, 3, 30)
    mean, variance = 0, 1
    y_normal = np.exp(-np.square(x_normal-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    xmin, xmax = -3, 3
    for pp in range(n_params):
        fig = plt.figure(figsize=(3,3))
        plt.title(rf'{param_labels[pp]}')
        for i in range(sigmas_from_truth_arr.shape[0]):
            plt.hist(sigmas_from_truth_arr[i][:,pp], bins=np.linspace(xmin, xmax, n_bins),
                    color=color_arr[i], label=label_arr[i], alpha=alpha, density=True,
                    histtype=histtype, lw=lw)
        plt.xlabel(r'$\sigma_\text{MN}$')
        plt.ylabel(r'normalized density', fontsize=12)

        plt.axvline(0, color='grey')
        plt.xlim(xmin, xmax)
        plt.plot(x_normal, y_normal, color='black', lw=1, label=r'$\mathcal{N}(0,1)$')
        if label_arr is not None:
            fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))
            
    # plot chi^2s
    from scipy.stats import chi2
    x_normal = np.linspace(0, 10, 30)
    mean, variance = 0, 1
    y_normal = np.exp(-np.square(x_normal-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    xmin, xmax = 0, 8

    fig = plt.figure(figsize=(3,3))
    plt.xlabel(r'$\chi^2$')
    plt.ylabel(r'$N$ in bin')
    for i in range(chi2s_arr.shape[0]):
        plt.hist(chi2s_arr[i], bins=np.linspace(xmin, xmax, n_bins), 
                 color=color_arr[i], label=label_arr[i],
                 alpha=alpha, density=True, lw=lw, histtype=histtype)
    df = n_params #TODO check ??
    plt.plot(x_normal, chi2.pdf(x_normal, df), color='black', lw=1, label=rf'$\chi^2$ PDF, df={df}')
    plt.xlim(xmin, xmax)
    if label_arr is not None:
        fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))




def plot_contours(samples_arr, labels, colors, param_names, param_label_dict, 
                  smooth_arr=None, bins_arr=None,
                  truth_loc={}, title=None, extents={}, 
                  figsize=(7,7), fn_save=None):

    import chainconsumer
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
            legend_kwargs={'bbox_to_anchor': (2.2, 1.6)}
        )
    )

    c.add_truth(chainconsumer.Truth(location=truth_loc))

    fig = c.plotter.plot(figsize=figsize)
    #ax = fig.gca()
    #ax.set_title(title)
    if title is not None:
        fig.suptitle(title)
    if fn_save is not None:
        plt.savefig(fn_save)
        
        
smooth_dict = {'mn': 1, 'sbi': 2, 'emcee': 2, 'dynesty': 2, 'fisher': 2}
bins_dict = {'mn': None, 'sbi': 4, 'emcee': 10, 'dynesty': 7, 'fisher': 4}

def plot_contours_inf(param_names, idx_obs, theta_obs_true,
                      inf_methods, tags_inf, tags_test=None,
                      colors=None, labels=None,
                      figsize=(7,7),
                      extents={}, title=None):
    if title is None:
        title = f'test model {idx_obs}'
    
    # Get samples for each method and determine which parameters are available
    samples_arr = []
    param_names_available = []
    valid_methods = []
    valid_labels = []
    valid_colors = []
    
    for i, inf_method in enumerate(inf_methods):
        if tags_test is None:
            tag_test = ''
        else:
            tag_test = tags_test[i]
        samples, param_names_samples = utils.get_samples(idx_obs, inf_method, tags_inf[i], tag_test=tag_test)
        
        # Check which requested parameters are available in this chain
        available_params = [pn for pn in param_names if pn in param_names_samples]
        
        if len(available_params) == 0:
            print(f"Warning: No requested parameters found in chain for method {inf_method}, tag {tags_inf[i]}. Skipping.")
            continue
            
        # Store the intersection of available parameters
        if i == 0:
            param_names_available = available_params
        else:
            param_names_available = [pn for pn in param_names_available if pn in available_params]
            
        valid_methods.append(inf_method)
        valid_labels.append(labels[i] if labels is not None else None)
        valid_colors.append(colors[i] if colors is not None else None)
    
    # Check if we have any common parameters
    if len(param_names_available) == 0:
        print("Error: No common parameters found across all chains.")
        return
        
    if len(param_names_available) < len(param_names):
        missing_params = [pn for pn in param_names if pn not in param_names_available]
        print(f"Warning: Parameters {missing_params} not available in all chains. Plotting only: {param_names_available}")
    
    # Now collect samples for the available parameters
    samples_arr = []
    for i, inf_method in enumerate(valid_methods):
        if tags_test is None:
            tag_test = ''
        else:
            tag_test = tags_test[i]
        samples, param_names_samples = utils.get_samples(idx_obs, inf_method, tags_inf[i], tag_test=tag_test)
        i_pn = [list(param_names_samples).index(pn) for pn in param_names_available]
        samples_arr.append(samples[:,i_pn])

    smooth_arr = [smooth_dict[method] for method in valid_methods]
    bins_arr = [bins_dict[method] for method in valid_methods]
    if valid_colors[0] is None:
        valid_colors = [utils.color_dict_methods[meth] for meth in valid_methods]

    # Create truth location for available parameters only
    theta_obs_true_available = [theta_obs_true[param_names.index(pn)] for pn in param_names_available]
    truth_loc = dict(zip(param_names_available, theta_obs_true_available))

    # DEBUGGING: Check the distribution of each parameter
    # for j, param_name in enumerate(param_names_available):
    #     param_samples = samples_arr[0][:, j]
    #     print(f"{param_name}:")
    #     print(f"  Min: {param_samples.min():.6f}")
    #     print(f"  Max: {param_samples.max():.6f}")
    #     print(f"  Mean: {param_samples.mean():.6f}")
    #     print(f"  Std: {param_samples.std():.6f}")
    #     print(f"  Unique values: {len(np.unique(param_samples))}")
        
    plot_contours(samples_arr, valid_labels, valid_colors, param_names_available, utils.param_label_dict, 
                        smooth_arr=smooth_arr, bins_arr=bins_arr,
                        truth_loc=truth_loc, title=title, figsize=figsize,
                        extents=extents, fn_save=None)
    
 # for backwards compatibility
def plot_overdensity_field(tracer_field, normalize=False, vmax=None, 
                      title=None, show_labels=True, show_colorbar=True,
                      figsize=(6,6), symlog=False):
   
    plot_field(tracer_field, normalize=normalize, vmax=vmax, 
                      title=title, show_labels=show_labels, show_colorbar=show_colorbar,
                      figsize=figsize, symlog=symlog)
                    

def plot_field(tracer_field, normalize=False, vmin=None, vmax=None, 
                title=None, show_labels=True, show_colorbar=True,
                zslice_min=0, zslice_max=1, figsize=(6,6), log=False, symlog=False, 
                overdensity=True, label_cbar=None):

        if normalize:
            tracer_field /= np.max(np.abs(tracer_field))
        
        if vmax is None:
            vmax = 3*np.std(tracer_field)
       
        if tracer_field.ndim==3:
            field_2d = np.mean(tracer_field[:,:,zslice_min:zslice_max], axis=-1)
        elif tracer_field.ndim==2:
            field_2d = tracer_field
        else:
            raise ValueError("field must be 2d or 3d!")

        plt.figure(figsize=figsize, facecolor=(1,1,1,0))
        plt.title(title, fontsize=16)
        
        if symlog:
            from matplotlib.colors import SymLogNorm
            linthresh = 0.1*vmax
            linscale = 1.0
            if vmin is None:
                vmin = -vmax
            norm = SymLogNorm(
                    linthresh=linthresh, linscale=linscale,
                    vmin=vmin, vmax=vmax
                    )
        elif log:
            if vmin is None:
                vmin = np.min(tracer_field[tracer_field>0])
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if vmin is None:
                if overdensity:
                    vmin = -vmax
                else:
                    vmin = np.min(tracer_field[tracer_field>0])
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        if overdensity:
            cmap = 'RdBu'
        else:
            cmap = 'Blues'
                
        im = plt.imshow(field_2d, norm=norm, cmap=cmap)
        ax = plt.gca()        
        
        if show_colorbar:
            if label_cbar is None:
                if overdensity:
                    label_cbar = r'overdensity $\delta$'
                else:
                    label_cbar = r'density'
            cbar = plt.colorbar(im, label=label_cbar, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12) 
            
        if not show_labels:    
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        plt.show()
        
        
        
def plot_pnn(pnn, kk_emu=None, pnn_emu=None):
    fig, axarr = plt.subplots(2, 3, figsize=(20,10), height_ratios=[2,1])
    ax, ax_err = axarr

    labels_pnn = utils.labels_pnn

    contf=0
    import itertools
    prod = np.array(list(itertools.combinations_with_replacement(np.arange(5),r=2)))

    for ii in range(0,len(prod)):
        
        kk = pnn[ii]['k']
        pk = pnn[ii]['pk']

        ax[contf].loglog(kk, pk,
                        color='C'+str(ii), label=labels_pnn[ii], alpha=1, ls='-')
        
        if pnn_emu is not None:
            pk_emu = pnn_emu[ii]
            ax[contf].loglog(kk_emu, pk_emu, ls=':', color='C'+str(ii), 
                            )
            ax_err[contf].semilogx(kk_emu, (pk[i_k_emu]/pk_emu)-1, 
                                ls='-', color='C'+str(ii),)
                
            
        ax[contf].legend(loc='lower left', frameon=True, fancybox=True, fontsize=14)
        
        ax_err[contf].set_xlabel(r'$k[h/$Mpc]', size=30)
        ax_err[contf].axhline(0, ls='-', color='grey')
        ax_err[contf].set_ylim(-0.05, 0.05)
        
        if ii%5==0 and ii>0:
            contf+=1

    ax[0].set_ylabel(r'$P_{ij}(k)$', size=26)
    ax_err[0].set_ylabel(r'$\Delta P_{ij}(k) / P_{ij,\text{emu}}(k) - 1$', size=26)
    ax[0].set_ylim(1e2)
    ax[1].set_ylim(1e2)
    ax[2].set_ylim(1e-1)

