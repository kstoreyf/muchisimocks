"""Functions moved out of code/plotter.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/plotter.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import utils_plot
import utils_inference
from plotter import *  # noqa: F401,F403


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


def plot_hists_var(theta_true_arr, theta_pred_arr, var_pred_arr, param_labels,
                   label_arr=None, color_arr=None, n_bins=20, xlim_auto=True,
                   alpha=0.5, histtype='bar', lw=2):
    
    if theta_true_arr.ndim==2:
        theta_true_arr = np.array([theta_true_arr])
        theta_pred_arr = np.array([theta_pred_arr])
        var_pred_arr = np.array([var_pred_arr])
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
            legend_color_text=False,
            legend_kwargs={'bbox_to_anchor': (1.05, 1.0),
                           'labelcolor': 'black'}
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


def plot_pnn(pnn, kk_emu=None, pnn_emu=None):
    fig, axarr = plt.subplots(2, 3, figsize=(20,10), height_ratios=[2,1])
    ax, ax_err = axarr

    labels_pnn = utils_plot.labels_pnn

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
            # Align measured P(k) onto the emulator k-grid for the residual panel
            pk_on_emu = np.interp(kk_emu, kk, pk)
            ax_err[contf].semilogx(kk_emu, (pk_on_emu/pk_emu)-1, 
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


def plot_catalog_slice(cat, z_center, z_width, ax=None, s=0.1, alpha=0.7, color='blue'):
    """
    Plot a slice of the galaxy catalog as a scatter plot.
    
    Parameters:
    -----------
    cat : h5py.File
        The catalog file
    z_center : float
        Center z-coordinate of the slice
    z_width : float
        Width of the slice in z-direction
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    s : float
        Marker size for scatter plot
    alpha : float
        Transparency of markers
    color : str
        Color of markers
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Get galaxy positions
    gal_pos = cat['gal_pos'][:]
    
    # Define slice boundaries
    z_min = z_center - z_width/2
    z_max = z_center + z_width/2
    
    # Select galaxies in the slice
    mask = (gal_pos[:, 2] >= z_min) & (gal_pos[:, 2] <= z_max)
    slice_pos = gal_pos[mask]
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Scatter plot of x,y positions
    ax.scatter(slice_pos[:, 0], slice_pos[:, 1], s=s, alpha=alpha, c=color)
    
    # Labels and formatting
    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    ax.set_title(f'Galaxy Catalog Slice: z = {z_center:.1f} ± {z_width/2:.1f} Mpc/h\n'
                f'({len(slice_pos)} galaxies)', size=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax

