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



def plot_hists_mean(fracdiffs_arr, param_labels, label_arr=None,
                    color_arr=['salmon'], bins=20, xlim_auto=True,
                    alpha=0.5, histtype='bar'):
    if fracdiffs_arr.ndim==2:
        fracdiffs_arr = np.array([fracdiffs_arr])
    n_params = fracdiffs_arr.shape[-1]
        
    for pp in range(n_params):
        fig = plt.figure(figsize=(3,3))
        for i, fracdiffs in enumerate(fracdiffs_arr):
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
            fig.legend(fontsize=12, bbox_to_anchor=(1.7, 0.9))


def plot_hists_cov(theta_true_arr, theta_pred_arr, covs_pred_arr, param_labels,
                   label_arr=None, color_arr=None, nbins=20, xlim_auto=True,
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
            plt.hist(sigmas_from_truth_arr[i][:,pp], bins=np.linspace(xmin, xmax, nbins),
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
        plt.hist(chi2s_arr[i], bins=np.linspace(xmin, xmax, nbins), 
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
        
        
smooth_dict = {'mn': 1, 'emcee': 2, 'dynesty': 2}
bins_dict = {'mn': None, 'emcee': 10, 'dynesty': 7}

def plot_contours_inf(param_names, idx_obs, theta_obs_true,
                      inf_methods, tags_inf, 
                      colors=None, labels=None, labels_extra=None):
    title = f'test model {idx_obs}'
    truth_loc = dict(zip(param_names, theta_obs_true))

    samples_arr = []
    for i, inf_method in enumerate(inf_methods):
        samples = utils.get_samples(idx_obs, inf_method, tags_inf[i])
        samples_arr.append(samples)

    smooth_arr = [smooth_dict[method] for method in inf_methods]
    bins_arr = [bins_dict[method] for method in inf_methods]
    if colors is None:
        colors = [utils.color_dict_methods[meth] for meth in inf_methods]
    if labels is None:
        labels = [utils.label_dict_methods[meth] for meth in inf_methods]
    if labels_extra is not None:
        labels = [labels[i] + ' ' + labels_extra[i] for i in range(len(labels))]

    plot_contours(samples_arr, labels, colors, param_names, utils.param_label_dict, 
                        smooth_arr=smooth_arr, bins_arr=bins_arr,
                        truth_loc=truth_loc, title=title, extents={}, fn_save=None)
 # for backwards compatibility

def plot_overdensity_field(tracer_field, normalize=False, vmax=None, 
                      title=None, show_labels=True, show_colorbar=True,
                      slice_width=1, figsize=(6,6), symlog=False):
   
    plot_field(tracer_field, normalize=normalize, vmax=vmax, 
                      title=title, show_labels=show_labels, show_colorbar=show_colorbar,
                      slice_width=slice_width, figsize=figsize, symlog=symlog)
                    

def plot_field(tracer_field, normalize=False, vmin=None, vmax=None, 
                title=None, show_labels=True, show_colorbar=True,
                zslice_min=0, zslice_max=1, figsize=(6,6), log=False, symlog=False, 
                overdensity=True):

        print(np.min(tracer_field), np.max(tracer_field))

        if normalize:
            tracer_field /= np.max(np.abs(tracer_field))
        print(np.min(tracer_field), np.max(tracer_field))
        
        if vmax is None:
            vmax = 3*np.std(tracer_field)
       
        print(tracer_field.shape)
        if tracer_field.ndim==3:
            field_2d = np.mean(tracer_field[:,:,zslice_min:zslice_max], axis=-1)
        elif tracer_field.ndim==2:
            field_2d = tracer_field
        else:
            raise ValueError("field must be 2d or 3d!")
        print(field_2d.shape)

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
            if overdensity:
                label = r'overdensity $\delta$'
            else:
                label = r'density'
            cbar = plt.colorbar(im, label=label, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12) 
            
        if not show_labels:    
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        plt.show()