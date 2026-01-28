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
            


def plot_dists_mean_subplots(
    theta_pred_arr, theta_true_arr, param_names, param_names_plot=None, param_label_dict=None,
    n_rows=None, n_cols=None, label_arr=None,
    color_arr=['salmon'], n_bins=20, xlim_auto=True,
    alpha=0.5, histtype='bar', plot_cdf=False,
    title=None, unreparameterize=False,
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
        unreparameterize: If True, convert reparameterized parameters back to original form.
    """
    # Handle unreparameterization if requested
    if unreparameterize:
        # Define parameters that are reparameterized
        params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
        params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']
        
        # Check if we have sigma8_cold
        if 'sigma8_cold' not in param_names:
            print("Warning: unreparameterize=True but 'sigma8_cold' not in param_names. Skipping unreparameterization.")
            unreparameterize = False
        
        if unreparameterize:
            # Make copies to avoid modifying original arrays
            theta_pred_arr = theta_pred_arr.copy()
            theta_true_arr = theta_true_arr.copy()
            
            # Get sigma8_cold index
            idx_sigma8 = param_names.index('sigma8_cold')
            
            # Create mapping from reparameterized names to original names and indices
            param_names_new = param_names.copy()
            reparam_to_orig = {}
            
            for i, pn in enumerate(param_names):
                if pn.startswith('sigma8_cold_x_') or pn.startswith('sigma8_cold_sq_x_'):
                    # Extract original parameter name
                    if pn.startswith('sigma8_cold_sq_x_'):
                        orig_param_name = pn.replace('sigma8_cold_sq_x_', '')
                    else:
                        orig_param_name = pn.replace('sigma8_cold_x_', '')
                    
                    # Check if this is a known reparameterized parameter
                    if orig_param_name in params_sigma8 or orig_param_name in params_sigma8_squared:
                        reparam_to_orig[i] = orig_param_name
                        # Update param_names to use original name
                        param_names_new[i] = orig_param_name
            
            # Convert reparameterized parameters back to original
            if theta_pred_arr.ndim == 2:
                theta_pred_arr = np.array([theta_pred_arr])
                theta_true_arr = np.array([theta_true_arr])
            
            for i in range(len(theta_pred_arr)):
                sigma8_vals = theta_pred_arr[i][:, idx_sigma8]
                sigma8_true_vals = theta_true_arr[i][:, idx_sigma8]
                
                for reparam_idx, orig_param_name in reparam_to_orig.items():
                    if orig_param_name in params_sigma8:
                        # Divide by sigma8
                        theta_pred_arr[i][:, reparam_idx] = theta_pred_arr[i][:, reparam_idx] / sigma8_vals
                        theta_true_arr[i][:, reparam_idx] = theta_true_arr[i][:, reparam_idx] / sigma8_true_vals
                    elif orig_param_name in params_sigma8_squared:
                        # Divide by sigma8^2
                        theta_pred_arr[i][:, reparam_idx] = theta_pred_arr[i][:, reparam_idx] / (sigma8_vals ** 2)
                        theta_true_arr[i][:, reparam_idx] = theta_true_arr[i][:, reparam_idx] / (sigma8_true_vals ** 2)
            
            # Update param_names
            param_names = param_names_new
            
            # Reshape back if needed
            if len(theta_pred_arr) == 1:
                theta_pred_arr = theta_pred_arr[0]
                theta_true_arr = theta_true_arr[0]
    
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
    row_height = 3 if title is None else 3.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, row_height*n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

    # Collect legend handles and labels from the first subplot
    legend_handles = []
    legend_labels = []

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
                line, = ax.plot(sorted_data, yvals, color=color_arr[i], label=label, lw=2)
                if ax_idx == 0:  # Only collect from first subplot to avoid duplicates
                    legend_handles.append(line)
                    legend_labels.append(label)
                if not xlim_auto:
                    ax.set_xlim(xmin, xmax)
            else:
                # Plot histogram
                n, bins_hist, patches = ax.hist(data, bins=bins, alpha=alpha,
                        color=color_arr[i], label=label, histtype=histtype, lw=2)
                if ax_idx == 0 and label is not None:  # Only collect from first subplot to avoid duplicates
                    # Use the first patch as the legend handle
                    if len(patches) > 0:
                        legend_handles.append(patches[0])
                        legend_labels.append(label)
        ax.set_xlabel(rf'$\Delta${param_labels[ax_idx]}/{param_labels[ax_idx]}', fontsize=14)
        if plot_cdf:
            ax.set_ylabel('CDF', fontsize=14)
        else:
            ax.set_ylabel(r'$N$ in bin', fontsize=14)
        ax.axvline(0, color='grey')

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')

    # Place legend outside the plotting area on the right side
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, fontsize=10, 
                  bbox_to_anchor=(1.02, 0.5), loc='center left')

    # Set overall figure title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.9)

    plt.tight_layout()
    plt.show()

    return fracdiffs_arr


def plot_comp_mean_subplots(
    theta_pred_arr, theta_true_arr, covs_pred_arr, param_names, param_names_plot=None, param_label_dict=None,
    n_rows=None, n_cols=None, label_arr=None,
    color_arr=['salmon'], alpha=0.5,
    title=None, N_plot=None, unreparameterize=False,
):
    """
    Plot predicted vs true parameter values for selected parameters in subplots.
    Includes vertical error bars on predictions and a 1:1 line.

    Args:
        theta_pred_arr: Predicted parameter values (array).
        theta_true_arr: True parameter values (array).
        covs_pred_arr: Covariance matrices for predictions (array).
        param_names: List of parameter names (for indexing).
        param_names_plot: List of parameter names to show (subset of param_names).
        param_label_dict: Dict mapping param_names to labels.
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        label_arr: List of labels for each set of predictions.
        color_arr: List of colors for each set of predictions.
        alpha: Alpha for scatter points.
        title: Overall figure title.
        N_plot: If provided, randomly sample this many points to plot (per dataset).
        unreparameterize: If True, convert reparameterized parameters back to original form.
    """
    # Handle unreparameterization if requested
    if unreparameterize:
        # Define parameters that are reparameterized
        params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
        params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']
        
        # Check if we have sigma8_cold
        if 'sigma8_cold' not in param_names:
            print("Warning: unreparameterize=True but 'sigma8_cold' not in param_names. Skipping unreparameterization.")
            unreparameterize = False
        
        if unreparameterize:
            # Make copies to avoid modifying original arrays
            theta_pred_arr = theta_pred_arr.copy()
            theta_true_arr = theta_true_arr.copy()
            covs_pred_arr = covs_pred_arr.copy()
            
            # Get sigma8_cold index
            idx_sigma8 = param_names.index('sigma8_cold')
            
            # Create mapping from reparameterized names to original names and indices
            param_names_new = param_names.copy()
            reparam_to_orig = {}
            
            for i, pn in enumerate(param_names):
                if pn.startswith('sigma8_cold_x_') or pn.startswith('sigma8_cold_sq_x_'):
                    # Extract original parameter name
                    if pn.startswith('sigma8_cold_sq_x_'):
                        orig_param_name = pn.replace('sigma8_cold_sq_x_', '')
                    else:
                        orig_param_name = pn.replace('sigma8_cold_x_', '')
                    
                    # Check if this is a known reparameterized parameter
                    if orig_param_name in params_sigma8 or orig_param_name in params_sigma8_squared:
                        reparam_to_orig[i] = orig_param_name
                        # Update param_names to use original name
                        param_names_new[i] = orig_param_name
            
            # Convert reparameterized parameters back to original
            if theta_pred_arr.ndim == 2:
                theta_pred_arr = np.array([theta_pred_arr])
                theta_true_arr = np.array([theta_true_arr])
                covs_pred_arr = np.array([covs_pred_arr])
            
            for i in range(len(theta_pred_arr)):
                sigma8_vals = theta_pred_arr[i][:, idx_sigma8]
                sigma8_true_vals = theta_true_arr[i][:, idx_sigma8]
                
                for reparam_idx, orig_param_name in reparam_to_orig.items():
                    if orig_param_name in params_sigma8:
                        # Divide by sigma8
                        theta_pred_arr[i][:, reparam_idx] = theta_pred_arr[i][:, reparam_idx] / sigma8_vals
                        theta_true_arr[i][:, reparam_idx] = theta_true_arr[i][:, reparam_idx] / sigma8_true_vals
                    elif orig_param_name in params_sigma8_squared:
                        # Divide by sigma8^2
                        theta_pred_arr[i][:, reparam_idx] = theta_pred_arr[i][:, reparam_idx] / (sigma8_vals ** 2)
                        theta_true_arr[i][:, reparam_idx] = theta_true_arr[i][:, reparam_idx] / (sigma8_true_vals ** 2)
                
                # Also need to update covariance matrices for reparameterized parameters
                # This is more complex - for now, we'll just update the diagonal elements
                # A full transformation would require Jacobian, but for error bars we mainly need diagonal
                for j in range(len(covs_pred_arr[i])):
                    cov = covs_pred_arr[i][j].copy()
                    sigma8_val = sigma8_vals[j]
                    
                    for reparam_idx, orig_param_name in reparam_to_orig.items():
                        if orig_param_name in params_sigma8:
                            # Variance scales as (1/sigma8)^2
                            cov[reparam_idx, reparam_idx] = cov[reparam_idx, reparam_idx] / (sigma8_val ** 2)
                            # Update off-diagonal elements involving this parameter
                            for k in range(len(param_names)):
                                if k != reparam_idx:
                                    if k == idx_sigma8:
                                        # Covariance with sigma8 needs special handling
                                        # For now, set to 0 (approximation)
                                        cov[reparam_idx, k] = 0
                                        cov[k, reparam_idx] = 0
                                    else:
                                        # Covariance scales as 1/sigma8
                                        cov[reparam_idx, k] = cov[reparam_idx, k] / sigma8_val
                                        cov[k, reparam_idx] = cov[k, reparam_idx] / sigma8_val
                        elif orig_param_name in params_sigma8_squared:
                            # Variance scales as (1/sigma8^2)^2 = 1/sigma8^4
                            cov[reparam_idx, reparam_idx] = cov[reparam_idx, reparam_idx] / (sigma8_val ** 4)
                            # Update off-diagonal elements involving this parameter
                            for k in range(len(param_names)):
                                if k != reparam_idx:
                                    if k == idx_sigma8:
                                        # Covariance with sigma8 needs special handling
                                        # For now, set to 0 (approximation)
                                        cov[reparam_idx, k] = 0
                                        cov[k, reparam_idx] = 0
                                    else:
                                        # Covariance scales as 1/sigma8^2
                                        cov[reparam_idx, k] = cov[reparam_idx, k] / (sigma8_val ** 2)
                                        cov[k, reparam_idx] = cov[k, reparam_idx] / (sigma8_val ** 2)
                    
                    covs_pred_arr[i][j] = cov
            
            # Update param_names
            param_names = param_names_new
            
            # Reshape back if needed
            if len(theta_pred_arr) == 1:
                theta_pred_arr = theta_pred_arr[0]
                theta_true_arr = theta_true_arr[0]
                covs_pred_arr = covs_pred_arr[0]
    
    if theta_pred_arr.ndim == 2:
        theta_pred_arr = np.array([theta_pred_arr])
        theta_true_arr = np.array([theta_true_arr])
        covs_pred_arr = np.array([covs_pred_arr])
    n_params = theta_pred_arr.shape[-1]

    if param_names_plot is None:
        param_names_plot = param_names
    idxs_plot = [param_names.index(pn) for pn in param_names_plot]
    param_labels = [param_label_dict[pn] for pn in param_names_plot]

    if label_arr is None:
        label_arr = [None] * len(theta_pred_arr)
    if color_arr is None:
        color_arr = ['salmon'] * len(theta_pred_arr)

    if n_cols is None and n_rows is None:
        n_cols = len(param_names_plot)
        n_rows = 1
    row_height = 3 if title is None else 3.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, row_height*n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

    # Collect legend handles and labels from the first subplot
    legend_handles = []
    legend_labels = []

    for ax_idx, pp in enumerate(idxs_plot):
        ax = axes[ax_idx]
        ax.set_title(rf'{param_labels[ax_idx]}', fontsize=22)
        
        for i in range(len(theta_pred_arr)):
            pred_vals = theta_pred_arr[i][:, pp]
            true_vals = theta_true_arr[i][:, pp]
            
            # Extract errors for this parameter
            errs = []
            for cc, cov_pred in enumerate(covs_pred_arr[i]):
                err = np.sqrt(np.diag(cov_pred))
                errs.append(err[pp])
            errs = np.array(errs)
            
            # Filter out NaN values
            mask = ~(np.isnan(pred_vals) | np.isnan(true_vals) | np.isnan(errs))
            pred_vals_clean = pred_vals[mask]
            true_vals_clean = true_vals[mask]
            errs_clean = errs[mask]
            
            if len(pred_vals_clean) == 0:
                continue
            
            # Randomly sample N_plot points if requested
            if N_plot is not None and len(pred_vals_clean) > N_plot:
                indices = np.random.choice(len(pred_vals_clean), size=N_plot, replace=False)
                pred_vals_clean = pred_vals_clean[indices]
                true_vals_clean = true_vals_clean[indices]
                errs_clean = errs_clean[indices]
            
            label = label_arr[i] if label_arr is not None else None
            
            # Plot with error bars
            lines = ax.errorbar(true_vals_clean, pred_vals_clean, yerr=errs_clean,
                       fmt='o', color=color_arr[i], label=label, alpha=alpha,
                       markersize=3, capsize=2, capthick=1)
            
            # Collect legend handles from first subplot only
            if ax_idx == 0 and label is not None:
                # errorbar returns (line, caplines, barlinecols), we want the line
                legend_handles.append(lines[0])
                legend_labels.append(label)
        
        # Add 1:1 line
        # Determine axis limits from data
        all_true = []
        all_pred = []
        for i in range(len(theta_pred_arr)):
            pred_vals = theta_pred_arr[i][:, pp]
            true_vals = theta_true_arr[i][:, pp]
            mask = ~(np.isnan(pred_vals) | np.isnan(true_vals))
            all_true.extend(true_vals[mask])
            all_pred.extend(pred_vals[mask])
        
        if len(all_true) > 0 and len(all_pred) > 0:
            min_val = min(np.min(all_true), np.min(all_pred))
            max_val = max(np.max(all_true), np.max(all_pred))
            line_11, = ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7, label='1:1' if ax_idx == 0 else None)
            if ax_idx == 0:
                legend_handles.append(line_11)
                legend_labels.append('1:1')
        
        ax.set_xlabel(rf'True {param_labels[ax_idx]}', fontsize=14)
        ax.set_ylabel(rf'Predicted {param_labels[ax_idx]}', fontsize=14)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')

    # Place legend outside the plotting area on the right side
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, fontsize=10, 
                  bbox_to_anchor=(1.02, 0.5), loc='center left')

    # Set overall figure title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.9)

    plt.tight_layout()
    plt.show()

    return None


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
    alpha=0.5, histtype='bar', lw=2, n_rows=1, n_cols=None, plot_cdf=False,
    title=None,
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
    row_height = 3 if title is None else 3.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, row_height*n_rows))
    axes = np.array(axes).reshape(-1)

    x_normal = np.linspace(-3, 3, 200)
    mean, variance = 0, 1
    y_normal = norm.pdf(x_normal, mean, np.sqrt(variance))
    cdf_normal = norm.cdf(x_normal, mean, np.sqrt(variance))
    xmin, xmax = -3, 3

    # Collect legend handles and labels from the first subplot
    legend_handles = []
    legend_labels = []
    
    for ax_idx, pp in enumerate(idxs_plot):
        ax = axes[ax_idx]
        ax.set_title(rf'{param_labels[ax_idx]}', fontsize=22)
        for i in range(sigmas_from_truth_arr.shape[0]):
            data = sigmas_from_truth_arr[i][:, pp]
            if plot_cdf:
                sorted_data = np.sort(data)
                yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
                line, = ax.plot(sorted_data, yvals, color=color_arr[i], label=label_arr[i], lw=2)
                if ax_idx == 0:  # Only collect from first subplot to avoid duplicates
                    legend_handles.append(line)
                    legend_labels.append(label_arr[i])
            else:
                n, bins, patches = ax.hist(data, bins=np.linspace(xmin, xmax, nbins),
                                           color=color_arr[i], label=label_arr[i], alpha=alpha, density=True,
                                           histtype=histtype, lw=lw)
                line, = ax.plot(x_normal, y_normal, color='black', lw=1, label=r'$\mathcal{N}(0,1)$' if i == 0 else None)
                if ax_idx == 0 and i == 0:  # Only collect from first subplot and first iteration
                    legend_handles.append(line)
                    legend_labels.append(r'$\mathcal{N}(0,1)$')
        if plot_cdf:
            # Add Gaussian CDF last in legend, only once
            line_cdf, = ax.plot(x_normal, cdf_normal, color='black', lw=1, label='Gaussian CDF', ls='--')
            if ax_idx == 0:  # Only collect from first subplot
                legend_handles.append(line_cdf)
                legend_labels.append('Gaussian CDF')
            ax.set_ylabel('CDF', fontsize=12)
        else:
            ax.set_ylabel(r'normalized density', fontsize=12)
        ax.set_xlabel(rf'$\Delta${param_labels[ax_idx]}/$\sigma$({param_labels[ax_idx]})', fontsize=14)
        ax.axvline(0, color='grey')
        ax.set_xlim(xmin, xmax)

    # Hide unused axes
    for ax in axes[len(idxs_plot):]:
        ax.axis('off')
    
    # Place legend outside the plotting area on the right side
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, fontsize=10, 
                  bbox_to_anchor=(1.02, 0.5), loc='center left')
        
    # Set overall figure title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.9)
        
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
            legend_kwargs={'bbox_to_anchor': (1.05, 1.0)}
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
                      extents={}, title=None, unreparameterize=False):
    if title is None:
        title = f'test model {idx_obs}'
    
    # Get all chains and their available parameters
    all_chains_data = []
    all_param_names_per_chain = []
    
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
            
        all_chains_data.append({
            'samples': samples,
            'param_names_samples': param_names_samples,
            'available_params': available_params,
            'inf_method': inf_method,
            'label': labels[i] if labels is not None else None,
            'color': colors[i] if colors is not None else None
        })
        all_param_names_per_chain.append(available_params)
    
    if len(all_chains_data) == 0:
        print("Error: No valid chains found.")
        return
    
    # Find which parameters are available in at least one chain
    # Preserve the order from param_names, but replace original params with reparameterized versions when found
    param_names_any_available = []
    # Track which original parameters have been replaced by reparameterized versions
    replaced_params = set()
    
    # Define parameters to multiply by sigma_8 vs sigma_8^2
    params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
    params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']
    
    for pn in param_names:
        # Determine the reparameterized name based on parameter type
        if pn in params_sigma8_squared:
            reparam_name = f'sigma8_cold_sq_x_{pn}'
        elif pn in params_sigma8:
            reparam_name = f'sigma8_cold_x_{pn}'
        else:
            reparam_name = None
        
        has_reparam = False
        if reparam_name:
            has_reparam = any(reparam_name in chain_data['param_names_samples'] 
                             for chain_data in all_chains_data)
        has_original = any(pn in chain_params for chain_params in all_param_names_per_chain)
        
        if unreparameterize:
            # Always use original parameter names when unreparameterizing
            if has_reparam or has_original:
                param_names_any_available.append(pn)
        else:
            # Use reparameterized version if available, otherwise original
            if has_reparam:
                # Use reparameterized version if available
                param_names_any_available.append(reparam_name)
                replaced_params.add(pn)
            elif has_original:
                # Use original version
                param_names_any_available.append(pn)
    
    if len(param_names_any_available) == 0:
        print("Error: No requested parameters found in any chains.")
        return
        
    # Check for missing parameters (accounting for reparameterized replacements)
    missing_params = []
    for pn in param_names:
        if unreparameterize:
            # When unreparameterizing, we only check if the original param is available
            if pn not in param_names_any_available:
                missing_params.append(pn)
        else:
            # When not unreparameterizing, check both original and reparameterized
            if pn in params_sigma8_squared:
                reparam_name = f'sigma8_cold_sq_x_{pn}'
            elif pn in params_sigma8:
                reparam_name = f'sigma8_cold_x_{pn}'
            else:
                reparam_name = None
            
            if pn not in param_names_any_available:
                if reparam_name and reparam_name not in param_names_any_available:
                    missing_params.append(pn)
                elif not reparam_name:
                    missing_params.append(pn)
    
    if len(missing_params) > 0:
        if unreparameterize:
            print(f"Warning: Parameters {missing_params} not found in any chains (neither original nor reparameterized). Plotting only: {param_names_any_available}")
        else:
            print(f"Warning: Parameters {missing_params} not found in any chains (neither original nor reparameterized). Plotting only: {param_names_any_available}")
    
    # Create chains for chainconsumer - only include chains that have at least one parameter
    import chainconsumer
    c = chainconsumer.ChainConsumer()
    
    for chain_data in all_chains_data:
        # Get samples for parameters that this chain has
        # If unreparameterize=True, convert reparameterized params back to original
        available_in_this_chain = []
        param_mapping = {}  # Maps original param names to their indices/sources in chain
        
        for pn in param_names_any_available:
            # Check if it's in the chain's param_names_samples (could be original or reparameterized)
            if pn in chain_data['param_names_samples']:
                available_in_this_chain.append(pn)
                param_mapping[pn] = pn  # Direct mapping
            elif unreparameterize:
                # Check if there's a reparameterized version we can convert back
                if pn in params_sigma8_squared:
                    reparam_name = f'sigma8_cold_sq_x_{pn}'
                elif pn in params_sigma8:
                    reparam_name = f'sigma8_cold_x_{pn}'
                else:
                    reparam_name = None
                
                if reparam_name and reparam_name in chain_data['param_names_samples']:
                    available_in_this_chain.append(pn)
                    param_mapping[pn] = reparam_name  # Map original name to reparameterized name
        
        if len(available_in_this_chain) == 0:
            continue
        
        # Check if we need sigma8_cold for unreparameterization
        need_sigma8 = False
        if unreparameterize:
            for pn in available_in_this_chain:
                if pn in param_mapping and (param_mapping[pn].startswith('sigma8_cold_x_') or param_mapping[pn].startswith('sigma8_cold_sq_x_')):
                    need_sigma8 = True
                    break
        
        # Get sigma8_cold if needed
        sigma8_samples = None
        if need_sigma8:
            if 'sigma8_cold' in chain_data['param_names_samples']:
                idx_sigma8 = list(chain_data['param_names_samples']).index('sigma8_cold')
                sigma8_samples = chain_data['samples'][:, idx_sigma8]
            else:
                print(f"Warning: Need sigma8_cold for unreparameterization but not found in chain. Skipping unreparameterization for this chain.")
                need_sigma8 = False
        
        # Extract samples for available parameters
        samples_dict = {}
        for pn in available_in_this_chain:
            if pn in param_mapping:
                source_name = param_mapping[pn]
                if source_name == pn:
                    # Direct parameter - just extract
                    idx = list(chain_data['param_names_samples']).index(pn)
                    samples_dict[pn] = chain_data['samples'][:, idx]
                elif (source_name.startswith('sigma8_cold_x_') or source_name.startswith('sigma8_cold_sq_x_')) and need_sigma8:
                    # Reparameterized parameter - convert back to original
                    idx_reparam = list(chain_data['param_names_samples']).index(source_name)
                    reparam_samples = chain_data['samples'][:, idx_reparam]
                    
                    # Extract original parameter name
                    if source_name.startswith('sigma8_cold_sq_x_'):
                        orig_param_name = source_name.replace('sigma8_cold_sq_x_', '')
                    else:
                        orig_param_name = source_name.replace('sigma8_cold_x_', '')
                    
                    # Determine if we divide by sigma8 or sigma8^2
                    params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
                    params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']
                    
                    if orig_param_name in params_sigma8:
                        # Divide by sigma8
                        samples_dict[pn] = reparam_samples / sigma8_samples
                    elif orig_param_name in params_sigma8_squared:
                        # Divide by sigma8^2
                        samples_dict[pn] = reparam_samples / (sigma8_samples ** 2)
                    else:
                        print(f"Warning: Unknown reparameterized parameter {source_name}. Skipping.")
                else:
                    # Shouldn't happen, but handle gracefully
                    idx = list(chain_data['param_names_samples']).index(source_name)
                    samples_dict[pn] = chain_data['samples'][:, idx]
        
        samples_df = pd.DataFrame(samples_dict)
        
        # Get color and smoothing settings
        color = chain_data['color']
        if color is None:
            color = utils.color_dict_methods[chain_data['inf_method']]
        
        # smooth = smooth_dict[chain_data['inf_method']]
        # bins = bins_dict[chain_data['inf_method']]
        smooth = 4
        bins = 8
        
        c.add_chain(chainconsumer.Chain(
            samples=samples_df,
            name=chain_data['label'],
            color=color,
            smooth=smooth,
            bins=bins,
            #plot_cloud=True,
            plot_cloud=False,
        ))

    # Set up plot configuration
    c.set_plot_config(
        chainconsumer.PlotConfig(
            flip=True,
            labels=utils.param_label_dict,
            #contour_label_font_size=12,
            summary_font_size=0,
            extents=extents,
            legend_kwargs={'bbox_to_anchor': (1.05, 1.0), 
                           'fontsize':18}
        )
    )

    # c.set_override(
    #     chainconsumer.ChainConfig(
    #         shade_alpha=1.0,
    #         sigmas=[0,1,2],
    #         shade_gradient=1.0,
    #         plot_cloud=True,
    #     )
    # )
    
    # Create truth location for available parameters only
    # Note: param_names and theta_obs_true are in original (non-reparameterized) form,
    # but param_names_any_available may contain reparameterized parameter names from the chains
    truth_loc = {}
    for pn in param_names_any_available:
        # Check if this is a reparameterized parameter (from the chains)
        if pn.startswith('sigma8_cold_x_') or pn.startswith('sigma8_cold_sq_x_'):
            # Extract the original parameter name
            if pn.startswith('sigma8_cold_sq_x_'):
                orig_param_name = pn.replace('sigma8_cold_sq_x_', '')
            else:
                orig_param_name = pn.replace('sigma8_cold_x_', '')
            
            # Check if we have sigma8_cold and the original parameter in param_names (original form)
            if 'sigma8_cold' in param_names and orig_param_name in param_names:
                idx_sigma8 = param_names.index('sigma8_cold')
                idx_orig = param_names.index(orig_param_name)
                
                if idx_sigma8 < len(theta_obs_true) and idx_orig < len(theta_obs_true):
                    sigma8_val = theta_obs_true[idx_sigma8]
                    orig_val = theta_obs_true[idx_orig]
                    
                    # Determine if we multiply by sigma8 or sigma8^2
                    params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
                    params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']
                    
                    if orig_param_name in params_sigma8:
                        truth_loc[pn] = orig_val * sigma8_val
                    elif orig_param_name in params_sigma8_squared:
                        truth_loc[pn] = orig_val * (sigma8_val ** 2)
                    else:
                        print(f"Warning: Unknown reparameterized parameter {pn}. Skipping truth value.")
                else:
                    print(f"Warning: Indices out of range for reparameterized parameter {pn}. Skipping truth value.")
            else:
                print(f"Warning: Cannot compute truth for reparameterized parameter {pn}: missing 'sigma8_cold' or '{orig_param_name}' in param_names. Skipping truth value.")
        else:
            # Regular parameter - check if it's in the original param_names
            if pn in param_names:
                param_idx = param_names.index(pn)
                if param_idx < len(theta_obs_true):
                    truth_loc[pn] = theta_obs_true[param_idx]
                else:
                    print(f"Warning: Parameter {pn} (index {param_idx}) not found in theta_obs_true (length {len(theta_obs_true)}). Skipping truth value.")
            else:
                # Parameter is in chains but not in original param_names - might be a different naming
                # Try to find it or skip
                print(f"Warning: Parameter {pn} found in chains but not in original param_names. Skipping truth value.")
    
    c.add_truth(chainconsumer.Truth(location=truth_loc))

    # Plot - ChainConsumer will automatically show all parameters that appear in any chain
    # and leave subplots blank for parameter combinations where data is missing
    fig = c.plotter.plot(figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # Don't return the figure to avoid duplicate display in notebooks
    plt.show()
    return None


def plot_contours_inf_reparam(param_names, idx_obs, theta_obs_true,
                      inf_methods, tags_inf, tags_test=None,
                      colors=None, labels=None,
                      figsize=(7,7),
                      extents={}, title=None, reparameterize=False,
                      params_show_reparameterize=None):
    if title is None:
        title = f'test model {idx_obs}'
    
    if reparameterize:
        assert 'sigma8_cold' in param_names, "reparameterize=True requires 'sigma8_cold' in param_names"
        assert 'b1' in param_names, "reparameterize=True requires 'b1' in param_names"
        if params_show_reparameterize is not None:
            assert params_show_reparameterize == ['sigma8xb1', 'sigma8_cold'] or params_show_reparameterize == ['sigma8xb1', 'b1'], \
                "params_show_reparameterize must be either ['sigma8xb1', 'sigma8_cold'] or ['sigma8xb1', 'b1']"
            assert params_show_reparameterize[0] == 'sigma8xb1', "sigma8xb1 must be first in params_show_reparameterize"
    
    # Get all chains and their available parameters
    all_chains_data = []
    all_param_names_per_chain = []
    
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
            
        all_chains_data.append({
            'samples': samples,
            'param_names_samples': param_names_samples,
            'available_params': available_params,
            'inf_method': inf_method,
            'label': labels[i] if labels is not None else None,
            'color': colors[i] if colors is not None else None
        })
        all_param_names_per_chain.append(available_params)
    
    if len(all_chains_data) == 0:
        print("Error: No valid chains found.")
        return
    
    # Find which parameters are available in at least one chain
    param_names_any_available = []
    for pn in param_names:
        if any(pn in chain_params for chain_params in all_param_names_per_chain):
            param_names_any_available.append(pn)
    
    if len(param_names_any_available) == 0:
        print("Error: No requested parameters found in any chains.")
        return
        
    if len(param_names_any_available) < len(param_names):
        missing_params = [pn for pn in param_names if pn not in param_names_any_available]
        print(f"Warning: Parameters {missing_params} not found in any chains. Plotting only: {param_names_any_available}")
    
    # Handle reparameterization: insert sigma8xb1 at appropriate position, keep all other parameters in original order
    if reparameterize:
        # Remove the parameter that's NOT being shown (between sigma8_cold and b1)
        if params_show_reparameterize is not None:
            if 'sigma8_cold' in params_show_reparameterize:
                # Showing sigma8xb1 and sigma8_cold, so remove b1
                if 'b1' in param_names_any_available:
                    param_names_any_available.remove('b1')
            elif 'b1' in params_show_reparameterize:
                # Showing sigma8xb1 and b1, so remove sigma8_cold
                if 'sigma8_cold' in param_names_any_available:
                    param_names_any_available.remove('sigma8_cold')
        else:
            # Default: remove b1 (replaced by sigma8xb1)
            if 'b1' in param_names_any_available:
                param_names_any_available.remove('b1')
        
        # Insert sigma8xb1 at the appropriate position
        if 'sigma8xb1' not in param_names_any_available:
            if params_show_reparameterize is not None:
                if 'sigma8_cold' in params_show_reparameterize and 'sigma8_cold' in param_names_any_available:
                    # Insert after sigma8_cold
                    idx = param_names_any_available.index('sigma8_cold')
                    param_names_any_available.insert(idx + 1, 'sigma8xb1')
                elif 'b1' in params_show_reparameterize and 'b1' in param_names_any_available:
                    # Insert before b1
                    idx = param_names_any_available.index('b1')
                    param_names_any_available.insert(idx, 'sigma8xb1')
                else:
                    # Fallback: insert at the beginning
                    param_names_any_available.insert(0, 'sigma8xb1')
            else:
                # Default: insert before where b1 was (which we just removed)
                # Since b1 is removed, insert where sigma8_cold is (after it)
                if 'sigma8_cold' in param_names_any_available:
                    idx = param_names_any_available.index('sigma8_cold')
                    param_names_any_available.insert(idx + 1, 'sigma8xb1')
                else:
                    param_names_any_available.insert(0, 'sigma8xb1')
    
    # Create chains for chainconsumer - only include chains that have at least one parameter
    import chainconsumer
    c = chainconsumer.ChainConsumer()
    
    for chain_data in all_chains_data:
        # Build list of parameters to include for this chain
        available_in_this_chain = []
        for pn in param_names_any_available:
            if pn == 'sigma8xb1':
                # Only include sigma8xb1 if we can compute it (both sigma8_cold and b1 available)
                if reparameterize and 'sigma8_cold' in chain_data['available_params'] and 'b1' in chain_data['available_params']:
                    available_in_this_chain.append(pn)
            else:
                # Include other parameters if available in the chain
                if pn in chain_data['available_params']:
                    available_in_this_chain.append(pn)
        
        if len(available_in_this_chain) == 0:
            continue
            
        # Extract samples for available parameters, maintaining original order
        samples_dict = {}
        for pn in available_in_this_chain:
            if reparameterize and pn == 'sigma8xb1':
                # Compute sigma8xb1 from sigma8_cold and b1
                i_sigma8 = list(chain_data['param_names_samples']).index('sigma8_cold')
                i_b1 = list(chain_data['param_names_samples']).index('b1')
                sigma8_samples = chain_data['samples'][:, i_sigma8]
                b1_samples = chain_data['samples'][:, i_b1]
                samples_dict[pn] = sigma8_samples * b1_samples
            else:
                # Regular parameter extraction
                i_pn = list(chain_data['param_names_samples']).index(pn)
                samples_dict[pn] = chain_data['samples'][:, i_pn]
        
        samples_df = pd.DataFrame(samples_dict)
        
        # Get color and smoothing settings
        color = chain_data['color']
        if color is None:
            color = utils.color_dict_methods[chain_data['inf_method']]
        
        # smooth = smooth_dict[chain_data['inf_method']]
        # bins = bins_dict[chain_data['inf_method']]
        smooth = 4
        bins = 8
        
        c.add_chain(chainconsumer.Chain(
            samples=samples_df,
            name=chain_data['label'],
            color=color,
            smooth=smooth,
            bins=bins,
            #plot_cloud=True,
            plot_cloud=False,
        ))

    # Set up plot configuration
    c.set_plot_config(
        chainconsumer.PlotConfig(
            flip=True,
            labels=utils.param_label_dict,
            #contour_label_font_size=12,
            summary_font_size=0,
            extents=extents,
            legend_kwargs={'bbox_to_anchor': (1.05, 1.0), 
                           'fontsize':18}
        )
    )

    # c.set_override(
    #     chainconsumer.ChainConfig(
    #         shade_alpha=1.0,
    #         sigmas=[0,1,2],
    #         shade_gradient=1.0,
    #         plot_cloud=True,
    #     )
    # )
    
    # Create truth location for available parameters only
    truth_loc = {}
    for pn in param_names_any_available:
        if reparameterize and pn == 'sigma8xb1':
            # Compute sigma8xb1 from truth values
            idx_sigma8 = param_names.index('sigma8_cold')
            idx_b1 = param_names.index('b1')
            if idx_sigma8 < len(theta_obs_true) and idx_b1 < len(theta_obs_true):
                truth_loc[pn] = theta_obs_true[idx_sigma8] * theta_obs_true[idx_b1]
        elif pn in param_names:
            param_idx = param_names.index(pn)
            if param_idx < len(theta_obs_true):
                truth_loc[pn] = theta_obs_true[param_idx]
            else:
                print(f"Warning: Parameter {pn} (index {param_idx}) not found in theta_obs_true (length {len(theta_obs_true)}). Skipping truth value.")
    
    c.add_truth(chainconsumer.Truth(location=truth_loc))

    # Plot - ChainConsumer will automatically show all parameters that appear in any chain
    # and leave subplots blank for parameter combinations where data is missing
    fig = c.plotter.plot(figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # Don't return the figure to avoid duplicate display in notebooks
    plt.show()
    return None
     
    
    
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