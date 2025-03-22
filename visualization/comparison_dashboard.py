#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WMAP vs Planck Comparison Dashboard

This module provides visualization tools for comparing analysis results
between WMAP and Planck CMB data.
"""

from __future__ import print_function
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path for importing
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Set plot style
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def create_custom_colormap():
    """Create a custom colormap for significance visualization."""
    # Define colors for p-values
    # Red: p < 0.01 (highly significant)
    # Orange: 0.01 <= p < 0.05 (significant)
    # Yellow: 0.05 <= p < 0.1 (marginally significant)
    # White/Gray: p >= 0.1 (not significant)
    colors = [(0.8, 0.1, 0.1), (0.9, 0.6, 0.1), (0.9, 0.9, 0.2), (0.95, 0.95, 0.95)]
    positions = [0, 0.33, 0.67, 1]
    
    return LinearSegmentedColormap.from_list('significance_cmap', list(zip(positions, colors)))


def plot_phi_optimality_comparison(wmap_results, planck_results, ax=None):
    """
    Plot comparison of phi optimality scores between WMAP and Planck.
    
    Args:
        wmap_results (dict): WMAP analysis results
        planck_results (dict): Planck analysis results
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract phi optimality scores for each test
    test_names = []
    wmap_phis = []
    planck_phis = []
    
    for test_name in wmap_results['phi_optimality']:
        if test_name in planck_results['phi_optimality']:
            wmap_phi = wmap_results['phi_optimality'][test_name]
            planck_phi = planck_results['phi_optimality'][test_name]
            
            # Handle different data types
            if not isinstance(wmap_phi, dict) and not isinstance(planck_phi, dict):
                test_names.append(test_name.replace('_', ' ').title())
                wmap_phis.append(wmap_phi)
                planck_phis.append(planck_phi)
    
    # Create bar chart
    x = np.arange(len(test_names))
    width = 0.35
    
    ax.bar(x - width/2, wmap_phis, width, label='WMAP', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, planck_phis, width, label='Planck', color='#ff7f0e', alpha=0.8)
    
    # Add correlation coefficient
    if wmap_phis and planck_phis:
        corr = np.corrcoef(wmap_phis, planck_phis)[0, 1]
        ax.text(0.02, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_xlabel('Test')
    ax.set_ylabel('Phi Optimality Score')
    ax.set_title('Comparison of Phi Optimality Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)  # Phi optimality ranges from -1 to 1
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    return ax


def plot_significance_comparison(wmap_results, planck_results, ax=None):
    """
    Plot comparison of significance (p-values) between WMAP and Planck.
    
    Args:
        wmap_results (dict): WMAP analysis results
        planck_results (dict): Planck analysis results
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract p-values for each test
    test_names = []
    wmap_pvals = []
    planck_pvals = []
    
    for test_name in wmap_results['significance']:
        if test_name in planck_results['significance']:
            test_names.append(test_name.replace('_', ' ').title())
            wmap_pvals.append(wmap_results['significance'][test_name])
            planck_pvals.append(planck_results['significance'][test_name])
    
    # Create bar chart
    x = np.arange(len(test_names))
    width = 0.35
    
    ax.bar(x - width/2, wmap_pvals, width, label='WMAP', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, planck_pvals, width, label='Planck', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Test')
    ax.set_ylabel('P-value')
    ax.set_title('Comparison of Statistical Significance')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add significance thresholds
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
    
    # Use log scale for better visualization of small p-values
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.0)
    
    return ax


def plot_constants_comparison(wmap_results, planck_results, ax=None):
    """
    Plot comparison of best constants between WMAP and Planck.
    
    Args:
        wmap_results (dict): WMAP analysis results
        planck_results (dict): Planck analysis results
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if best_constants exists in results
    if 'best_constants' not in wmap_results or 'best_constants' not in planck_results:
        ax.text(0.5, 0.5, 'Best constants data not available', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Extract best constants data
    wmap_constants = wmap_results['best_constants']
    planck_constants = planck_results['best_constants']
    
    # Get common constants
    common_constants = set(wmap_constants.keys()) & set(planck_constants.keys())
    
    if not common_constants:
        ax.text(0.5, 0.5, 'No common constants found', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Prepare data for plotting
    constants = sorted(list(common_constants))
    wmap_values = [wmap_constants[c] for c in constants]
    planck_values = [planck_constants[c] for c in constants]
    
    # Create bar chart
    x = np.arange(len(constants))
    width = 0.35
    
    ax.bar(x - width/2, wmap_values, width, label='WMAP', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, planck_values, width, label='Planck', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Mathematical Constant')
    ax.set_ylabel('Frequency')
    ax.set_title('Comparison of Best Mathematical Constants')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in constants], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_combined_significance(wmap_results, planck_results, ax=None):
    """
    Plot comparison of combined significance between WMAP and Planck.
    
    Args:
        wmap_results (dict): WMAP analysis results
        planck_results (dict): Planck analysis results
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Check if combined_significance exists in results
    if 'combined_significance' not in wmap_results or 'combined_significance' not in planck_results:
        ax.text(0.5, 0.5, 'Combined significance data not available', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Extract combined significance data
    wmap_combined = wmap_results['combined_significance']
    planck_combined = planck_results['combined_significance']
    
    # Create a simple comparison visualization
    labels = ['WMAP', 'Planck']
    values = [wmap_combined, planck_combined]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    
    # Use color based on significance
    colors = ['green' if v < 0.05 else 'red' for v in values]
    
    ax.barh(y_pos, values, align='center', color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Combined P-value')
    ax.set_title('Combined Statistical Significance')
    
    # Add significance thresholds
    ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axvline(x=0.01, color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
    
    # Add text labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    return ax


def plot_data_comparison(wmap_data, planck_data, ax=None):
    """
    Plot comparison of raw data between WMAP and Planck.
    
    Args:
        wmap_data (numpy.ndarray): WMAP data
        planck_data (numpy.ndarray): Planck data
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure both datasets have the same length for plotting
    min_length = min(len(wmap_data), len(planck_data))
    wmap_data = wmap_data[:min_length]
    planck_data = planck_data[:min_length]
    
    # Plot a sample of the data (first 100 points)
    sample_size = min(100, min_length)
    x = np.arange(sample_size)
    
    ax.plot(x, wmap_data[:sample_size], label='WMAP', color='#1f77b4', alpha=0.8)
    ax.plot(x, planck_data[:sample_size], label='Planck', color='#ff7f0e', alpha=0.8)
    
    # Calculate and display correlation
    corr = np.corrcoef(wmap_data, planck_data)[0, 1]
    ax.text(0.02, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Raw Data Comparison (First 100 Points)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_power_spectrum_comparison(wmap_data, planck_data, ax=None):
    """
    Plot comparison of power spectra between WMAP and Planck.
    
    Args:
        wmap_data (numpy.ndarray): WMAP data
        planck_data (numpy.ndarray): Planck data
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
        
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure both datasets have the same length
    min_length = min(len(wmap_data), len(planck_data))
    wmap_data = wmap_data[:min_length]
    planck_data = planck_data[:min_length]
    
    # Calculate power spectra
    from scipy import signal
    
    # Use a reasonable segment size for the data
    nperseg = min(1024, min_length//4)
    
    wmap_f, wmap_psd = signal.welch(wmap_data, nperseg=nperseg)
    planck_f, planck_psd = signal.welch(planck_data, nperseg=nperseg)
    
    # Plot power spectra
    ax.semilogy(wmap_f, wmap_psd, label='WMAP', color='#1f77b4', alpha=0.8)
    ax.semilogy(planck_f, planck_psd, label='Planck', color='#ff7f0e', alpha=0.8)
    
    # Calculate and display power ratio
    power_ratio = np.sum(wmap_psd) / np.sum(planck_psd)
    ax.text(0.02, 0.95, f'Power Ratio (WMAP/Planck): {power_ratio:.3f}', 
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Power Spectrum Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def create_comparison_dashboard(wmap_results_file, planck_results_file, 
                               wmap_data_file=None, planck_data_file=None,
                               output_dir=None):
    """
    Create a comprehensive dashboard comparing WMAP and Planck results.
    
    Args:
        wmap_results_file (str): Path to WMAP results JSON file
        planck_results_file (str): Path to Planck results JSON file
        wmap_data_file (str): Path to WMAP data file (optional)
        planck_data_file (str): Path to Planck data file (optional)
        output_dir (str): Directory to save the dashboard
        
    Returns:
        str: Path to the saved dashboard image
    """
    # Load results
    with open(wmap_results_file, 'r') as f:
        wmap_results = json.load(f)
    
    with open(planck_results_file, 'r') as f:
        planck_results = json.load(f)
    
    # Load data if provided
    wmap_data = None
    planck_data = None
    
    if wmap_data_file and os.path.exists(wmap_data_file):
        wmap_data = np.load(wmap_data_file)
    
    if planck_data_file and os.path.exists(planck_data_file):
        planck_data = np.load(planck_data_file)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Plot phi optimality comparison
    ax1 = fig.add_subplot(gs[0, :])
    plot_phi_optimality_comparison(wmap_results, planck_results, ax1)
    
    # Plot significance comparison
    ax2 = fig.add_subplot(gs[1, :2])
    plot_significance_comparison(wmap_results, planck_results, ax2)
    
    # Plot combined significance
    ax3 = fig.add_subplot(gs[1, 2])
    plot_combined_significance(wmap_results, planck_results, ax3)
    
    # Plot constants comparison
    ax4 = fig.add_subplot(gs[2, :2])
    plot_constants_comparison(wmap_results, planck_results, ax4)
    
    # Plot data comparison if data is available
    if wmap_data is not None and planck_data is not None:
        ax5 = fig.add_subplot(gs[2, 2])
        plot_data_comparison(wmap_data, planck_data, ax5)
    
    # Add title and adjust layout
    fig.suptitle('WMAP vs Planck: Cosmic Consciousness Analysis Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save the dashboard
    if output_dir is None:
        output_dir = os.path.dirname(wmap_results_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dashboard_path = os.path.join(output_dir, 'wmap_planck_comparison_dashboard.png')
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("Comparison dashboard saved to: {}".format(dashboard_path))
    return dashboard_path


def create_detailed_comparison_report(wmap_results_file, planck_results_file, output_dir=None):
    """
    Create a detailed text report comparing WMAP and Planck results.
    
    Args:
        wmap_results_file (str): Path to WMAP results JSON file
        planck_results_file (str): Path to Planck results JSON file
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the saved report file
    """
    # Load results
    with open(wmap_results_file, 'r') as f:
        wmap_results = json.load(f)
    
    with open(planck_results_file, 'r') as f:
        planck_results = json.load(f)
    
    # Create report
    report = []
    report.append("=" * 80)
    report.append("WMAP vs Planck: Cosmic Consciousness Analysis Comparison Report")
    report.append("=" * 80)
    report.append("")
    
    # Add phi optimality comparison
    report.append("Phi Optimality Comparison:")
    report.append("-" * 40)
    
    for test_name in sorted(wmap_results['phi_optimality'].keys()):
        if test_name in planck_results['phi_optimality']:
            wmap_phi = wmap_results['phi_optimality'][test_name]
            planck_phi = planck_results['phi_optimality'][test_name]
            
            # Handle different data types
            if not isinstance(wmap_phi, dict) and not isinstance(planck_phi, dict):
                report.append("{:<30} WMAP: {:.4f}  Planck: {:.4f}  Ratio: {:.4f}".format(
                    test_name.replace('_', ' ').title(),
                    wmap_phi,
                    planck_phi,
                    wmap_phi/planck_phi if planck_phi != 0 else float('inf')
                ))
    
    report.append("")
    
    # Add significance comparison
    report.append("Statistical Significance Comparison (p-values):")
    report.append("-" * 40)
    
    for test_name in sorted(wmap_results['significance'].keys()):
        if test_name in planck_results['significance']:
            wmap_sig = wmap_results['significance'][test_name]
            planck_sig = planck_results['significance'][test_name]
            
            # Add significance indicators
            wmap_indicator = "***" if wmap_sig < 0.01 else "**" if wmap_sig < 0.05 else "*" if wmap_sig < 0.1 else ""
            planck_indicator = "***" if planck_sig < 0.01 else "**" if planck_sig < 0.05 else "*" if planck_sig < 0.1 else ""
            
            report.append("{:<30} WMAP: {:.4f} {}  Planck: {:.4f} {}".format(
                test_name.replace('_', ' ').title(),
                wmap_sig, wmap_indicator,
                planck_sig, planck_indicator
            ))
    
    report.append("")
    report.append("Significance indicators: *** p<0.01, ** p<0.05, * p<0.1")
    report.append("")
    
    # Add combined significance
    if 'combined_significance' in wmap_results and 'combined_significance' in planck_results:
        report.append("Combined Statistical Significance:")
        report.append("-" * 40)
        
        wmap_combined = wmap_results['combined_significance']
        planck_combined = planck_results['combined_significance']
        
        # Add significance indicators
        wmap_indicator = "***" if wmap_combined < 0.01 else "**" if wmap_combined < 0.05 else "*" if wmap_combined < 0.1 else ""
        planck_indicator = "***" if planck_combined < 0.01 else "**" if planck_combined < 0.05 else "*" if planck_combined < 0.1 else ""
        
        report.append("WMAP:   {:.6f} {}".format(wmap_combined, wmap_indicator))
        report.append("Planck: {:.6f} {}".format(planck_combined, planck_indicator))
        report.append("")
    
    # Add best constants comparison
    if 'best_constants' in wmap_results and 'best_constants' in planck_results:
        report.append("Best Mathematical Constants Comparison:")
        report.append("-" * 40)
        
        wmap_constants = wmap_results['best_constants']
        planck_constants = planck_results['best_constants']
        
        # Get common constants
        common_constants = set(wmap_constants.keys()) & set(planck_constants.keys())
        
        for const in sorted(common_constants):
            wmap_val = wmap_constants[const]
            planck_val = planck_constants[const]
            
            report.append("{:<20} WMAP: {:<5}  Planck: {:<5}".format(
                const.replace('_', ' ').title(),
                wmap_val,
                planck_val
            ))
        
        report.append("")
    
    # Add conclusion
    report.append("Conclusion:")
    report.append("-" * 40)
    
    # Calculate overall correlation of phi optimality scores
    wmap_phi_values = []
    planck_phi_values = []
    
    for test_name in wmap_results['phi_optimality']:
        if test_name in planck_results['phi_optimality']:
            wmap_phi = wmap_results['phi_optimality'][test_name]
            planck_phi = planck_results['phi_optimality'][test_name]
            
            if not isinstance(wmap_phi, dict) and not isinstance(planck_phi, dict):
                wmap_phi_values.append(wmap_phi)
                planck_phi_values.append(planck_phi)
    
    if wmap_phi_values and planck_phi_values:
        correlation = np.corrcoef(wmap_phi_values, planck_phi_values)[0, 1]
        report.append("Overall correlation between WMAP and Planck phi optimality scores: {:.4f}".format(correlation))
        
        if correlation > 0.7:
            report.append("The high correlation suggests strong agreement between WMAP and Planck results,")
            report.append("indicating that the observed patterns are consistent across different CMB datasets.")
        elif correlation > 0.3:
            report.append("The moderate correlation suggests partial agreement between WMAP and Planck results,")
            report.append("with some consistent patterns across different CMB datasets.")
        else:
            report.append("The low correlation suggests limited agreement between WMAP and Planck results,")
            report.append("indicating that the observed patterns may be dataset-specific.")
    
    report.append("")
    report.append("=" * 80)
    
    # Save the report
    if output_dir is None:
        output_dir = os.path.dirname(wmap_results_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, 'wmap_planck_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print("Comparison report saved to: {}".format(report_path))
    return report_path


if __name__ == "__main__":
    """
    Main function to demonstrate usage of the comparison dashboard.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='WMAP vs Planck Comparison Dashboard')
    parser.add_argument('--wmap-results', type=str, required=True,
                        help='Path to WMAP results JSON file')
    parser.add_argument('--planck-results', type=str, required=True,
                        help='Path to Planck results JSON file')
    parser.add_argument('--wmap-data', type=str, default=None,
                        help='Path to WMAP data file (optional)')
    parser.add_argument('--planck-data', type=str, default=None,
                        help='Path to Planck data file (optional)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the dashboard and report')
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard_path = create_comparison_dashboard(
        args.wmap_results,
        args.planck_results,
        args.wmap_data,
        args.planck_data,
        args.output_dir
    )
    
    # Create report
    report_path = create_detailed_comparison_report(
        args.wmap_results,
        args.planck_results,
        args.output_dir
    )
    
    print("\nComparison complete.")
    print("Dashboard: {}".format(dashboard_path))
    print("Report: {}".format(report_path))
