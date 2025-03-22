#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for analyzing WMAP data with the Cosmic Consciousness Analysis framework.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime

# Add parent directory to path for importing from Cosmic_Consciousness_Analysis
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cosmic_analysis_path = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis')
sys.path.append(cosmic_analysis_path)

# Try to import from Cosmic Consciousness Analysis framework
try:
    from analysis.analysis import Analyzer, CosmicConsciousnessAnalyzer
    from core_framework.data_handler import get_simulated_data
    HAS_COSMIC_FRAMEWORK = True
except ImportError:
    print("Warning: Could not import Cosmic Consciousness Analysis framework.")
    print("Make sure the framework is installed at: {}".format(cosmic_analysis_path))
    HAS_COSMIC_FRAMEWORK = False


def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to the WMAP power spectrum file
            
    Returns:
        tuple: (ell, power, error) arrays
    """
    try:
        # Load the power spectrum data
        # WMAP binned power spectrum format:
        # Column 1 = mean multipole moment l for the bin
        # Column 2 = smallest l contributing to the bin
        # Column 3 = largest l contributing to the bin
        # Column 4 = mean value of TT power spectrum (l(l+1)/2pi * C_l)
        # Column 5 = error for binned value
        data = np.loadtxt(file_path, comments='#')
        
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value (l(l+1)Cl/2π in μK²)
        error = data[:, 4]  # Error
        
        return ell, power, error
        
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """
    Preprocess data for analysis.
    
    Args:
        data (numpy.ndarray): Input data array
        smooth (bool): Whether to apply smoothing
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Preprocessed data
    """
    processed_data = data.copy()
    
    # Apply smoothing if requested
    if smooth:
        window = np.ones(smooth_window) / smooth_window
        processed_data = np.convolve(processed_data, window, mode='same')
    
    # Remove linear trend if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Normalize if requested
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def analyze_golden_ratio(data, n_shuffles=1000):
    """
    Simple test to analyze golden ratio patterns in the data.
    
    Args:
        data (numpy.ndarray): Input data array
        n_shuffles (int): Number of shuffles for significance testing
        
    Returns:
        dict: Analysis results
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate correlations between data points separated by golden ratio
    correlations = []
    for i in range(len(data) - int(phi)):
        j = i + int(phi)
        correlations.append(data[i] * data[j])
    
    # Calculate mean correlation
    mean_correlation = np.mean(correlations)
    
    # Perform significance testing
    shuffled_correlations = []
    for _ in range(n_shuffles):
        shuffled_data = np.random.permutation(data)
        shuffle_corrs = []
        for i in range(len(shuffled_data) - int(phi)):
            j = i + int(phi)
            shuffle_corrs.append(shuffled_data[i] * shuffled_data[j])
        shuffled_correlations.append(np.mean(shuffle_corrs))
    
    # Calculate p-value
    p_value = np.mean([1 if abs(sc) >= abs(mean_correlation) else 0 for sc in shuffled_correlations])
    
    return {
        'correlation': mean_correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_cosmic_analysis(data, output_dir):
    """
    Run the Cosmic Consciousness Analysis on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        
    Returns:
        dict: Analysis results
    """
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        return None
    
    # Create analyzer
    analyzer = CosmicConsciousnessAnalyzer(
        data=data,
        output_dir=output_dir,
        phi_bias=0.1
    )
    
    # Run all tests
    print("\nRunning Cosmic Consciousness Analysis on WMAP data...")
    results = analyzer.run_analysis(
        tests='all',
        visualize=True,
        report=True,
        parallel=True,
        n_jobs=-1
    )
    
    return results


def main():
    # Path to WMAP power spectrum file
    power_spectrum_file = 'wmap_data/raw_data/wmap_binned_tt_spectrum_9yr_v5.txt'
    
    # Check if file exists
    if not os.path.exists(power_spectrum_file):
        print("Error: WMAP power spectrum file not found: {}".format(power_spectrum_file))
        return 1
    
    # Load WMAP power spectrum
    print("Loading WMAP power spectrum...")
    ell, power, error = load_wmap_power_spectrum(power_spectrum_file)
    
    if ell is None:
        print("Error loading WMAP power spectrum.")
        return 1
    
    print("Loaded WMAP power spectrum with {} multipoles".format(len(ell)))
    
    # Preprocess power spectrum data
    print("Preprocessing data...")
    processed_power = preprocess_data(
        power, 
        smooth=True, 
        smooth_window=5, 
        normalize=True, 
        detrend=True
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', "wmap_test_{}".format(timestamp))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze golden ratio patterns
    print("Analyzing golden ratio patterns...")
    results = analyze_golden_ratio(processed_power, n_shuffles=1000)
    
    print("\nBasic Analysis Results:")
    print("  Golden Ratio Correlation: {:.4f}".format(results['correlation']))
    print("  P-value: {:.4f}".format(results['p_value']))
    print("  Significant: {}".format(results['significant']))
    
    # Save results to file
    with open(os.path.join(output_dir, 'basic_analysis_results.txt'), 'w') as f:
        f.write("WMAP Basic Analysis Results\n")
        f.write("==========================\n\n")
        f.write("Golden Ratio Correlation: {:.4f}\n".format(results['correlation']))
        f.write("P-value: {:.4f}\n".format(results['p_value']))
        f.write("Significant: {}\n".format(results['significant']))
    
    # Plot the power spectrum
    plt.figure(figsize=(12, 6))
    
    # Original power spectrum
    plt.subplot(2, 1, 1)
    plt.errorbar(ell, power, yerr=error, fmt='o', markersize=3, alpha=0.5)
    plt.xlabel('Multipole (l)')
    plt.ylabel('Power (uK^2)')
    plt.title('WMAP 9-year Temperature Power Spectrum')
    plt.grid(True, alpha=0.3)
    
    # Processed power spectrum
    plt.subplot(2, 1, 2)
    plt.plot(ell, processed_power)
    plt.xlabel('Multipole (l)')
    plt.ylabel('Normalized Power')
    plt.title('Processed Power Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'wmap_power_spectrum.png'), dpi=150)
    print("Plot saved to: {}".format(os.path.join(output_dir, 'wmap_power_spectrum.png')))
    
    # Run full Cosmic Consciousness Analysis if framework is available
    if HAS_COSMIC_FRAMEWORK:
        cosmic_output_dir = os.path.join(output_dir, 'cosmic_analysis')
        if not os.path.exists(cosmic_output_dir):
            os.makedirs(cosmic_output_dir)
            
        cosmic_results = run_cosmic_analysis(processed_power, cosmic_output_dir)
        
        if cosmic_results:
            print("\nCosmic Consciousness Analysis completed successfully.")
            print("Results saved to: {}".format(cosmic_output_dir))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
