#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resonance Analysis Test for WMAP and Planck CMB data.

This script implements the Resonance Analysis Test, which examines resonance patterns
in the CMB power spectrum that may indicate harmonic organization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from datetime import datetime
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None


def load_planck_power_spectrum(file_path):
    """Load Planck CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        return ell, power, error
    except Exception as e:
        print("Error loading Planck power spectrum: {}".format(str(e)))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """Preprocess data for analysis."""
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


def find_resonance_peaks(ell, power, prominence=0.5, width=3):
    """
    Find resonance peaks in the power spectrum.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        prominence (float): Required prominence of peaks
        width (int): Required width of peaks
        
    Returns:
        tuple: (peak_indices, peak_ells, peak_powers)
    """
    # Find peaks in the power spectrum
    peak_indices, peak_properties = signal.find_peaks(
        power, prominence=prominence, width=width
    )
    
    # Get the multipole moments and power values at the peaks
    peak_ells = ell[peak_indices]
    peak_powers = power[peak_indices]
    
    return peak_indices, peak_ells, peak_powers


def calculate_resonance_ratios(peak_ells):
    """
    Calculate ratios between adjacent resonance peaks.
    
    Args:
        peak_ells (numpy.ndarray): Multipole moments at peaks
        
    Returns:
        numpy.ndarray: Ratios between adjacent peaks
    """
    if len(peak_ells) < 2:
        return np.array([])
    
    # Calculate ratios between adjacent peaks
    ratios = peak_ells[1:] / peak_ells[:-1]
    
    return ratios


def analyze_resonance_patterns(ratios):
    """
    Analyze patterns in resonance ratios.
    
    Args:
        ratios (numpy.ndarray): Ratios between adjacent peaks
        
    Returns:
        tuple: (mean_ratio, std_ratio, is_harmonic)
    """
    if len(ratios) == 0:
        return 0, 0, False
    
    # Calculate mean and standard deviation of ratios
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    # Check if ratios are close to integer values (harmonic resonance)
    # or close to the golden ratio (1.618...)
    golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
    
    # Calculate distances to integer ratios and golden ratio
    int_distances = np.min([np.abs(ratios - np.round(ratios)), 
                           np.abs(ratios - golden_ratio)], axis=0)
    
    # If the average distance is small, consider it harmonic
    is_harmonic = np.mean(int_distances) < 0.1
    
    return mean_ratio, std_ratio, is_harmonic


def calculate_resonance_score(ell, power, peak_ells, peak_powers):
    """
    Calculate a resonance score based on peak spacing and amplitude patterns.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        peak_ells (numpy.ndarray): Multipole moments at peaks
        peak_powers (numpy.ndarray): Power values at peaks
        
    Returns:
        float: Resonance score between 0 and 1
    """
    if len(peak_ells) < 3:
        return 0.0
    
    # Calculate spacing between peaks
    peak_spacing = np.diff(peak_ells)
    
    # Check for regularity in spacing (lower variance = more regular)
    spacing_variance = np.var(peak_spacing) / np.mean(peak_spacing)**2
    spacing_score = np.exp(-spacing_variance)  # Higher for more regular spacing
    
    # Check for patterns in peak amplitudes
    # Calculate autocorrelation of peak powers
    if len(peak_powers) > 1:
        autocorr = np.correlate(peak_powers, peak_powers, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Higher autocorrelation at lag 1 indicates stronger patterns
        amplitude_score = autocorr[1] if len(autocorr) > 1 else 0
    else:
        amplitude_score = 0
    
    # Combine scores (with spacing given more weight)
    resonance_score = 0.7 * spacing_score + 0.3 * max(0, amplitude_score)
    
    # Ensure score is between 0 and 1
    resonance_score = max(0, min(1, resonance_score))
    
    return resonance_score


def run_monte_carlo(ell, power, n_simulations=100, prominence=0.5, width=3):
    """
    Run Monte Carlo simulations to assess the significance of resonance patterns.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of simulations
        prominence (float): Required prominence of peaks
        width (int): Required width of peaks
        
    Returns:
        tuple: (p_value, phi_optimality, actual_score, sim_scores)
    """
    # Calculate resonance score for actual data
    peak_indices, peak_ells, peak_powers = find_resonance_peaks(
        ell, power, prominence=prominence, width=width
    )
    actual_score = calculate_resonance_score(ell, power, peak_ells, peak_powers)
    
    # Run simulations
    sim_scores = []
    for i in range(n_simulations):
        if i % 10 == 0:
            print("  Simulation {}/{}".format(i, n_simulations))
        
        # Create random permutation of the power spectrum
        sim_power = np.random.permutation(power)
        
        # Find peaks and calculate resonance score
        sim_peak_indices, sim_peak_ells, sim_peak_powers = find_resonance_peaks(
            ell, sim_power, prominence=prominence, width=width
        )
        sim_score = calculate_resonance_score(ell, sim_power, sim_peak_ells, sim_peak_powers)
        
        sim_scores.append(sim_score)
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_score else 0 for sim in sim_scores])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_scores)
    sim_std = np.std(sim_scores)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_score - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    return p_value, phi_optimality, actual_score, sim_scores, peak_ells, peak_powers


def plot_resonance_results(ell, power, peak_ells, peak_powers, p_value, phi_optimality, 
                          sim_scores, actual_score, title, output_path):
    """Plot resonance analysis results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot power spectrum with peaks
        ax1.plot(ell, power, 'b-', alpha=0.7, label='Power Spectrum')
        ax1.scatter(peak_ells, peak_powers, color='r', s=50, 
                   label='Resonance Peaks (n={})'.format(len(peak_ells)))
        
        # Add vertical lines at peaks
        for peak_ell in peak_ells:
            ax1.axvline(peak_ell, color='r', linestyle='--', alpha=0.3)
        
        ax1.set_xlabel('Multipole Moment (ℓ)')
        ax1.set_ylabel('Power')
        ax1.set_title('Power Spectrum with Resonance Peaks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add resonance score
        ax1.text(0.05, 0.95, 'Resonance Score = {:.4f}'.format(actual_score), 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        ax2.hist(sim_scores, bins=min(30, len(sim_scores)//3), 
                alpha=0.7, color='gray', label='Random Simulations')
        ax2.axvline(actual_score, color='r', linestyle='--', linewidth=2, 
                   label='Actual Score: {:.4f}'.format(actual_score))
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Resonance Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Add text with results
        plt.figtext(0.5, 0.01, 'P-value: {:.4f} | Phi-Optimality: {:.4f}'.format(p_value, phi_optimality), 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print("Warning: Error in plotting resonance results: {}".format(str(e)))
        print("Continuing with analysis...")


def run_resonance_test(ell, power, output_dir, name, n_simulations=100, prominence=0.5, width=3):
    """Run resonance analysis test on the provided data."""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Monte Carlo simulations
    p_value, phi_optimality, actual_score, sim_scores, peak_ells, peak_powers = run_monte_carlo(
        ell, power, n_simulations=n_simulations, prominence=prominence, width=width
    )
    
    # Calculate resonance ratios
    ratios = calculate_resonance_ratios(peak_ells)
    mean_ratio, std_ratio, is_harmonic = analyze_resonance_patterns(ratios)
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_resonance.png'.format(name.lower()))
    plot_resonance_results(
        ell, power, peak_ells, peak_powers, p_value, phi_optimality, sim_scores, actual_score,
        'Resonance Analysis Test: {} CMB Data'.format(name), plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_resonance.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('Resonance Analysis Test Results: {} CMB Data\n'.format(name))
        f.write('=' * 50 + '\n\n')
        f.write('Resonance Score: {:.6f}\n'.format(actual_score))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Significant: {}\n\n'.format(p_value < 0.05))
        
        f.write('Number of Resonance Peaks: {}\n'.format(len(peak_ells)))
        f.write('Mean Ratio Between Peaks: {:.6f}\n'.format(mean_ratio))
        f.write('Standard Deviation of Ratios: {:.6f}\n'.format(std_ratio))
        f.write('Harmonic Pattern Detected: {}\n\n'.format(is_harmonic))
        
        f.write('Peak Multipoles and Powers:\n')
        for i, (ell_val, power_val) in enumerate(zip(peak_ells, peak_powers)):
            f.write('  Peak {}: ℓ = {:.1f}, Power = {:.6f}\n'.format(i+1, ell_val, power_val))
        
        if len(ratios) > 0:
            f.write('\nRatios Between Adjacent Peaks:\n')
            for i, ratio in enumerate(ratios):
                f.write('  Ratio {}/{}: {:.6f}\n'.format(i+2, i+1, ratio))
        
        f.write('\nInterpretation:\n')
        if actual_score > 0.7 and p_value < 0.05:
            f.write('  Strong resonance pattern: The CMB power spectrum shows significant harmonic organization.\n')
            f.write('  This suggests a fundamental wave-like organization in the cosmic microwave background.\n')
        elif actual_score > 0.5 and p_value < 0.1:
            f.write('  Moderate resonance pattern: The CMB power spectrum shows some harmonic organization.\n')
            f.write('  This suggests partial wave-like organization in the cosmic microwave background.\n')
        else:
            f.write('  Weak resonance pattern: The CMB power spectrum shows limited harmonic organization.\n')
            f.write('  This suggests minimal wave-like organization in the cosmic microwave background.\n')
        
        f.write('\nAnalysis performed on: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('Number of simulations: {}\n'.format(n_simulations))
    
    print('{} Resonance Analysis Test Results:'.format(name))
    print('  Resonance Score: {:.6f}'.format(actual_score))
    print('  P-value: {:.6f}'.format(p_value))
    print('  Phi-Optimality: {:.6f}'.format(phi_optimality))
    print('  Significant: {}'.format(p_value < 0.05))
    print('  Number of Peaks: {}'.format(len(peak_ells)))
    print('  Mean Ratio: {:.6f}'.format(mean_ratio))
    print('  Harmonic Pattern: {}'.format(is_harmonic))
    
    return {
        'resonance_score': actual_score,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05,
        'peak_ells': peak_ells,
        'peak_powers': peak_powers,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'is_harmonic': is_harmonic
    }


def compare_results(wmap_results, planck_results, output_dir):
    """Compare resonance analysis test results between WMAP and Planck data."""
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        score_diff = abs(wmap_results['resonance_score'] - planck_results['resonance_score'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'resonance_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Resonance Analysis Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Resonance Score: {:.6f}\n'.format(wmap_results['resonance_score']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n'.format(wmap_results['significant']))
            f.write('WMAP Number of Peaks: {}\n'.format(len(wmap_results['peak_ells'])))
            f.write('WMAP Mean Ratio: {:.6f}\n'.format(wmap_results['mean_ratio']))
            f.write('WMAP Harmonic Pattern: {}\n\n'.format(wmap_results['is_harmonic']))
            
            f.write('Planck Resonance Score: {:.6f}\n'.format(planck_results['resonance_score']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n'.format(planck_results['significant']))
            f.write('Planck Number of Peaks: {}\n'.format(len(planck_results['peak_ells'])))
            f.write('Planck Mean Ratio: {:.6f}\n'.format(planck_results['mean_ratio']))
            f.write('Planck Harmonic Pattern: {}\n\n'.format(planck_results['is_harmonic']))
            
            f.write('Difference in Resonance Score: {:.6f}\n'.format(score_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of resonance scores and phi-optimality
            metrics = ['Resonance Score', 'Phi-Optimality']
            wmap_values = [wmap_results['resonance_score'], wmap_results['phi_optimality']]
            planck_values = [planck_results['resonance_score'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Resonance Analysis: WMAP vs Planck')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # Add text with p-values
            for i, metric in enumerate(metrics):
                ax1.text(i - width/2, wmap_values[i] + 0.02, 
                        'p={:.4f}'.format(wmap_results["p_value"]), 
                        ha='center', va='bottom', color='blue', fontweight='bold')
                ax1.text(i + width/2, planck_values[i] + 0.02, 
                        'p={:.4f}'.format(planck_results["p_value"]), 
                        ha='center', va='bottom', color='red', fontweight='bold')
            
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Comparison of peak distributions
            wmap_peak_ells = wmap_results['peak_ells']
            planck_peak_ells = planck_results['peak_ells']
            
            # Create histograms of peak distributions
            bins = np.linspace(0, max(np.max(wmap_peak_ells) if len(wmap_peak_ells) > 0 else 0, 
                                     np.max(planck_peak_ells) if len(planck_peak_ells) > 0 else 0) + 50, 20)
            
            if len(wmap_peak_ells) > 0:
                ax2.hist(wmap_peak_ells, bins=bins, alpha=0.5, color='blue', label='WMAP Peaks')
            
            if len(planck_peak_ells) > 0:
                ax2.hist(planck_peak_ells, bins=bins, alpha=0.5, color='red', label='Planck Peaks')
            
            ax2.set_xlabel('Multipole Moment (ℓ)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Resonance Peaks')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'resonance_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Resonance Score: {:.6f}".format(score_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print("Continuing with analysis...")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Resonance Analysis Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=100, 
                        help='Number of simulations for Monte Carlo. Default: 100')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Required prominence of peaks. Default: 0.5')
    parser.add_argument('--width', type=int, default=3,
                        help='Required width of peaks. Default: 3')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/resonance_TIMESTAMP')
    
    args = parser.parse_args()
    
    # Path to data files
    wmap_file = 'wmap_data/raw_data/wmap_binned_tt_spectrum_9yr_v5.txt'
    planck_file = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis/planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: {}".format(wmap_file))
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: {}".format(planck_file))
        print("Please make sure the Planck data is available in the Cosmic_Consciousness_Analysis repository.")
        return 1
    
    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', "resonance_{}".format(timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results dictionaries
    wmap_results = None
    planck_results = None
    
    # Process WMAP data if requested
    if not args.planck_only:
        # Load WMAP data
        print("Loading WMAP power spectrum...")
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        
        if wmap_ell is None:
            print("Error loading WMAP power spectrum.")
            return 1
        
        print("Loaded WMAP power spectrum with {} multipoles".format(len(wmap_ell)))
        
        # Preprocess WMAP data
        print("Preprocessing WMAP data...")
        wmap_processed = preprocess_data(
            wmap_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run resonance analysis test on WMAP data
        print("Running resonance analysis test on WMAP data...")
        wmap_results = run_resonance_test(
            wmap_ell, 
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            prominence=args.prominence,
            width=args.width
        )
    
    # Process Planck data if requested
    if not args.wmap_only:
        # Load Planck data
        print("Loading Planck power spectrum...")
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        
        if planck_ell is None:
            print("Error loading Planck power spectrum.")
            return 1
        
        print("Loaded Planck power spectrum with {} multipoles".format(len(planck_ell)))
        
        # Preprocess Planck data
        print("Preprocessing Planck data...")
        planck_processed = preprocess_data(
            planck_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run resonance analysis test on Planck data
        print("Running resonance analysis test on Planck data...")
        planck_results = run_resonance_test(
            planck_ell,
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck',
            n_simulations=args.n_simulations,
            prominence=args.prominence,
            width=args.width
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck resonance analysis test results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nResonance analysis test complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
