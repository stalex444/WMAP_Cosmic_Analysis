#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GR-Specific Coherence Test for WMAP and Planck CMB data.

This script implements the GR-Specific Coherence Test, which tests coherence
specifically in golden ratio related regions of the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
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


def find_golden_ratio_pairs(ell, max_ell=1000, max_pairs=50, use_efficient=False, timeout_seconds=30):
    """
    Find pairs of multipole moments related by the golden ratio.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        max_ell (int): Maximum multipole moment to consider
        max_pairs (int, optional): Maximum number of pairs to return (for memory efficiency)
        use_efficient (bool): Whether to use a memory-efficient algorithm for large datasets
        timeout_seconds (int): Maximum time in seconds to spend searching for pairs
        
    Returns:
        list: List of (ell1, ell2) pairs related by the golden ratio
    """
    print("Finding golden ratio pairs with max_ell =", max_ell)
    print("Input ell array has {} elements".format(len(ell)))
    
    golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
    tolerance = 0.05  # Tolerance for matching
    
    # Filter ell values to be within range
    valid_ell = ell[ell <= max_ell]
    print("After filtering, valid_ell has {} elements".format(len(valid_ell)))
    
    # Set a timeout
    start_time = datetime.now()
    timeout = False
    
    # For very large arrays or when explicitly requested, use a more memory-efficient approach
    if use_efficient or len(valid_ell) > 500:
        print("Using memory-efficient algorithm for large dataset")
        gr_pairs = []
        valid_ell_array = np.array(valid_ell)
        
        # Sample a subset of the data if it's very large
        if len(valid_ell_array) > 500:
            # Take a systematic sample across the range
            sample_size = min(500, len(valid_ell_array))
            indices = np.linspace(0, len(valid_ell_array)-1, sample_size).astype(int)
            valid_ell_array = valid_ell_array[indices]
            print("Sampled down to {} elements".format(len(valid_ell_array)))
        
        # Use a loop-based approach that's more memory efficient
        for i, ell1 in enumerate(valid_ell_array):
            if i % 50 == 0:
                print("Processing element {} of {}".format(i, len(valid_ell_array)))
                
                # Check for timeout
                if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                    print("Timeout reached after {} seconds. Returning pairs found so far.".format(timeout_seconds))
                    timeout = True
                    break
                    
            # Only check a subset of potential pairs to improve performance
            step = max(1, len(valid_ell_array[i+1:]) // 100)
            for j, ell2 in enumerate(valid_ell_array[i+1::step], i+1):
                ratio = ell2 / ell1
                if abs(ratio - golden_ratio) < tolerance:
                    gr_pairs.append((ell1, ell2))
                    if max_pairs and len(gr_pairs) >= max_pairs:
                        print("Reached maximum number of pairs: {}".format(max_pairs))
                        return gr_pairs
    else:
        # Use vectorized operations for better performance with smaller datasets
        gr_pairs = []
        valid_ell_array = np.array(valid_ell)
        
        try:
            # Pre-compute all possible ratios using broadcasting
            print("Creating ratio matrix...")
            ell_matrix = valid_ell_array.reshape(-1, 1)  # Column vector
            
            # Check if the matrix would be too large
            matrix_size = len(valid_ell_array) * len(valid_ell_array) * 8  # Size in bytes (8 bytes per float64)
            if matrix_size > 1e8:  # 100 MB limit
                print("Warning: Ratio matrix would be too large ({}MB). Switching to efficient algorithm.".format(matrix_size/1e6))
                # Recursively call with efficient algorithm
                return find_golden_ratio_pairs(ell, max_ell=max_ell, max_pairs=max_pairs, use_efficient=True, timeout_seconds=timeout_seconds)
                
            ratio_matrix = valid_ell_array / ell_matrix   # Broadcasting creates a matrix of all ratios
            
            # Find indices where the ratio is close to the golden ratio
            # and the second index is greater than the first (to avoid duplicates)
            print("Finding matching ratio indices...")
            row_indices, col_indices = np.where(
                (np.abs(ratio_matrix - golden_ratio) < tolerance) & 
                (np.arange(len(valid_ell_array)).reshape(-1, 1) < np.arange(len(valid_ell_array)))
            )
            
            # Create pairs from the indices
            print("Found {} potential golden ratio pairs".format(len(row_indices)))
            for i, j in zip(row_indices, col_indices):
                gr_pairs.append((valid_ell_array[i], valid_ell_array[j]))
                if max_pairs and len(gr_pairs) >= max_pairs:
                    break
                    
                # Check for timeout
                if i % 1000 == 0 and (datetime.now() - start_time).total_seconds() > timeout_seconds:
                    print("Timeout reached after {} seconds. Returning pairs found so far.".format(timeout_seconds))
                    timeout = True
                    break
                    
            if timeout:
                # Exit the outer loop if timeout occurred in the inner loop
                pass
                
        except MemoryError:
            print("Memory error encountered. Switching to efficient algorithm.")
            # Recursively call with efficient algorithm
            return find_golden_ratio_pairs(ell, max_ell=max_ell, max_pairs=max_pairs, use_efficient=True, timeout_seconds=timeout_seconds)
    
    if timeout and len(gr_pairs) == 0:
        # If we timed out and found no pairs, create at least one pair
        print("Creating at least one pair after timeout")
        # Find the two values closest to golden ratio relationship
        best_pair = None
        best_diff = float('inf')
        
        # Sample a small subset for quick calculation
        sample_size = min(100, len(valid_ell))
        sample_indices = np.linspace(0, len(valid_ell)-1, sample_size).astype(int)
        sample_ell = valid_ell[sample_indices]
        
        for i, ell1 in enumerate(sample_ell):
            for j, ell2 in enumerate(sample_ell[i+1:], i+1):
                ratio = ell2 / ell1
                diff = abs(ratio - golden_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (ell1, ell2)
        
        if best_pair:
            gr_pairs.append(best_pair)
    
    print("Returning {} golden ratio pairs".format(len(gr_pairs)))
    return gr_pairs


def calculate_coherence(power, ell, gr_pairs, max_pairs_to_process=100):
    """
    Calculate coherence specifically in golden ratio related regions.
    
    Args:
        power (numpy.ndarray): Power spectrum values
        ell (numpy.ndarray): Multipole moments
        gr_pairs (list): List of (ell1, ell2) pairs related by the golden ratio
        max_pairs_to_process (int): Maximum number of pairs to process for performance
        
    Returns:
        tuple: (coherence_values, mean_coherence)
    """
    coherence_values = []
    
    # Limit the number of pairs to process
    if len(gr_pairs) > max_pairs_to_process:
        print("Limiting coherence calculation to {} pairs out of {}".format(
            max_pairs_to_process, len(gr_pairs)))
        # Use a systematic sample to ensure good coverage
        indices = np.linspace(0, len(gr_pairs)-1, max_pairs_to_process).astype(int)
        pairs_to_process = [gr_pairs[i] for i in indices]
    else:
        pairs_to_process = gr_pairs
    
    # Pre-compute indices for all pairs at once to avoid repeated searches
    ell_array = np.array(ell)
    
    for ell1, ell2 in pairs_to_process:
        try:
            # Find indices closest to the ell values
            idx1 = np.argmin(np.abs(ell_array - ell1))
            idx2 = np.argmin(np.abs(ell_array - ell2))
            
            # Get power values
            power1 = power[idx1]
            power2 = power[idx2]
            
            # Calculate coherence as correlation between power values in a small window
            window_size = 3
            window1_start = max(0, idx1 - window_size // 2)
            window1_end = min(len(power), idx1 + window_size // 2 + 1)
            window2_start = max(0, idx2 - window_size // 2)
            window2_end = min(len(power), idx2 + window_size // 2 + 1)
            
            window1 = power[window1_start:window1_end]
            window2 = power[window2_start:window2_end]
            
            # Ensure windows are the same length for correlation
            min_length = min(len(window1), len(window2))
            if min_length <= 1:
                # Skip if windows are too small
                continue
                
            window1 = window1[:min_length]
            window2 = window2[:min_length]
            
            # Calculate correlation coefficient
            try:
                corr, _ = stats.pearsonr(window1, window2)
                coherence_values.append(abs(corr))  # Use absolute value for coherence
            except Exception as e:
                print("Error calculating correlation: {}".format(str(e)))
                # Skip this pair
                continue
                
        except Exception as e:
            print("Error processing pair ({}, {}): {}".format(ell1, ell2, str(e)))
            continue
    
    # Calculate mean coherence
    if not coherence_values:
        print("Warning: No valid coherence values calculated")
        return [], 0.0
        
    mean_coherence = np.mean(coherence_values)
    
    return coherence_values, mean_coherence


def run_monte_carlo(power, ell, n_simulations=30, max_ell=1000, use_efficient=False, max_pairs=50):
    """
    Run Monte Carlo simulations to assess the significance of GR-specific coherence.
    
    Args:
        power (numpy.ndarray): Power spectrum values
        ell (numpy.ndarray): Multipole moments
        n_simulations (int): Number of simulations
        max_ell (int): Maximum multipole moment to consider
        use_efficient (bool): Whether to use a memory-efficient algorithm
        max_pairs (int): Maximum number of golden ratio pairs to analyze
        
    Returns:
        tuple: (p_value, phi_optimality, actual_coherence, sim_coherences, gr_pairs, coherence_values)
    """
    start_time = datetime.now()
    
    # Find golden ratio pairs with a shorter timeout
    gr_pairs = find_golden_ratio_pairs(ell, max_ell=max_ell, max_pairs=max_pairs, 
                                       use_efficient=use_efficient, timeout_seconds=min(30, 120//4))
    
    # Calculate actual coherence
    print("Calculating actual coherence...")
    coherence_values, actual_coherence = calculate_coherence(power, ell, gr_pairs)
    
    # Check if we have enough time left for simulations
    elapsed_time = (datetime.now() - start_time).total_seconds()
    remaining_time = 120 - elapsed_time
    
    if remaining_time <= 0:
        print("Warning: Timeout reached before simulations could start.")
        # Return a conservative p-value
        return 0.5, 0.0, actual_coherence, [actual_coherence], gr_pairs, coherence_values
    
    # Adjust number of simulations based on remaining time
    time_per_sim_estimate = 0.5  # Initial estimate: 0.5 seconds per simulation
    adjusted_n_simulations = min(n_simulations, int(remaining_time / time_per_sim_estimate))
    
    if adjusted_n_simulations < n_simulations:
        print("Reducing simulations from {} to {} due to time constraints".format(
            n_simulations, adjusted_n_simulations))
        n_simulations = max(30, adjusted_n_simulations)  # Ensure at least 30 simulations
    
    # Run simulations
    print("Running {} Monte Carlo simulations...".format(n_simulations))
    sim_coherences = []
    significant_count = 0
    required_for_significance = 0.05 * n_simulations  # Number of simulations needed to exceed significance threshold
    
    for i in range(n_simulations):
        # Check for timeout every 5 simulations
        if i % 5 == 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > 120:
                print("Timeout reached after {} seconds and {} simulations.".format(
                    elapsed_time, i))
                break
            
            # Update progress more frequently
            print("  Simulation {}/{} ({:.1f}%)".format(i, n_simulations, 100.0 * i / n_simulations))
            
            # Update time estimate per simulation for future reference
            if i > 0:
                time_per_sim_estimate = elapsed_time / i
                print("  Estimated time per simulation: {:.2f} seconds".format(time_per_sim_estimate))
        
        # Create random permutation of the power spectrum
        sim_power = np.random.permutation(power)
        
        # Calculate coherence for simulated data
        _, sim_coherence = calculate_coherence(sim_power, ell, gr_pairs)
        sim_coherences.append(sim_coherence)
        
        # Check if simulation exceeds actual coherence
        if sim_coherence >= actual_coherence:
            significant_count += 1
            
            # Early stopping if we've already exceeded the significance threshold
            if significant_count > required_for_significance:
                print("  Early stopping at simulation {}: already exceeded significance threshold".format(i+1))
                break
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_coherence else 0 for sim in sim_coherences])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_coherences)
    sim_std = np.std(sim_coherences)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_coherence - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    # Report final statistics
    print("Completed {} simulations in {:.2f} seconds".format(
        len(sim_coherences), (datetime.now() - start_time).total_seconds()))
    print("p-value: {:.4f}, phi-optimality: {:.4f}".format(p_value, phi_optimality))
    
    return p_value, phi_optimality, actual_coherence, sim_coherences, gr_pairs, coherence_values


def plot_gr_specific_coherence_results(ell, power, gr_pairs, coherence_values, 
                                      p_value, phi_optimality, sim_coherences, 
                                      actual_coherence, title, output_path):
    """Plot GR-specific coherence analysis results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot power spectrum with GR-related regions highlighted
        ax1.plot(ell, power, 'b-', alpha=0.7, label='Power Spectrum')
        
        # Highlight GR-related regions
        for i, (ell1, ell2) in enumerate(gr_pairs):
            # Find indices closest to the ell values
            idx1 = np.argmin(np.abs(ell - ell1))
            idx2 = np.argmin(np.abs(ell - ell2))
            
            # Highlight regions
            ax1.axvline(ell[idx1], color='r', linestyle='--', alpha=0.5)
            ax1.axvline(ell[idx2], color='r', linestyle='--', alpha=0.5)
            
            # Add connecting line
            ax1.plot([ell[idx1], ell[idx2]], [power[idx1], power[idx2]], 'g-', alpha=0.3)
            
            # Add text with coherence value if available
            if i < len(coherence_values):
                midpoint_x = (ell[idx1] + ell[idx2]) / 2
                midpoint_y = (power[idx1] + power[idx2]) / 2
                ax1.text(midpoint_x, midpoint_y, '{:.2f}'.format(coherence_values[i]), 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Multipole Moment (ℓ)')
        ax1.set_ylabel('Power')
        ax1.set_title('Power Spectrum with Golden Ratio Related Regions')
        ax1.grid(True, alpha=0.3)
        
        # Add text with number of GR pairs
        ax1.text(0.05, 0.95, 'Number of GR Pairs: {}'.format(len(gr_pairs)), 
                transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        ax2.hist(sim_coherences, bins=min(20, len(sim_coherences)//5 + 1), 
                alpha=0.7, color='gray', label='Random Simulations')
        ax2.axvline(actual_coherence, color='r', linestyle='--', linewidth=2, 
                   label='Actual Coherence: {:.4f}'.format(actual_coherence))
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Mean Coherence')
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
        print("Warning: Error in plotting GR-specific coherence results: {}".format(str(e)))
        print("Continuing with analysis...")


def run_gr_specific_coherence_test(ell, power, output_dir, name, n_simulations=30, max_ell=1000, use_efficient=False, max_pairs=50):
    """Run GR-specific coherence test on the provided data."""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Monte Carlo simulations
    p_value, phi_optimality, actual_coherence, sim_coherences, gr_pairs, coherence_values = run_monte_carlo(
        power, ell, n_simulations=n_simulations, max_ell=max_ell, use_efficient=use_efficient, max_pairs=max_pairs
    )
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_gr_specific_coherence.png'.format(name.lower()))
    plot_gr_specific_coherence_results(
        ell, power, gr_pairs, coherence_values, p_value, phi_optimality, 
        sim_coherences, actual_coherence, 'GR-Specific Coherence Test: {} CMB Data'.format(name), 
        plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_gr_specific_coherence.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('GR-Specific Coherence Test Results: {} CMB Data\n'.format(name))
        f.write('=' * 50 + '\n\n')
        f.write('Mean Coherence: {:.6f}\n'.format(actual_coherence))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Significant: {}\n'.format(p_value < 0.05))
        f.write('Number of Golden Ratio Pairs: {}\n\n'.format(len(gr_pairs)))
        
        f.write('Golden Ratio Pairs and Coherence Values:\n')
        for i, ((ell1, ell2), coherence) in enumerate(zip(gr_pairs, coherence_values)):
            f.write('  Pair {}: ℓ1 = {:.1f}, ℓ2 = {:.1f}, Coherence = {:.6f}\n'.format(
                i+1, ell1, ell2, coherence))
        
        f.write('\nInterpretation:\n')
        if p_value < 0.05 and actual_coherence > 0.5:
            f.write('  Strong GR-specific coherence: The CMB power spectrum shows significant coherence\n')
            f.write('  in regions related by the golden ratio, suggesting a fundamental organizational principle.\n')
        elif p_value < 0.05:
            f.write('  Significant GR-specific coherence: The CMB power spectrum shows significant coherence\n')
            f.write('  in regions related by the golden ratio, suggesting non-random organization.\n')
        elif actual_coherence > 0.5:
            f.write('  Moderate GR-specific coherence: While not statistically significant, the CMB power spectrum\n')
            f.write('  shows moderate coherence in regions related by the golden ratio.\n')
        else:
            f.write('  Weak GR-specific coherence: The CMB power spectrum does not show significant coherence\n')
            f.write('  in regions related by the golden ratio beyond what would be expected by chance.\n')
        
        f.write('\nAnalysis performed on: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('Number of simulations: {}\n'.format(n_simulations))
    
    print('{} GR-Specific Coherence Test Results:'.format(name))
    print('  Mean Coherence: {:.6f}'.format(actual_coherence))
    print('  P-value: {:.6f}'.format(p_value))
    print('  Phi-Optimality: {:.6f}'.format(phi_optimality))
    print('  Significant: {}'.format(p_value < 0.05))
    print('  Number of Golden Ratio Pairs: {}'.format(len(gr_pairs)))
    
    return {
        'mean_coherence': actual_coherence,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05,
        'gr_pairs': gr_pairs,
        'coherence_values': coherence_values
    }


def compare_results(wmap_results, planck_results, output_dir):
    """Compare GR-specific coherence test results between WMAP and Planck data."""
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        coherence_diff = abs(wmap_results['mean_coherence'] - planck_results['mean_coherence'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'gr_specific_coherence_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('GR-Specific Coherence Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Mean Coherence: {:.6f}\n'.format(wmap_results['mean_coherence']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n'.format(wmap_results['significant']))
            f.write('WMAP Number of GR Pairs: {}\n\n'.format(len(wmap_results['gr_pairs'])))
            
            f.write('Planck Mean Coherence: {:.6f}\n'.format(planck_results['mean_coherence']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n'.format(planck_results['significant']))
            f.write('Planck Number of GR Pairs: {}\n\n'.format(len(planck_results['gr_pairs'])))
            
            f.write('Difference in Mean Coherence: {:.6f}\n'.format(coherence_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of mean coherence and phi-optimality
            metrics = ['Mean Coherence', 'Phi-Optimality']
            wmap_values = [wmap_results['mean_coherence'], wmap_results['phi_optimality']]
            planck_values = [planck_results['mean_coherence'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('GR-Specific Coherence: WMAP vs Planck')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # Add text with p-values
            ax1.text(0 - width/2, wmap_values[0] + 0.02, 
                    'p={:.4f}'.format(wmap_results["p_value"]), 
                    ha='center', va='bottom', color='blue', fontweight='bold')
            ax1.text(0 + width/2, planck_values[0] + 0.02, 
                    'p={:.4f}'.format(planck_results["p_value"]), 
                    ha='center', va='bottom', color='red', fontweight='bold')
            
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Histogram of coherence values
            wmap_coherence = wmap_results['coherence_values']
            planck_coherence = planck_results['coherence_values']
            
            bins = np.linspace(0, 1, 20)
            
            if len(wmap_coherence) > 0:
                ax2.hist(wmap_coherence, bins=bins, alpha=0.5, color='blue', label='WMAP')
            
            if len(planck_coherence) > 0:
                ax2.hist(planck_coherence, bins=bins, alpha=0.5, color='red', label='Planck')
            
            ax2.set_xlabel('Coherence Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Coherence Values')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'gr_specific_coherence_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Mean Coherence: {:.6f}".format(coherence_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print("Continuing with analysis...")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run GR-Specific Coherence Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=30, 
                        help='Number of simulations for Monte Carlo. Default: 30')
    parser.add_argument('--max-ell', type=int, default=1000,
                        help='Maximum multipole moment to consider. Default: 1000')
    parser.add_argument('--max-pairs', type=int, default=50,
                        help='Maximum number of golden ratio pairs to analyze. Default: 50')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/gr_specific_coherence_TIMESTAMP')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # Set debug flag
    debug = args.debug
    
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
        output_dir = os.path.join('results', "gr_specific_coherence_{}".format(timestamp))
    
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
        
        # Run GR-specific coherence test on WMAP data
        print("Running GR-specific coherence test on WMAP data...")
        wmap_results = run_gr_specific_coherence_test(
            wmap_ell, 
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            max_ell=min(args.max_ell, np.max(wmap_ell)),
            use_efficient=False,  # Use vectorized approach for small WMAP dataset
            max_pairs=args.max_pairs
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
        
        # For Planck data, use a smaller max_ell to prevent hanging
        planck_max_ell = min(args.max_ell, 500)  # Limit to 500 for Planck data
        print("Using max_ell = {} for Planck data analysis".format(planck_max_ell))
        
        # Preprocess Planck data
        print("Preprocessing Planck data...")
        planck_processed = preprocess_data(
            planck_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run GR-specific coherence test on Planck data
        print("Running GR-specific coherence test on Planck data...")
        planck_results = run_gr_specific_coherence_test(
            planck_ell,
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck',
            n_simulations=args.n_simulations,
            max_ell=planck_max_ell,
            use_efficient=True,  # Use memory-efficient approach for large Planck dataset
            max_pairs=args.max_pairs
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck GR-specific coherence test results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nGR-specific coherence test complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
