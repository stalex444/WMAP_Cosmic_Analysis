Tutorials
=========

This section provides step-by-step tutorials for common tasks with the WMAP Cosmic Analysis framework.

Tutorial 1: Basic WMAP Data Analysis
-----------------------------------

This tutorial walks through the process of downloading WMAP data and running a basic analysis.

Step 1: Download WMAP Data
~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's download the WMAP power spectrum data:

.. code-block:: python

    from wmap_data.wmap_data_handler import download_wmap_data
    
    # Create a directory for the data
    import os
    os.makedirs('wmap_data', exist_ok=True)
    
    # Download WMAP power spectrum data
    download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')
    
    print("WMAP data downloaded successfully")

Step 2: Load and Preprocess the Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let's load and preprocess the data:

.. code-block:: python

    from wmap_data.wmap_data_handler import load_wmap_data, preprocess_data
    
    # Load the data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')
    
    # Preprocess the data
    preprocessed_data = preprocess_data(
        wmap_data,
        detrend=True,
        normalize=True,
        smooth=True,
        smooth_window=5
    )
    
    print(f"Data loaded and preprocessed, shape: {preprocessed_data.shape}")

Step 3: Run the Golden Ratio Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let's run the Golden Ratio test:

.. code-block:: python

    from analysis.golden_ratio_test import GoldenRatioTest
    import matplotlib.pyplot as plt
    
    # Initialize the test
    gr_test = GoldenRatioTest()
    
    # Run the test
    results = gr_test.run(
        preprocessed_data,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=True,
        phi_bias=0.1
    )
    
    # Print results
    print(f"Phi-optimality: {results['phi_optimality']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Confidence interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    
    # Display the visualization
    if 'figure' in results:
        plt.show()

Step 4: Save and Interpret the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, let's save and interpret the results:

.. code-block:: python

    import json
    import os
    
    # Create a directory for the results
    os.makedirs('results', exist_ok=True)
    
    # Save the results to a JSON file
    result_dict = {
        'phi_optimality': float(results['phi_optimality']),
        'p_value': float(results['p_value']),
        'confidence_interval': [float(results['confidence_interval'][0]), float(results['confidence_interval'][1])]
    }
    
    with open('results/golden_ratio_results.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Interpret the results
    if results['p_value'] < 0.05:
        print("The golden ratio pattern is statistically significant (p < 0.05)")
        if results['confidence_interval'][0] > 0:
            print("The confidence interval excludes zero, further supporting significance")
    else:
        print("The golden ratio pattern is not statistically significant (p >= 0.05)")
    
    print("Results saved to results/golden_ratio_results.json")

Tutorial 2: Comparative Analysis of WMAP and Planck Data
------------------------------------------------------

This tutorial demonstrates how to compare WMAP and Planck data.

Step 1: Download Both Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wmap_data.wmap_data_handler import download_wmap_data
    
    # Create a directory for the data
    import os
    os.makedirs('wmap_data', exist_ok=True)
    
    # Download WMAP and Planck power spectrum data
    download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')
    download_wmap_data(data_type='PLANCK_SPECTRUM', output_dir='wmap_data')
    
    print("Both datasets downloaded successfully")

Step 2: Load and Align the Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wmap_data.wmap_data_handler import load_wmap_data, align_datasets
    
    # Load both datasets
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')
    planck_data = load_wmap_data('wmap_data/planck_tt_spectrum_2018.txt')
    
    # Align the datasets to ensure they cover the same multipole range
    wmap_aligned, planck_aligned = align_datasets(wmap_data, planck_data)
    
    print(f"Datasets aligned, shapes: WMAP {wmap_aligned.shape}, Planck {planck_aligned.shape}")

Step 3: Run Tests on Both Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from analysis.transfer_entropy_test import TransferEntropyTest
    
    # Initialize the test
    te_test = TransferEntropyTest()
    
    # Run the test on WMAP data
    wmap_results = te_test.run(
        wmap_aligned,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=False,
        max_data_points=500,
        num_bins=10
    )
    
    # Run the test on Planck data
    planck_results = te_test.run(
        planck_aligned,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=False,
        max_data_points=500,
        num_bins=10
    )
    
    # Print results
    print("WMAP Results:")
    print(f"Transfer Entropy: {wmap_results['transfer_entropy']:.4f}")
    print(f"p-value: {wmap_results['p_value']:.4f}")
    
    print("\nPlanck Results:")
    print(f"Transfer Entropy: {planck_results['transfer_entropy']:.4f}")
    print(f"p-value: {planck_results['p_value']:.4f}")

Step 4: Create a Comparison Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from visualization.visualization_utils import create_comparison_plot
    import matplotlib.pyplot as plt
    
    # Create a comparison plot
    fig, ax = create_comparison_plot(
        'Transfer Entropy',
        [wmap_results['transfer_entropy'], planck_results['transfer_entropy']],
        ['WMAP', 'Planck'],
        [wmap_results['p_value'], planck_results['p_value']]
    )
    
    # Add a title and adjust layout
    plt.title('Comparison of Transfer Entropy: WMAP vs Planck')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/wmap_planck_comparison.png', dpi=300)
    plt.savefig('results/wmap_planck_comparison.pdf')
    
    # Show the figure
    plt.show()
    
    print("Comparison visualization saved to results/wmap_planck_comparison.png and .pdf")

Tutorial 3: Using the Optimized Scale Transition Test
--------------------------------------------------

This tutorial demonstrates how to use the optimized Scale Transition Test.

Step 1: Load WMAP Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wmap_data.wmap_data_handler import load_wmap_data
    
    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')
    
    print(f"Data loaded, shape: {wmap_data.shape}")

Step 2: Run the Optimized Scale Transition Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from analysis.scale_transition_test import ScaleTransitionTest
    import time
    
    # Initialize the test
    st_test = ScaleTransitionTest()
    
    # Run the test with optimized parameters
    print("Running Scale Transition Test with optimized parameters...")
    start_time = time.time()
    
    results = st_test.run(
        wmap_data,
        timeout_seconds=60,          # Timeout to prevent hanging
        num_simulations=30,          # Reduced from 100 to 30
        early_stopping=True,         # Enable early stopping
        visualize=True,              # Generate visualization
        max_clusters=10,             # Limit number of clusters
        progress_reporting=True      # Show progress
    )
    
    elapsed_time = time.time() - start_time
    print(f"Test completed in {elapsed_time:.2f} seconds")
    
    # Print results
    print(f"Cluster Quality: {results['cluster_quality']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Number of transition points: {len(results['transition_points'])}")
    print(f"Transition points: {results['transition_points']}")

Step 3: Visualize the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # If visualization was generated, display it
    if 'figure' in results:
        plt.figure(results['figure'].number)
        plt.tight_layout()
        plt.savefig('results/scale_transition_results.png', dpi=300)
        plt.show()
    else:
        # Create a custom visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the data
        ax.plot(range(len(wmap_data)), wmap_data, 'b-', alpha=0.7, label='WMAP Data')
        
        # Mark transition points
        for point in results['transition_points']:
            ax.axvline(x=point, color='r', linestyle='--', alpha=0.7)
            ax.text(point, ax.get_ylim()[1] * 0.9, f'{point}', 
                    rotation=90, verticalalignment='top')
        
        ax.set_xlabel('Multipole Moment (ℓ)')
        ax.set_ylabel('Power')
        ax.set_title('Scale Transition Analysis Results')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/scale_transition_results.png', dpi=300)
        plt.show()
    
    print("Scale transition visualization saved to results/scale_transition_results.png")

Tutorial 4: Using the Optimized Transfer Entropy Test
--------------------------------------------------

This tutorial demonstrates how to use the optimized Transfer Entropy Test.

Step 1: Load and Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wmap_data.wmap_data_handler import load_wmap_data, preprocess_data
    import numpy as np
    
    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')
    
    # Preprocess the data
    preprocessed_data = preprocess_data(
        wmap_data,
        detrend=True,
        normalize=True,
        smooth=False
    )
    
    # Split the data into two parts for transfer entropy analysis
    # (e.g., low vs high multipole moments)
    midpoint = len(preprocessed_data) // 2
    data_low = preprocessed_data[:midpoint]
    data_high = preprocessed_data[midpoint:]
    
    print(f"Data prepared, shapes: Low {data_low.shape}, High {data_high.shape}")

Step 2: Run the Optimized Transfer Entropy Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from analysis.transfer_entropy_test import TransferEntropyTest
    import time
    
    # Initialize the test
    te_test = TransferEntropyTest()
    
    # Run the test with optimized parameters
    print("Running Transfer Entropy Test with optimized parameters...")
    start_time = time.time()
    
    # Calculate transfer entropy from low to high multipole moments
    low_to_high_results = te_test.calculate_transfer_entropy(
        data_low, data_high, num_bins=10
    )
    
    # Calculate transfer entropy from high to low multipole moments
    high_to_low_results = te_test.calculate_transfer_entropy(
        data_high, data_low, num_bins=10
    )
    
    # Run the full test with statistical significance
    results = te_test.run(
        preprocessed_data,
        timeout_seconds=60,          # Timeout to prevent hanging
        num_simulations=30,          # Reduced from 100 to 30
        early_stopping=True,         # Enable early stopping
        visualize=True,              # Generate visualization
        max_data_points=500,         # Limit data points to 500
        num_bins=10                  # Number of bins for probability estimation
    )
    
    elapsed_time = time.time() - start_time
    print(f"Test completed in {elapsed_time:.2f} seconds")
    
    # Print results
    print(f"Transfer Entropy (Low to High): {low_to_high_results:.4f}")
    print(f"Transfer Entropy (High to Low): {high_to_low_results:.4f}")
    print(f"Net Information Flow: {low_to_high_results - high_to_low_results:.4f}")
    print(f"Overall Transfer Entropy: {results['transfer_entropy']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")

Step 3: Visualize and Interpret the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create a custom visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot the transfer entropy results
    labels = ['Low to High', 'High to Low']
    values = [low_to_high_results, high_to_low_results]
    
    ax1.bar(labels, values, color=['blue', 'green'])
    ax1.set_ylabel('Transfer Entropy')
    ax1.set_title('Information Flow Between Multipole Scales')
    
    # Add a horizontal line at zero for reference
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add value labels on top of the bars
    for i, v in enumerate(values):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # If visualization was generated by the test, display it in the second subplot
    if 'figure' in results:
        # Extract the axes from the figure
        result_fig = results['figure']
        result_ax = result_fig.axes[0]
        
        # Copy the plot to our new figure
        ax2.plot(result_ax.lines[0].get_xdata(), result_ax.lines[0].get_ydata(),
                color=result_ax.lines[0].get_color(), label=result_ax.lines[0].get_label())
        
        if len(result_ax.lines) > 1:
            ax2.plot(result_ax.lines[1].get_xdata(), result_ax.lines[1].get_ydata(),
                    color=result_ax.lines[1].get_color(), label=result_ax.lines[1].get_label())
        
        ax2.set_xlabel(result_ax.get_xlabel())
        ax2.set_ylabel(result_ax.get_ylabel())
        ax2.set_title(result_ax.get_title())
        ax2.legend()
    else:
        # Create a simple histogram of the data
        ax2.hist(preprocessed_data, bins=30, alpha=0.7, color='blue')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Preprocessed Data')
    
    plt.tight_layout()
    plt.savefig('results/transfer_entropy_results.png', dpi=300)
    plt.show()
    
    print("Transfer entropy visualization saved to results/transfer_entropy_results.png")
    
    # Interpret the results
    print("\nInterpretation:")
    if results['p_value'] < 0.05:
        print("The transfer entropy test shows statistically significant information flow (p < 0.05)")
        
        if low_to_high_results > high_to_low_results:
            print("There is a net information flow from low to high multipole moments")
            print("This suggests that larger cosmic structures may influence smaller structures")
        else:
            print("There is a net information flow from high to low multipole moments")
            print("This suggests that smaller cosmic structures may influence larger structures")
    else:
        print("The transfer entropy test does not show statistically significant information flow (p >= 0.05)")
        print("This suggests that different cosmic scales may be relatively independent")
