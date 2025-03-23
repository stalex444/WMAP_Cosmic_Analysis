Examples
========

This section provides practical examples of using the WMAP Cosmic Analysis framework for various analysis tasks.

Basic Analysis Example
--------------------

This example demonstrates how to load WMAP data and run a basic golden ratio analysis:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data, preprocess_data
    from analysis.golden_ratio_test import GoldenRatioTest

    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

    # Preprocess the data
    preprocessed_data = preprocess_data(
        wmap_data,
        detrend=True,
        normalize=True,
        smooth=True,
        smooth_window=5
    )

    # Initialize and run the golden ratio test
    gr_test = GoldenRatioTest()
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
    
    # Check if the result is statistically significant
    if results['p_value'] < 0.05:
        print("The result is statistically significant (p < 0.05)")
    else:
        print("The result is not statistically significant (p >= 0.05)")

    # Show the visualization if it was generated
    if 'figure' in results:
        plt.show()

Comparative Analysis Example
--------------------------

This example shows how to compare WMAP and Planck data using the framework:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data
    from analysis.transfer_entropy_test import TransferEntropyTest
    from visualization.visualization_utils import create_comparison_plot

    # Load WMAP and Planck data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')
    planck_data = load_wmap_data('wmap_data/planck_tt_spectrum_2018.txt')

    # Initialize the transfer entropy test
    te_test = TransferEntropyTest()

    # Run the test on WMAP data
    wmap_results = te_test.run(
        wmap_data,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=False,
        max_data_points=500,
        num_bins=10
    )

    # Run the test on Planck data
    planck_results = te_test.run(
        planck_data,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=False,
        max_data_points=500,
        num_bins=10
    )

    # Print comparison
    print("WMAP Transfer Entropy:", wmap_results['transfer_entropy'])
    print("WMAP p-value:", wmap_results['p_value'])
    print("Planck Transfer Entropy:", planck_results['transfer_entropy'])
    print("Planck p-value:", planck_results['p_value'])

    # Create comparison plot
    fig, ax = create_comparison_plot(
        'Transfer Entropy',
        [wmap_results['transfer_entropy'], planck_results['transfer_entropy']],
        ['WMAP', 'Planck'],
        [wmap_results['p_value'], planck_results['p_value']]
    )
    plt.tight_layout()
    plt.show()

Batch Processing Example
----------------------

This example demonstrates how to run multiple tests with different parameters and aggregate the results:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data
    from analysis.golden_ratio_test import GoldenRatioTest
    from analysis.transfer_entropy_test import TransferEntropyTest
    from analysis.scale_transition_test import ScaleTransitionTest

    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

    # Define parameter grid
    phi_bias_values = [0.0, 0.1, 0.2, 0.3]
    num_simulations = 30
    timeout_seconds = 60

    # Initialize tests
    gr_test = GoldenRatioTest()
    te_test = TransferEntropyTest()
    st_test = ScaleTransitionTest()

    # Initialize results storage
    results_data = []

    # Run parameter sweep
    for phi_bias in phi_bias_values:
        print(f"Running tests with phi_bias = {phi_bias}")
        
        # Run golden ratio test
        gr_results = gr_test.run(
            wmap_data,
            timeout_seconds=timeout_seconds,
            num_simulations=num_simulations,
            early_stopping=True,
            visualize=False,
            phi_bias=phi_bias
        )
        
        # Run transfer entropy test
        te_results = te_test.run(
            wmap_data,
            timeout_seconds=timeout_seconds,
            num_simulations=num_simulations,
            early_stopping=True,
            visualize=False,
            max_data_points=500,
            num_bins=10
        )
        
        # Run scale transition test
        st_results = st_test.run(
            wmap_data,
            timeout_seconds=timeout_seconds,
            num_simulations=num_simulations,
            early_stopping=True,
            visualize=False,
            max_clusters=10
        )
        
        # Store results
        results_data.append({
            'phi_bias': phi_bias,
            'gr_phi_optimality': gr_results['phi_optimality'],
            'gr_p_value': gr_results['p_value'],
            'te_value': te_results['transfer_entropy'],
            'te_p_value': te_results['p_value'],
            'st_quality': st_results['cluster_quality'],
            'st_p_value': st_results['p_value']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Print results table
    print("\nResults Summary:")
    print(results_df)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot phi-optimality vs phi_bias
    axes[0].plot(results_df['phi_bias'], results_df['gr_phi_optimality'], 'o-')
    axes[0].set_xlabel('Phi Bias')
    axes[0].set_ylabel('Phi-Optimality')
    axes[0].set_title('Golden Ratio Test: Phi-Optimality vs Phi Bias')
    axes[0].grid(True)
    
    # Plot transfer entropy vs phi_bias
    axes[1].plot(results_df['phi_bias'], results_df['te_value'], 'o-')
    axes[1].set_xlabel('Phi Bias')
    axes[1].set_ylabel('Transfer Entropy')
    axes[1].set_title('Transfer Entropy Test: TE Value vs Phi Bias')
    axes[1].grid(True)
    
    # Plot scale transition quality vs phi_bias
    axes[2].plot(results_df['phi_bias'], results_df['st_quality'], 'o-')
    axes[2].set_xlabel('Phi Bias')
    axes[2].set_ylabel('Cluster Quality')
    axes[2].set_title('Scale Transition Test: Cluster Quality vs Phi Bias')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save results to CSV
    results_df.to_csv('parameter_sweep_results.csv', index=False)
    print("Results saved to parameter_sweep_results.csv")

Optimized Scale Transition Test Example
-------------------------------------

This example demonstrates how to use the optimized Scale Transition Test with timeout and early stopping:

.. code-block:: python

    import numpy as np
    import time
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data
    from analysis.scale_transition_test import ScaleTransitionTest

    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

    # Initialize the scale transition test
    st_test = ScaleTransitionTest()

    # Run the test with optimized parameters
    print("Running Scale Transition Test with optimized parameters...")
    start_time = time.time()
    
    results = st_test.run(
        wmap_data,
        timeout_seconds=90,          # Increased timeout for demonstration
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
    
    # Check if the result is statistically significant
    if results['p_value'] < 0.05:
        print("The result is statistically significant (p < 0.05)")
    else:
        print("The result is not statistically significant (p >= 0.05)")
    
    # Show the visualization if it was generated
    if 'figure' in results:
        plt.show()

Optimized Transfer Entropy Test Example
-------------------------------------

This example demonstrates how to use the optimized Transfer Entropy Test with limited data points and efficient calculation:

.. code-block:: python

    import numpy as np
    import time
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data
    from analysis.transfer_entropy_test import TransferEntropyTest

    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

    # Initialize the transfer entropy test
    te_test = TransferEntropyTest()

    # Run the test with optimized parameters
    print("Running Transfer Entropy Test with optimized parameters...")
    start_time = time.time()
    
    results = te_test.run(
        wmap_data,
        timeout_seconds=60,          # Timeout to prevent hanging
        num_simulations=30,          # Reduced from 100 to 30
        early_stopping=True,         # Enable early stopping
        visualize=True,              # Generate visualization
        max_data_points=500,         # Limit data points to 500
        num_bins=10,                 # Number of bins for probability estimation
        progress_reporting=True      # Show progress
    )
    
    elapsed_time = time.time() - start_time
    print(f"Test completed in {elapsed_time:.2f} seconds")
    
    # Print results
    print(f"Transfer Entropy: {results['transfer_entropy']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Reference value: {results['reference_value']:.4f}")
    
    # Check if the result is statistically significant
    if results['p_value'] < 0.05:
        print("The result is statistically significant (p < 0.05)")
    else:
        print("The result is not statistically significant (p >= 0.05)")
    
    # Show the visualization if it was generated
    if 'figure' in results:
        plt.show()

Python 2.7 Compatibility Example
------------------------------

This example demonstrates how to ensure compatibility with Python 2.7:

.. code-block:: python

    # Ensure compatibility with both Python 2.7 and 3.x
    from __future__ import division, print_function, absolute_import

    import numpy as np
    import matplotlib
    # Set a compatible backend for older matplotlib versions
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from wmap_data.wmap_data_handler import load_wmap_data
    from analysis.golden_ratio_test import GoldenRatioTest

    # Load WMAP data
    wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

    # Initialize the golden ratio test
    gr_test = GoldenRatioTest()

    # Run the test with Python 2.7 compatible parameters
    print("Running Golden Ratio Test with Python 2.7 compatibility...")
    
    # Use string formatting compatible with Python 2.7
    print("Using Python 2.7 compatible string formatting: %s" % "example")
    
    # Ensure proper division (Python 2.7 would use integer division by default)
    ratio = 5 / 2
    print("5 / 2 = {0}".format(ratio))  # Should be 2.5, not 2
    
    results = gr_test.run(
        wmap_data,
        timeout_seconds=60,
        num_simulations=30,
        early_stopping=True,
        visualize=True,
        phi_bias=0.1
    )
    
    # Print results using Python 2.7 compatible formatting
    print("Phi-optimality: {0:.4f}".format(results['phi_optimality']))
    print("p-value: {0:.4f}".format(results['p_value']))
    print("Confidence interval: [{0:.4f}, {1:.4f}]".format(
        results['confidence_interval'][0],
        results['confidence_interval'][1]
    ))
    
    # Save figure instead of showing it (more compatible with headless environments)
    if 'figure' in results:
        results['figure'].savefig('golden_ratio_results.png', dpi=300)
        print("Figure saved to golden_ratio_results.png")

Configuration-Based Analysis Example
----------------------------------

This example demonstrates how to use the configuration system for analysis:

.. code-block:: python

    import numpy as np
    import yaml
    import matplotlib.pyplot as plt
    from wmap_data.wmap_data_handler import load_wmap_data
    from config.config_loader import load_config
    from analysis.run_analysis import run_test_by_name

    # Define a custom configuration
    custom_config = {
        'data': {
            'wmap_data_path': 'wmap_data/wmap_tt_spectrum_9yr_v5.txt',
            'preprocessing': {
                'detrend': True,
                'normalize': True,
                'smooth': True,
                'smooth_window': 5
            }
        },
        'analysis': {
            'random_seed': 42,
            'num_simulations': 30,
            'timeout_seconds': 60,
            'early_stopping': True,
            'golden_ratio': {
                'phi_value': 1.618033988749895,
                'tolerance': 0.02,
                'phi_bias': 0.1
            },
            'transfer_entropy': {
                'max_data_points': 500,
                'num_bins': 10
            },
            'scale_transition': {
                'max_clusters': 10
            }
        },
        'visualization': {
            'enabled': True,
            'dpi': 300,
            'formats': ['png', 'pdf']
        },
        'output': {
            'results_dir': 'results',
            'save_results': True,
            'result_format': 'json'
        }
    }

    # Save the configuration to a YAML file
    with open('custom_config.yaml', 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)

    # Load the configuration
    config = load_config('custom_config.yaml')

    # Load data based on configuration
    wmap_data = load_wmap_data(config['data']['wmap_data_path'])

    # Run tests based on configuration
    test_names = ['golden_ratio', 'transfer_entropy', 'scale_transition']
    all_results = {}

    for test_name in test_names:
        print(f"Running {test_name} test...")
        results = run_test_by_name(
            test_name,
            wmap_data,
            config['analysis']['timeout_seconds'],
            config['analysis']['num_simulations'],
            config['analysis']['early_stopping'],
            config['visualization']['enabled'],
            **config['analysis'].get(test_name, {})
        )
        all_results[test_name] = results
        print(f"{test_name} test completed with p-value: {results['p_value']:.4f}")

    # Save results if configured
    if config['output']['save_results']:
        import json
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(config['output']['results_dir'], exist_ok=True)
        
        # Save results to JSON file
        output_file = os.path.join(config['output']['results_dir'], 'analysis_results.json')
        with open(output_file, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {}
            for test_name, results in all_results.items():
                serializable_results[test_name] = {
                    k: v.item() if hasattr(v, 'item') else v
                    for k, v in results.items()
                    if k != 'figure'  # Skip matplotlib figures
                }
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")

    # Show visualizations if enabled
    if config['visualization']['enabled']:
        for test_name, results in all_results.items():
            if 'figure' in results:
                plt.figure(results['figure'].number)
                plt.title(f"{test_name} test results")
                plt.tight_layout()
        
        plt.show()
