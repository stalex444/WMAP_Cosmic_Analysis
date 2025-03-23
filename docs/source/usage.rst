Usage Guide
===========

This section provides detailed instructions on how to use the WMAP Cosmic Analysis framework for analyzing CMB data.

Basic Usage
-----------

The main entry point for analysis is the ``run_wmap_analysis.py`` script, which provides a comprehensive interface for analyzing WMAP data:

.. code-block:: bash

   # Run all tests on WMAP data
   python run_wmap_analysis.py --all

   # Run specific tests
   python run_wmap_analysis.py --golden-ratio --coherence-analysis --fractal-analysis

   # Compare WMAP and Planck data
   python run_wmap_analysis.py --data-source both --compare --all

   # Use power spectrum instead of map
   python run_wmap_analysis.py --use-power-spectrum --all

   # Apply preprocessing
   python run_wmap_analysis.py --smooth --normalize --detrend --all

Command-line Options
-------------------

The ``run_wmap_analysis.py`` script supports numerous options:

Data Source Options
~~~~~~~~~~~~~~~~~~

- ``--data-source {wmap,planck,simulated,both}``: Data source to use (default: wmap)
- ``--data-dir``: Directory to save results (default: results)
- ``--wmap-file``: Path to WMAP data file (if not using default)
- ``--planck-file``: Path to Planck data file (if not using default)
- ``--use-power-spectrum``: Use power spectrum instead of map
- ``--data-size``: Size of data to analyze (default: 4096)
- ``--seed``: Random seed for reproducibility (default: 42)

Preprocessing Options
~~~~~~~~~~~~~~~~~~~

- ``--smooth``: Apply smoothing to the data
- ``--smooth-window``: Window size for smoothing (default: 5)
- ``--normalize``: Normalize the data
- ``--detrend``: Remove linear trend from data

Analysis Options
~~~~~~~~~~~~~~

- ``--phi-bias``: Bias factor for golden ratio tests (default: 0.1)
- ``--golden-ratio``: Run golden ratio test
- ``--coherence-analysis``: Run coherence analysis test
- ``--gr-specific-coherence``: Run GR-specific coherence test
- ``--hierarchical-organization``: Run hierarchical organization test
- ``--information-integration``: Run information integration test
- ``--scale-transition``: Run scale transition test
- ``--resonance-analysis``: Run resonance analysis test
- ``--fractal-analysis``: Run fractal analysis test
- ``--meta-coherence``: Run meta-coherence test
- ``--transfer-entropy``: Run transfer entropy test
- ``--all``: Run all tests

Performance Options
~~~~~~~~~~~~~~~~

- ``--timeout-seconds``: Maximum time in seconds for each test (default: 60)
- ``--num-simulations``: Number of Monte Carlo simulations (default: 30)
- ``--early-stopping``: Enable early stopping based on statistical significance (default: true)
- ``--visualize/--no-visualize``: Generate/don't generate visualizations
- ``--parallel/--no-parallel``: Use/don't use parallel processing
- ``--n-jobs``: Number of parallel jobs (default: -1, all cores)

Output Options
~~~~~~~~~~~~

- ``--report/--no-report``: Generate/don't generate detailed reports
- ``--output-format {json,csv,both}``: Format for saving results (default: json)

Comparison Options
~~~~~~~~~~~~~~~

- ``--compare``: Compare WMAP and Planck results

Performance Optimizations
-----------------------

The framework includes several optimizations to improve performance and prevent hanging:

Scale Transition Test
~~~~~~~~~~~~~~~~~~~~

The Scale Transition Test has been optimized with:

- A timeout mechanism to prevent indefinite hanging (default: 60 seconds)
- Reduced default simulations from 100 to 30 for better performance
- Early stopping based on statistical significance
- Limited number of clusters (default: 10) to improve performance

Example usage with optimizations:

.. code-block:: bash

   python run_wmap_analysis.py --scale-transition --timeout-seconds 120 --num-simulations 20 --no-visualize

Transfer Entropy Test
~~~~~~~~~~~~~~~~~~~~

The Transfer Entropy Test has been optimized with:

- Efficient calculation using NumPy's histogram functions
- Limited data points (default: 500) for better performance
- Reduced default simulations from 100 to 30
- Early stopping based on statistical significance
- Timeout mechanism to prevent hanging

Example usage with optimizations:

.. code-block:: bash

   python run_wmap_analysis.py --transfer-entropy --timeout-seconds 90 --max-data-points 300 --num-simulations 20

Python 2.7 Compatibility
-----------------------

The framework maintains compatibility with Python 2.7 using:

- Appropriate string formatting
- Proper division operators
- Compatible visualization code for older matplotlib versions

To run with Python 2.7:

.. code-block:: bash

   python2 run_wmap_analysis.py --all

Working with Visualization
-------------------------

Visualizations can be toggled with the ``--visualize/--no-visualize`` flag:

.. code-block:: bash

   # Run with visualizations
   python run_wmap_analysis.py --all --visualize

   # Run without visualizations (faster)
   python run_wmap_analysis.py --all --no-visualize

The ``visualization/comparison_dashboard.py`` module provides tools for creating comprehensive comparison dashboards:

.. code-block:: bash

   # Create a comparison dashboard from analysis results
   python visualization/comparison_dashboard.py \
       --wmap-results results/wmap/analysis_results.json \
       --planck-results results/planck/analysis_results.json \
       --output-dir results/comparison

Example Workflows
-----------------

Basic Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~

1. **Download WMAP data**:

   .. code-block:: python

      from wmap_data.wmap_data_handler import download_wmap_data
      download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')

2. **Run a specific test with optimized parameters**:

   .. code-block:: bash

      python run_wmap_analysis.py --golden-ratio --timeout-seconds 60 --num-simulations 30 --early-stopping

3. **View the results**:

   Results are saved in the specified output directory (default: ``results/``).

Comparative Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Download both WMAP and Planck data**:

   .. code-block:: python

      from wmap_data.wmap_data_handler import download_wmap_data
      download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')
      download_wmap_data(data_type='PLANCK_SPECTRUM', output_dir='wmap_data')

2. **Run analysis on both datasets**:

   .. code-block:: bash

      python run_wmap_analysis.py --data-source both --compare --all --timeout-seconds 60

3. **Create comparison dashboard**:

   .. code-block:: bash

      python visualization/comparison_dashboard.py \
          --wmap-results results/wmap_analysis_20230615_123456/wmap/analysis_results.json \
          --planck-results results/wmap_analysis_20230615_123456/planck/analysis_results.json \
          --output-dir results/wmap_analysis_20230615_123456

Batch Processing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

For analyzing multiple parameter combinations:

1. **Create a parameter sweep script**:

   .. code-block:: python

      import itertools
      import subprocess

      # Define parameter ranges
      phi_bias_values = [0.0, 0.1, 0.2, 0.3]
      num_simulations = [20, 30]
      tests = ['--golden-ratio', '--transfer-entropy', '--scale-transition']

      # Generate all combinations
      combinations = list(itertools.product(phi_bias_values, num_simulations, tests))

      # Run each combination
      for phi_bias, num_sim, test in combinations:
          cmd = f"python run_wmap_analysis.py {test} --phi-bias {phi_bias} --num-simulations {num_sim} --timeout-seconds 60 --no-visualize"
          print(f"Running: {cmd}")
          subprocess.call(cmd, shell=True)

2. **Run the parameter sweep**:

   .. code-block:: bash

      python parameter_sweep.py

3. **Aggregate and analyze results**:

   Use the provided analysis tools to compare results across parameter combinations.

Advanced Usage
--------------

Custom Data Analysis
~~~~~~~~~~~~~~~~~~~

You can use the framework's components in your own Python scripts:

.. code-block:: python

   from wmap_data.wmap_data_handler import load_wmap_data
   from analysis.golden_ratio_test import GoldenRatioTest
   from analysis.transfer_entropy_test import TransferEntropyTest

   # Load data
   wmap_data = load_wmap_data('wmap_data/wmap_tt_spectrum_9yr_v5.txt')

   # Run golden ratio test
   gr_test = GoldenRatioTest()
   gr_results = gr_test.run(
       wmap_data, 
       timeout_seconds=60,
       num_simulations=30,
       early_stopping=True,
       visualize=True
   )

   # Run transfer entropy test with optimized parameters
   te_test = TransferEntropyTest()
   te_results = te_test.run(
       wmap_data,
       max_data_points=500,
       timeout_seconds=60,
       num_simulations=30,
       early_stopping=True
   )

   # Print results
   print(f"Golden Ratio Test p-value: {gr_results['p_value']}")
   print(f"Transfer Entropy Test p-value: {te_results['p_value']}")

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~

The framework supports YAML-based configuration:

1. **Create a custom configuration file**:

   .. code-block:: yaml

      # custom_config.yaml
      data:
        wmap_data_path: "my_data/wmap_custom.txt"
        preprocessing:
          detrend: true
          normalize: true

      analysis:
        random_seed: 42
        num_simulations: 20
        timeout_seconds: 90
        early_stopping: true
        
        golden_ratio:
          phi_value: 1.618033988749895
          tolerance: 0.02
          
        transfer_entropy:
          max_data_points: 400
          num_bins: 12

      performance:
        parallel: true
        n_jobs: 4

2. **Use the configuration in your analysis**:

   .. code-block:: bash

      python run_wmap_analysis.py --config custom_config.yaml --all

   Or in Python:

   .. code-block:: python

      from config.config_loader import load_config
      from analysis.run_analysis import run_all_tests

      # Load custom configuration
      config = load_config('custom_config.yaml')

      # Run analysis with custom configuration
      results = run_all_tests(config)

Extending the Framework
----------------------

Adding a New Test
~~~~~~~~~~~~~~~~~

To add a new test to the framework:

1. **Create a new test class** in the ``analysis`` directory:

   .. code-block:: python

      # analysis/my_new_test.py
      import numpy as np
      from .base_test import BaseTest

      class MyNewTest(BaseTest):
          """
          My new test for analyzing CMB data.
          """
          
          def __init__(self):
              super(MyNewTest, self).__init__()
              self.name = "my_new_test"
          
          def run(self, data, timeout_seconds=60, num_simulations=30, 
                  early_stopping=True, visualize=False, **kwargs):
              """
              Run the test on the provided data.
              
              Parameters
              ----------
              data : array-like
                  The data to analyze
              timeout_seconds : int, optional
                  Maximum time in seconds for the test
              num_simulations : int, optional
                  Number of Monte Carlo simulations
              early_stopping : bool, optional
                  Whether to stop early based on statistical significance
              visualize : bool, optional
                  Whether to generate visualizations
              **kwargs : dict
                  Additional parameters
                  
              Returns
              -------
              dict
                  Test results
              """
              # Implement your test here
              # ...
              
              # Return results
              return {
                  'p_value': p_value,
                  'statistic': statistic,
                  # Other results...
              }

2. **Register the test** in ``analysis/__init__.py``:

   .. code-block:: python

      from .my_new_test import MyNewTest
      
      # Update the test registry
      TEST_REGISTRY = {
          # Existing tests...
          'my_new_test': MyNewTest,
      }

3. **Add command-line option** in ``run_wmap_analysis.py``:

   .. code-block:: python

      parser.add_argument('--my-new-test', action='store_true',
                          help='Run my new test')

4. **Update the configuration** in ``config/default_config.yaml``:

   .. code-block:: yaml

      analysis:
        # Existing configuration...
        my_new_test:
          param1: value1
          param2: value2
