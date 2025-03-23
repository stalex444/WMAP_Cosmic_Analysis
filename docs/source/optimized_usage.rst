Optimized Usage Guide
==================

This guide provides detailed instructions for running the optimized tests in the WMAP Cosmic Analysis framework using the command line interface. The optimized tests have been specifically enhanced to improve performance, prevent hanging, and ensure compatibility with Python 2.7.

Command Line Interface
--------------------

The WMAP Cosmic Analysis framework provides a comprehensive command line interface through the ``run_wmap_analysis.py`` script. This script allows you to run various tests with optimized parameters.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    python run_wmap_analysis.py [OPTIONS] [TESTS]

Common Options
~~~~~~~~~~~~

.. code-block:: text

    --timeout-seconds SECONDS   Maximum time in seconds for each test (default: 60)
    --num-simulations N         Number of Monte Carlo simulations (default: 30)
    --early-stopping            Enable early stopping based on statistical significance
    --visualize                 Generate visualizations
    --no-visualize              Disable visualizations
    --data-source SOURCE        Data source to use (wmap, planck, simulated, both)
    --seed SEED                 Random seed for reproducibility (default: 42)
    --phi-bias BIAS             Bias factor for golden ratio patterns (default: 0.1)

Running Optimized Tests
---------------------

Scale Transition Test
~~~~~~~~~~~~~~~~~~

The Scale Transition Test has been optimized to prevent hanging and improve performance.

.. code-block:: bash

    # Run with optimized parameters
    python run_wmap_analysis.py --scale-transition \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --max-clusters 10

    # Include visualization
    python run_wmap_analysis.py --scale-transition --visualize

Transfer Entropy Test
~~~~~~~~~~~~~~~~~~

The Transfer Entropy Test has been optimized to improve performance and prevent hanging.

.. code-block:: bash

    # Run with optimized parameters
    python run_wmap_analysis.py --transfer-entropy \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --max-data-points 500 \
        --num-bins 10

    # Include visualization
    python run_wmap_analysis.py --transfer-entropy --visualize

Golden Ratio Test
~~~~~~~~~~~~~~

The Golden Ratio Test has been enhanced with improved statistical analysis.

.. code-block:: bash

    # Run with enhanced parameters
    python run_wmap_analysis.py --golden-ratio \
        --phi-bias 0.1 \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --bootstrap-samples 10000

    # Include visualization
    python run_wmap_analysis.py --golden-ratio --visualize

Running Multiple Tests
-------------------

You can run multiple tests with optimized parameters in a single command:

.. code-block:: bash

    python run_wmap_analysis.py \
        --scale-transition \
        --transfer-entropy \
        --golden-ratio \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --visualize

Running All Tests
--------------

To run all tests with optimized parameters:

.. code-block:: bash

    python run_wmap_analysis.py --all \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --visualize

Using the Makefile
---------------

For convenience, the project includes a Makefile with targets for running optimized tests:

.. code-block:: bash

    # Run all tests with optimized parameters
    make run-optimized

    # Run with visualization
    make run-viz

Data Preprocessing Options
-----------------------

The framework provides several options for preprocessing the data before analysis:

.. code-block:: bash

    # Apply smoothing
    python run_wmap_analysis.py --smooth --smooth-window 5 --all

    # Normalize the data
    python run_wmap_analysis.py --normalize --all

    # Remove linear trend
    python run_wmap_analysis.py --detrend --all

    # Combine preprocessing options
    python run_wmap_analysis.py --smooth --normalize --detrend --all

Output Options
-----------

Control the output of the analysis with these options:

.. code-block:: bash

    # Save results to a specific directory
    python run_wmap_analysis.py --data-dir results/my_analysis --all

    # Generate detailed reports
    python run_wmap_analysis.py --report --all

    # Disable reports
    python run_wmap_analysis.py --no-report --all

Parallel Processing
----------------

The framework supports parallel processing to further improve performance:

.. code-block:: bash

    # Enable parallel processing
    python run_wmap_analysis.py --parallel --all

    # Specify number of parallel jobs
    python run_wmap_analysis.py --parallel --n-jobs 4 --all

    # Use all available cores
    python run_wmap_analysis.py --parallel --n-jobs -1 --all

Comparing WMAP and Planck Data
---------------------------

Compare results between WMAP and Planck datasets:

.. code-block:: bash

    # Run all tests on both datasets and compare
    python run_wmap_analysis.py --data-source both --compare --all

    # Run specific tests and compare
    python run_wmap_analysis.py --data-source both --compare \
        --golden-ratio --transfer-entropy

Example Workflows
--------------

Basic Analysis
~~~~~~~~~~~~

.. code-block:: bash

    # Download WMAP data
    python -c "from wmap_data.wmap_data_handler import download_wmap_data; download_wmap_data()"

    # Run all tests with optimized parameters
    python run_wmap_analysis.py --all \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --visualize

Comparative Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Download both WMAP and Planck data
    python -c "from wmap_data.wmap_data_handler import download_wmap_data; download_wmap_data(data_type='POWER_SPECTRUM'); download_wmap_data(data_type='PLANCK_POWER_SPECTRUM')"

    # Run comparative analysis with optimized parameters
    python run_wmap_analysis.py --data-source both --compare --all \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --visualize

    # Create comparison dashboard
    python visualization/comparison_dashboard.py \
        --wmap-results results/latest/wmap/analysis_results.json \
        --planck-results results/latest/planck/analysis_results.json \
        --output-dir results/latest/comparison

Focused Analysis
~~~~~~~~~~~~~

.. code-block:: bash

    # Run only the optimized tests
    python run_wmap_analysis.py \
        --scale-transition \
        --transfer-entropy \
        --golden-ratio \
        --timeout-seconds 60 \
        --num-simulations 30 \
        --early-stopping \
        --visualize

    # Focus on golden ratio patterns with increased sensitivity
    python run_wmap_analysis.py --golden-ratio \
        --phi-bias 0.2 \
        --timeout-seconds 90 \
        --num-simulations 50 \
        --bootstrap-samples 20000 \
        --visualize

Troubleshooting
-------------

If a test is still hanging despite the optimizations:

1. Decrease the ``--timeout-seconds`` value (e.g., 30 seconds)
2. Reduce the ``--num-simulations`` value (e.g., 20 or 15)
3. For the Scale Transition Test, reduce ``--max-clusters`` (e.g., 5)
4. For the Transfer Entropy Test, reduce ``--max-data-points`` (e.g., 300)

If you're getting memory errors:

1. Disable parallel processing with ``--no-parallel``
2. Reduce the data size with ``--data-size 2048`` (or smaller)
3. Run tests individually rather than all at once
