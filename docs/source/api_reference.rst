API Reference
=============

This section provides detailed documentation for the key modules and classes in the WMAP Cosmic Analysis framework.

Data Handling
------------

.. _wmap_data_handler:

wmap_data.wmap_data_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: wmap_data.wmap_data_handler

.. py:function:: download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')

   Download WMAP data files from the official LAMBDA archive.

   :param data_type: Type of data to download ('POWER_SPECTRUM', 'ILC_MAP', or 'ANALYSIS_MASK')
   :type data_type: str
   :param output_dir: Directory to save downloaded data
   :type output_dir: str
   :return: Path to the downloaded file
   :rtype: str

.. py:function:: load_wmap_data(file_path)

   Load WMAP data from a file.

   :param file_path: Path to the WMAP data file
   :type file_path: str
   :return: Loaded data as a numpy array
   :rtype: numpy.ndarray

.. py:function:: preprocess_data(data, detrend=False, normalize=False, smooth=False, smooth_window=5)

   Preprocess the data with optional detrending, normalization, and smoothing.

   :param data: Input data array
   :type data: numpy.ndarray
   :param detrend: Whether to remove linear trend
   :type detrend: bool
   :param normalize: Whether to normalize the data
   :type normalize: bool
   :param smooth: Whether to apply smoothing
   :type smooth: bool
   :param smooth_window: Window size for smoothing
   :type smooth_window: int
   :return: Preprocessed data
   :rtype: numpy.ndarray

Analysis Tests
-------------

.. _base_test:

analysis.base_test
~~~~~~~~~~~~~~~~

.. py:module:: analysis.base_test

.. py:class:: BaseTest

   Base class for all analysis tests.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, **kwargs)

      Run the test on the provided data.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param timeout_seconds: Maximum time in seconds for the test
      :type timeout_seconds: int
      :param num_simulations: Number of Monte Carlo simulations
      :type num_simulations: int
      :param early_stopping: Whether to stop early based on statistical significance
      :type early_stopping: bool
      :param visualize: Whether to generate visualizations
      :type visualize: bool
      :param kwargs: Additional parameters
      :type kwargs: dict
      :return: Test results
      :rtype: dict

   .. py:method:: _check_timeout(start_time, timeout_seconds)

      Check if the test has exceeded the timeout.

      :param start_time: Start time of the test
      :type start_time: float
      :param timeout_seconds: Timeout in seconds
      :type timeout_seconds: int
      :return: Whether the timeout has been exceeded
      :rtype: bool

   .. py:method:: _check_early_stopping(p_values, alpha=0.05, min_simulations=10)

      Check if early stopping criteria are met.

      :param p_values: List of p-values from simulations
      :type p_values: list
      :param alpha: Significance level
      :type alpha: float
      :param min_simulations: Minimum number of simulations before early stopping
      :type min_simulations: int
      :return: Whether to stop early
      :rtype: bool

.. _golden_ratio_test:

analysis.golden_ratio_test
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: analysis.golden_ratio_test

.. py:class:: GoldenRatioTest

   Test for golden ratio patterns in CMB data.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, phi_bias=0.1, **kwargs)

      Run the golden ratio test.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param timeout_seconds: Maximum time in seconds for the test
      :type timeout_seconds: int
      :param num_simulations: Number of Monte Carlo simulations
      :type num_simulations: int
      :param early_stopping: Whether to stop early based on statistical significance
      :type early_stopping: bool
      :param visualize: Whether to generate visualizations
      :type visualize: bool
      :param phi_bias: Bias factor for golden ratio patterns
      :type phi_bias: float
      :param kwargs: Additional parameters
      :type kwargs: dict
      :return: Test results including p-value, phi-optimality, and confidence intervals
      :rtype: dict

   .. py:method:: find_golden_ratio_pairs(ell, max_ell=1000, tolerance=0.02)

      Find pairs of multipole moments related by the golden ratio.

      :param ell: Array of multipole moments
      :type ell: numpy.ndarray
      :param max_ell: Maximum multipole moment to consider
      :type max_ell: int
      :param tolerance: Tolerance for considering a ratio as golden
      :type tolerance: float
      :return: List of (ell1, ell2) pairs related by the golden ratio
      :rtype: list

   .. py:method:: calculate_phi_optimality(data, gr_pairs)

      Calculate the phi-optimality score for the data.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param gr_pairs: List of golden ratio pairs
      :type gr_pairs: list
      :return: Phi-optimality score
      :rtype: float

.. _transfer_entropy_test:

analysis.transfer_entropy_test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: analysis.transfer_entropy_test

.. py:class:: TransferEntropyTest

   Test for information flow between scales using transfer entropy.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, max_data_points=500, num_bins=10, **kwargs)

      Run the transfer entropy test.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param timeout_seconds: Maximum time in seconds for the test
      :type timeout_seconds: int
      :param num_simulations: Number of Monte Carlo simulations
      :type num_simulations: int
      :param early_stopping: Whether to stop early based on statistical significance
      :type early_stopping: bool
      :param visualize: Whether to generate visualizations
      :type visualize: bool
      :param max_data_points: Maximum number of data points to use
      :type max_data_points: int
      :param num_bins: Number of bins for probability estimation
      :type num_bins: int
      :param kwargs: Additional parameters
      :type kwargs: dict
      :return: Test results including p-value, transfer entropy, and reference value
      :rtype: dict

   .. py:method:: calculate_transfer_entropy(x, y, num_bins=10)

      Calculate transfer entropy from x to y.

      :param x: Source time series
      :type x: numpy.ndarray
      :param y: Target time series
      :type y: numpy.ndarray
      :param num_bins: Number of bins for probability estimation
      :type num_bins: int
      :return: Transfer entropy value
      :rtype: float

.. _scale_transition_test:

analysis.scale_transition_test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: analysis.scale_transition_test

.. py:class:: ScaleTransitionTest

   Test for scale transitions in CMB data.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, max_clusters=10, **kwargs)

      Run the scale transition test.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param timeout_seconds: Maximum time in seconds for the test
      :type timeout_seconds: int
      :param num_simulations: Number of Monte Carlo simulations
      :type num_simulations: int
      :param early_stopping: Whether to stop early based on statistical significance
      :type early_stopping: bool
      :param visualize: Whether to generate visualizations
      :type visualize: bool
      :param max_clusters: Maximum number of clusters to consider
      :type max_clusters: int
      :param kwargs: Additional parameters
      :type kwargs: dict
      :return: Test results including p-value, transition points, and cluster quality
      :rtype: dict

   .. py:method:: find_scale_transitions(data, max_clusters=10)

      Find scale transition points in the data.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param max_clusters: Maximum number of clusters to consider
      :type max_clusters: int
      :return: List of transition points and cluster quality measure
      :rtype: tuple

Visualization
------------

.. _visualization_utils:

visualization.visualization_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: visualization.visualization_utils

.. py:function:: save_figure(fig, filename, dpi=300, formats=None)

   Save a figure in multiple formats with proper directory creation.

   :param fig: The figure to save
   :type fig: matplotlib.figure.Figure
   :param filename: Base filename without extension
   :type filename: str
   :param dpi: Resolution for raster formats
   :type dpi: int
   :param formats: List of formats to save (default: ['png', 'pdf', 'svg'])
   :type formats: list
   :return: None

.. py:function:: plot_power_spectrum(ell, cl, errors=None, title="CMB Power Spectrum", color=None, ax=None, label=None, show_peaks=False)

   Plot a CMB power spectrum with optional error bars and peak identification.

   :param ell: Multipole moments
   :type ell: array-like
   :param cl: Power spectrum values
   :type cl: array-like
   :param errors: Error values for the power spectrum
   :type errors: array-like
   :param title: Plot title
   :type title: str
   :param color: Line color
   :type color: str
   :param ax: Axes to plot on
   :type ax: matplotlib.axes.Axes
   :param label: Label for the line
   :type label: str
   :param show_peaks: Whether to identify and label acoustic peaks
   :type show_peaks: bool
   :return: The figure and axes objects
   :rtype: tuple

.. py:function:: plot_golden_ratio_pairs(ell, cl, gr_pairs, title="Golden Ratio Pairs in CMB", ax=None)

   Visualize golden ratio pairs in the CMB power spectrum.

   :param ell: Multipole moments
   :type ell: array-like
   :param cl: Power spectrum values
   :type cl: array-like
   :param gr_pairs: List of (ell1, ell2) pairs related by the golden ratio
   :type gr_pairs: list
   :param title: Plot title
   :type title: str
   :param ax: Axes to plot on
   :type ax: matplotlib.axes.Axes
   :return: The figure and axes objects
   :rtype: tuple

.. py:function:: create_summary_dashboard(results_dict, output_file='results_summary.png')

   Create a comprehensive dashboard summarizing all analysis results.

   :param results_dict: Dictionary containing results from various analyses
   :type results_dict: dict
   :param output_file: Path to save the dashboard image
   :type output_file: str
   :return: The figure object
   :rtype: matplotlib.figure.Figure

Configuration
------------

.. _config_loader:

config.config_loader
~~~~~~~~~~~~~~~~~~

.. py:module:: config.config_loader

.. py:function:: load_config(config_path=None, override_dict=None)

   Load configuration from a YAML file with optional overrides.

   :param config_path: Path to the configuration YAML file
   :type config_path: str
   :param override_dict: Dictionary with values to override in the configuration
   :type override_dict: dict
   :return: Configuration dictionary
   :rtype: dict

.. py:function:: validate_config(config)

   Validate the configuration dictionary.

   :param config: Configuration dictionary to validate
   :type config: dict
   :return: None
   :raises: ConfigurationError if the configuration is invalid

.. py:function:: config_from_args(args)

   Create a configuration dictionary from command line arguments.

   :param args: Command line arguments
   :type args: argparse.Namespace
   :return: Configuration dictionary with overrides from command line arguments
   :rtype: dict

.. py:function:: get_config(args=None, config_path=None)

   Get configuration from command line arguments and/or config file.

   :param args: Command line arguments
   :type args: argparse.Namespace
   :param config_path: Path to the configuration YAML file
   :type config_path: str
   :return: Configuration dictionary
   :rtype: dict

Utilities
--------

.. _utils:

utils.utils
~~~~~~~~~

.. py:module:: utils.utils

.. py:function:: setup_logging(level='INFO', log_file=None)

   Set up logging configuration.

   :param level: Logging level
   :type level: str
   :param log_file: Path to log file
   :type log_file: str
   :return: Logger object
   :rtype: logging.Logger

.. py:function:: generate_simulated_data(size=2048, phi_bias=0.0, seed=None)

   Generate simulated data with optional golden ratio bias.

   :param size: Size of the data array
   :type size: int
   :param phi_bias: Bias factor for golden ratio patterns
   :type phi_bias: float
   :param seed: Random seed for reproducibility
   :type seed: int
   :return: Simulated data array
   :rtype: numpy.ndarray

.. py:function:: bootstrap_confidence_interval(data, statistic_func, alpha=0.05, n_bootstrap=10000, seed=None)

   Calculate bootstrap confidence intervals for a statistic.

   :param data: Input data
   :type data: numpy.ndarray
   :param statistic_func: Function to calculate the statistic
   :type statistic_func: callable
   :param alpha: Significance level
   :type alpha: float
   :param n_bootstrap: Number of bootstrap samples
   :type n_bootstrap: int
   :param seed: Random seed for reproducibility
   :type seed: int
   :return: Tuple of (lower_bound, upper_bound)
   :rtype: tuple
