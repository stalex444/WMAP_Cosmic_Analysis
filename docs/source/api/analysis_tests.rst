Analysis Tests
==============

This section documents the analysis test modules and classes in the WMAP Cosmic Analysis framework.

Base Test
---------

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

Golden Ratio Test
----------------

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

      Calculate the phi-optimality score for the data using a logarithmic scale and sigmoid function for more stable results.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param gr_pairs: List of golden ratio pairs
      :type gr_pairs: list
      :return: Phi-optimality score
      :rtype: float

Transfer Entropy Test
-------------------

.. py:module:: analysis.transfer_entropy_test

.. py:class:: TransferEntropyTest

   Test for information flow between scales using transfer entropy.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, max_data_points=500, num_bins=10, **kwargs)

      Run the transfer entropy test with optimized parameters.

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

      Calculate transfer entropy from x to y using NumPy's histogram functions for efficient probability estimation.

      :param x: Source time series
      :type x: numpy.ndarray
      :param y: Target time series
      :type y: numpy.ndarray
      :param num_bins: Number of bins for probability estimation
      :type num_bins: int
      :return: Transfer entropy value
      :rtype: float

Scale Transition Test
------------------

.. py:module:: analysis.scale_transition_test

.. py:class:: ScaleTransitionTest

   Test for scale transitions in CMB data.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, max_clusters=10, **kwargs)

      Run the scale transition test with optimized parameters.

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

      Find scale transition points in the data with a limited number of clusters for better performance.

      :param data: The data to analyze
      :type data: numpy.ndarray
      :param max_clusters: Maximum number of clusters to consider
      :type max_clusters: int
      :return: List of transition points and cluster quality measure
      :rtype: tuple

GR-Specific Coherence Test
------------------------

.. py:module:: analysis.gr_specific_coherence_test

.. py:class:: GRSpecificCoherenceTest

   Test for coherence specifically related to the golden ratio.

   .. py:method:: run(data, timeout_seconds=60, num_simulations=30, early_stopping=True, visualize=False, **kwargs)

      Run the GR-specific coherence test.

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
      :return: Test results including p-value and coherence measure
      :rtype: dict
