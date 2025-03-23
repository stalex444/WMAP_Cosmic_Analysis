Utilities
=========

This section documents the utility modules and functions in the WMAP Cosmic Analysis framework.

utils.utils
----------

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

   Calculate bootstrap confidence intervals for a statistic with increased number of bootstrap samples for more reliable intervals.

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

.. py:function:: is_python2()

   Check if the code is running on Python 2.

   :return: True if running on Python 2, False otherwise
   :rtype: bool

.. py:function:: ensure_dir_exists(directory)

   Ensure that a directory exists, creating it if necessary.

   :param directory: Directory path
   :type directory: str
   :return: None

.. py:function:: format_time_estimate(seconds)

   Format a time estimate in a human-readable format.

   :param seconds: Time in seconds
   :type seconds: float
   :return: Formatted time string
   :rtype: str

utils.progress_reporter
---------------------

.. py:module:: utils.progress_reporter

.. py:class:: ProgressReporter

   Class for reporting progress during long-running operations.

   .. py:method:: __init__(total_steps, description='Progress', update_interval=1.0)

      Initialize the progress reporter.

      :param total_steps: Total number of steps
      :type total_steps: int
      :param description: Description of the operation
      :type description: str
      :param update_interval: Minimum interval between progress updates in seconds
      :type update_interval: float

   .. py:method:: update(current_step, additional_info=None)

      Update the progress.

      :param current_step: Current step
      :type current_step: int
      :param additional_info: Additional information to display
      :type additional_info: str
      :return: None

   .. py:method:: finish()

      Mark the operation as finished.

      :return: None

utils.timeout_handler
-------------------

.. py:module:: utils.timeout_handler

.. py:class:: TimeoutHandler

   Class for handling timeouts in long-running operations.

   .. py:method:: __init__(timeout_seconds)

      Initialize the timeout handler.

      :param timeout_seconds: Timeout in seconds
      :type timeout_seconds: int

   .. py:method:: check_timeout(start_time)

      Check if the operation has timed out.

      :param start_time: Start time of the operation
      :type start_time: float
      :return: True if timed out, False otherwise
      :rtype: bool

   .. py:method:: __enter__()

      Enter the context manager.

      :return: self
      :rtype: TimeoutHandler

   .. py:method:: __exit__(exc_type, exc_val, exc_tb)

      Exit the context manager.

      :param exc_type: Exception type
      :param exc_val: Exception value
      :param exc_tb: Exception traceback
      :return: True if the exception should be suppressed, False otherwise
      :rtype: bool
