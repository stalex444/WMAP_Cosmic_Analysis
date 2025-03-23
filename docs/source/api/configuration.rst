Configuration
=============

This section documents the configuration modules and functions in the WMAP Cosmic Analysis framework.

config.config_loader
------------------

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

Default Configuration
-------------------

The default configuration is defined in ``config/default_config.yaml`` and includes the following sections:

Data Configuration
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     wmap_data_path: "wmap_data/wmap_tt_spectrum_9yr_v5.txt"
     planck_data_path: "wmap_data/planck_tt_spectrum_2018.txt"
     preprocessing:
       detrend: false
       normalize: true
       smooth: false
       smooth_window: 5

Analysis Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   analysis:
     random_seed: 42
     num_simulations: 30
     timeout_seconds: 60
     early_stopping: true
     
     golden_ratio:
       phi_value: 1.618033988749895
       tolerance: 0.02
       phi_bias: 0.1
       
     transfer_entropy:
       max_data_points: 500
       num_bins: 10
       
     scale_transition:
       max_clusters: 10

Performance Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   performance:
     parallel: true
     n_jobs: -1  # Use all available cores

Visualization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   visualization:
     enabled: true
     dpi: 300
     formats: ["png", "pdf"]

Output Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   output:
     results_dir: "results"
     save_results: true
     result_format: "json"

Command Line Interface
--------------------

The configuration system is integrated with the command-line interface in ``run_wmap_analysis.py``, allowing users to override configuration values through command-line arguments:

.. code-block:: bash

   python run_wmap_analysis.py --config custom_config.yaml --timeout-seconds 90 --num-simulations 20
