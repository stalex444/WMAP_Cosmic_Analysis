Data Handling
=============

This section documents the modules and functions for handling WMAP data.

wmap_data.wmap_data_handler
--------------------------

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

.. py:function:: align_datasets(dataset1, dataset2)

   Align two datasets to ensure they cover the same multipole range.

   :param dataset1: First dataset
   :type dataset1: numpy.ndarray
   :param dataset2: Second dataset
   :type dataset2: numpy.ndarray
   :return: Tuple of aligned datasets
   :rtype: tuple
