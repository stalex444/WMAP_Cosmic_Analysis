Visualization
=============

This section documents the visualization modules and functions in the WMAP Cosmic Analysis framework.

visualization.visualization_utils
-------------------------------

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

.. py:function:: create_comparison_plot(metric_name, values, labels, p_values=None)

   Create a bar plot comparing a metric across different datasets.

   :param metric_name: Name of the metric being compared
   :type metric_name: str
   :param values: List of metric values
   :type values: list
   :param labels: List of labels for each value
   :type labels: list
   :param p_values: List of p-values for statistical significance
   :type p_values: list
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

visualization.comparison_dashboard
--------------------------------

.. py:module:: visualization.comparison_dashboard

.. py:function:: create_comparison_dashboard(wmap_results, planck_results, output_dir='results/comparison')

   Create a comprehensive dashboard comparing WMAP and Planck analysis results.

   :param wmap_results: Dictionary containing WMAP analysis results
   :type wmap_results: dict
   :param planck_results: Dictionary containing Planck analysis results
   :type planck_results: dict
   :param output_dir: Directory to save the dashboard images
   :type output_dir: str
   :return: Dictionary mapping test names to figure objects
   :rtype: dict

.. py:function:: plot_power_spectrum_comparison(wmap_data, planck_data, output_file='power_spectrum_comparison.png')

   Create a plot comparing WMAP and Planck power spectra.

   :param wmap_data: WMAP power spectrum data
   :type wmap_data: numpy.ndarray
   :param planck_data: Planck power spectrum data
   :type planck_data: numpy.ndarray
   :param output_file: Path to save the comparison image
   :type output_file: str
   :return: The figure object
   :rtype: matplotlib.figure.Figure
