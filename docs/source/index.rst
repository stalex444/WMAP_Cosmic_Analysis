Welcome to WMAP Cosmic Analysis Documentation
==========================================

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   optimized_usage
   theoretical_background
   optimized_tests
   statistical_enhancements
   api_reference
   examples
   tutorials

Introduction
-----------

The WMAP Cosmic Analysis framework extends the original Cosmic Consciousness Analysis framework to incorporate WMAP (Wilkinson Microwave Anisotropy Probe) data, enabling comprehensive analysis of Cosmic Microwave Background (CMB) patterns. This documentation provides detailed information on the theoretical background, installation instructions, usage examples, and API reference.

Key Features
-----------

* Download and process WMAP CMB temperature maps and power spectrum data
* Run a full suite of 10 statistical tests on WMAP data
* Compare results between WMAP and Planck datasets
* Visualize the comparative analysis through comprehensive dashboards
* Investigate patterns of cosmic organization using the golden ratio and other mathematical constants

Recent Optimizations
------------------

The framework has been recently optimized to improve performance and prevent hanging:

1. **Scale Transition Test** optimizations:
   * Added timeout mechanism
   * Reduced default simulations from 100 to 30
   * Implemented early stopping based on statistical significance
   * Limited the number of clusters to improve performance

2. **Transfer Entropy Test** optimizations:
   * Optimized the calculate_transfer_entropy function
   * Limited the data points used in calculations to 500
   * Reduced default simulations from 100 to 30
   * Added early stopping based on statistical significance

3. **Statistical Enhancements**:
   * Enhanced phi-optimality calculation
   * Improved bootstrap confidence intervals with 10,000 samples
   * Added parameter controls for reproducibility

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
