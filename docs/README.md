# WMAP Cosmic Analysis Documentation

This directory contains the Sphinx documentation for the WMAP Cosmic Analysis framework.

## Building the Documentation

### Prerequisites

To build the documentation, you need:

- Python 2.7 or Python 3.x
- Sphinx
- sphinx_rtd_theme (Read the Docs theme)
- Xcode Command Line Tools (for macOS users)

### Installation

1. Install Sphinx and the Read the Docs theme:

```bash
pip install sphinx sphinx_rtd_theme
```

2. For macOS users, install Xcode Command Line Tools if prompted:

```bash
xcode-select --install
```

### Building

You can build the documentation using the provided script:

```bash
./build_docs.sh
```

Or manually with the Makefile:

```bash
cd docs
make html
```

The built documentation will be available in the `build/html` directory. Open `build/html/index.html` in your web browser to view it.

## Documentation Structure

- `source/`: Contains the source files for the documentation
  - `conf.py`: Sphinx configuration file
  - `index.rst`: Main index page
  - `introduction.rst`: Introduction to the framework
  - `installation.rst`: Installation instructions
  - `usage.rst`: Usage guide
  - `theoretical_background.rst`: Theoretical background
  - `api_reference.rst`: API reference documentation
  - `examples.rst`: Example code
  - `tutorials.rst`: Step-by-step tutorials

- `build/`: Contains the built documentation (generated when you build)
- `Makefile`: Used by Sphinx to build the documentation
- `build_docs.sh`: Convenience script for building the documentation

## Recent Optimizations and Enhancements

The documentation reflects recent optimizations to the framework:

1. **Scale Transition Test** optimizations:
   - Added timeout mechanism to prevent indefinite hanging
   - Reduced default simulations from 100 to 30
   - Implemented early stopping based on statistical significance
   - Limited the number of clusters to improve performance
   - Added a --visualize flag for optional visualization
   - Made the code Python 2.7 compatible

2. **Transfer Entropy Test** optimizations:
   - Optimized the calculate_transfer_entropy function using NumPy's histogram functions
   - Limited the data points used in calculations to 500 for better performance
   - Reduced default simulations from 100 to 30
   - Added early stopping based on statistical significance
   - Implemented timeout mechanism to prevent hanging

3. **Statistical Enhancements**:
   - Enhanced phi-optimality calculation using a logarithmic scale and sigmoid function
   - Improved bootstrap confidence intervals with 10,000 samples
   - Added parameter controls for reproducibility and customization

## Contributing to the Documentation

To contribute to the documentation:

1. Edit the relevant `.rst` files in the `source/` directory
2. Build the documentation to preview your changes
3. Submit your changes via pull request
