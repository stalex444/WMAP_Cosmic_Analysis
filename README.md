# WMAP Cosmic Consciousness Analysis

This repository contains tools for analyzing WMAP (Wilkinson Microwave Anisotropy Probe) Cosmic Microwave Background (CMB) data within the Cosmic Consciousness Analysis framework. It allows for comparative analysis between WMAP and Planck datasets to investigate patterns of cosmic organization.

## Overview

The WMAP Cosmic Analysis framework extends the original Cosmic Consciousness Analysis framework to incorporate WMAP data, enabling:

1. Downloading and processing WMAP CMB temperature maps and power spectrum data
2. Running the full suite of 10 statistical tests on WMAP data
3. Comparing results between WMAP and Planck datasets
4. Visualizing the comparative analysis through comprehensive dashboards

## Repository Structure

```
WMAP_Cosmic_Analysis/
├── wmap_data/               # WMAP data handling modules and downloaded data
│   ├── wmap_data_handler.py # Core module for WMAP data operations
│   └── README.md            # Instructions for WMAP data
├── visualization/           # Visualization tools
│   └── comparison_dashboard.py # Dashboard for WMAP vs Planck comparison
├── analysis/                # Analysis modules specific to WMAP
├── utils/                   # Utility functions
├── tests/                   # Unit tests
├── run_wmap_analysis.py     # Main script to run analysis
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 2.7 or Python 3.x
- NumPy
- SciPy
- Matplotlib
- healpy (for HEALPix operations)
- astropy (for FITS file handling)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/WMAP_Cosmic_Analysis.git
   cd WMAP_Cosmic_Analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the Cosmic Consciousness Analysis framework is available in a parent directory or modify the import paths in the code.

## Usage

### Downloading WMAP Data

The `wmap_data_handler.py` module provides functions to download WMAP data:

```python
from wmap_data.wmap_data_handler import download_wmap_data

# Download WMAP power spectrum data
download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')

# Download WMAP ILC map
download_wmap_data(data_type='ILC_MAP', output_dir='wmap_data')

# Download WMAP analysis mask
download_wmap_data(data_type='ANALYSIS_MASK', output_dir='wmap_data')
```

### Running Analysis

The main analysis script `run_wmap_analysis.py` provides a comprehensive interface for analyzing WMAP data:

```bash
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
```

### Command-line Options

The `run_wmap_analysis.py` script supports numerous options:

#### Data Source Options
- `--data-source {wmap,planck,simulated,both}`: Data source to use (default: wmap)
- `--data-dir`: Directory to save results (default: results)
- `--wmap-file`: Path to WMAP data file (if not using default)
- `--planck-file`: Path to Planck data file (if not using default)
- `--use-power-spectrum`: Use power spectrum instead of map
- `--data-size`: Size of data to analyze (default: 4096)
- `--seed`: Random seed for reproducibility (default: 42)

#### Preprocessing Options
- `--smooth`: Apply smoothing to the data
- `--smooth-window`: Window size for smoothing (default: 5)
- `--normalize`: Normalize the data
- `--detrend`: Remove linear trend from data

#### Analysis Options
- `--phi-bias`: Bias factor for golden ratio tests (default: 0.1)
- `--golden-ratio`: Run golden ratio test
- `--coherence-analysis`: Run coherence analysis test
- `--gr-specific-coherence`: Run GR-specific coherence test
- `--hierarchical-organization`: Run hierarchical organization test
- `--information-integration`: Run information integration test
- `--scale-transition`: Run scale transition test
- `--resonance-analysis`: Run resonance analysis test
- `--fractal-analysis`: Run fractal analysis test
- `--meta-coherence`: Run meta-coherence test
- `--transfer-entropy`: Run transfer entropy test
- `--all`: Run all tests

#### Output Options
- `--visualize/--no-visualize`: Generate/don't generate visualizations
- `--report/--no-report`: Generate/don't generate detailed reports
- `--parallel/--no-parallel`: Use/don't use parallel processing
- `--n-jobs`: Number of parallel jobs (default: -1, all cores)

#### Comparison Options
- `--compare`: Compare WMAP and Planck results

### Visualization

The `visualization/comparison_dashboard.py` module provides tools for creating comprehensive comparison dashboards:

```bash
# Create a comparison dashboard from analysis results
python visualization/comparison_dashboard.py \
    --wmap-results results/wmap/analysis_results.json \
    --planck-results results/planck/analysis_results.json \
    --output-dir results/comparison
```

## Statistical Tests

The framework includes 10 statistical tests for analyzing CMB data:

1. **Golden Ratio Significance Test**: Tests if multipoles related by the golden ratio have statistically significant power
2. **Coherence Analysis Test**: Evaluates if the CMB spectrum shows more coherence than random chance
3. **GR-Specific Coherence Test**: Tests coherence specifically in golden ratio related regions
4. **Hierarchical Organization Test**: Checks for hierarchical patterns based on the golden ratio
5. **Information Integration Test**: Measures mutual information between adjacent spectrum regions
6. **Scale Transition Test**: Analyzes scale boundaries where organizational principles change
7. **Resonance Analysis Test**: Tests for resonance patterns in the power spectrum
8. **Fractal Analysis Test**: Uses the Hurst exponent to evaluate fractal behavior
9. **Meta-Coherence Test**: Analyzes coherence of local coherence measures
10. **Transfer Entropy Test**: Measures information flow between scales

Each test calculates a phi-optimality score (bounded between -1 and 1) and performs statistical significance testing using Monte Carlo simulations.

## Comparing WMAP and Planck Data

The framework provides tools for comparing results between WMAP and Planck datasets:

1. **Raw Data Comparison**: Correlation analysis between the raw datasets
2. **Power Spectrum Comparison**: Analysis of power distribution differences
3. **Phi Optimality Comparison**: Comparison of phi optimality scores across all tests
4. **Significance Comparison**: Comparison of statistical significance (p-values) across all tests
5. **Combined Significance**: Comparison of overall significance using Fisher's method
6. **Mathematical Constants**: Comparison of best-fit mathematical constants

## Example Workflow

1. **Download WMAP data**:
   ```python
   from wmap_data.wmap_data_handler import download_wmap_data
   download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')
   download_wmap_data(data_type='ILC_MAP', output_dir='wmap_data')
   ```

2. **Run analysis on both WMAP and Planck data**:
   ```bash
   python run_wmap_analysis.py --data-source both --compare --all
   ```

3. **Create comparison dashboard**:
   ```bash
   python visualization/comparison_dashboard.py \
       --wmap-results results/wmap_analysis_20230615_123456/wmap/analysis_results.json \
       --planck-results results/wmap_analysis_20230615_123456/planck/analysis_results.json \
       --output-dir results/wmap_analysis_20230615_123456
   ```

## References

- WMAP Data: [NASA LAMBDA - WMAP Products](https://lambda.gsfc.nasa.gov/product/wmap/current/)
- Planck Data: [ESA Planck Legacy Archive](https://pla.esac.esa.int/)
- HEALPix: [HEALPix - Hierarchical Equal Area isoLatitude Pixelization](https://healpix.sourceforge.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
