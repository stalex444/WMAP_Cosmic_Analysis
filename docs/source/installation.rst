Installation
============

This section provides detailed instructions for setting up the WMAP Cosmic Analysis framework on your system.

Prerequisites
------------

The framework requires the following:

* Python 2.7 or Python 3.x (Python 2.7 compatibility is maintained for legacy systems)
* Git (for cloning the repository)
* Approximately 2GB of free disk space (for WMAP data and analysis results)

Required Packages
----------------

The following Python packages are required:

* numpy (≥1.16.0): For numerical computations
* scipy (≥1.2.0): For scientific computing
* matplotlib (≥3.0.0): For visualization
* healpy (≥1.12.0): For HEALPix operations
* astropy (≥3.0.0): For FITS file handling
* scikit-learn (≥0.20.0): For machine learning algorithms
* pandas (≥0.24.0): For data manipulation
* seaborn (≥0.9.0): For statistical data visualization
* joblib (≥0.13.0): For parallel computing
* tqdm (≥4.31.0): For progress bars

These version constraints ensure compatibility with both Python 2.7 and Python 3.x.

Basic Installation
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/WMAP_Cosmic_Analysis.git
      cd WMAP_Cosmic_Analysis

2. Install the required packages:

   .. code-block:: bash

      pip install -r requirements.txt

   For Python 2.7, you may need to use pip2:

   .. code-block:: bash

      pip2 install -r requirements.txt

3. Verify the installation:

   .. code-block:: bash

      python -c "import wmap_data.wmap_data_handler; print('Installation successful')"

Setting Up a Virtual Environment
------------------------------

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

Using venv (Python 3.x):

.. code-block:: bash

   python -m venv wmap_env
   source wmap_env/bin/activate  # On Windows: wmap_env\\Scripts\\activate
   pip install -r requirements.txt

Using virtualenv (Python 2.7 or 3.x):

.. code-block:: bash

   pip install virtualenv
   virtualenv wmap_env
   source wmap_env/bin/activate  # On Windows: wmap_env\\Scripts\\activate
   pip install -r requirements.txt

Docker Installation
-----------------

For a fully reproducible environment, you can use Docker:

1. Build the Docker image:

   .. code-block:: bash

      docker build -t wmap_cosmic_analysis .

2. Run the container:

   .. code-block:: bash

      docker run -it --name wmap_analysis -v $(pwd)/results:/app/results wmap_cosmic_analysis

This will mount the results directory from the container to your local machine, allowing you to access analysis results.

Downloading WMAP Data
-------------------

After installation, you need to download the WMAP data:

.. code-block:: python

   from wmap_data.wmap_data_handler import download_wmap_data

   # Download WMAP power spectrum data
   download_wmap_data(data_type='POWER_SPECTRUM', output_dir='wmap_data')

   # Download WMAP ILC map (optional, larger file)
   download_wmap_data(data_type='ILC_MAP', output_dir='wmap_data')

   # Download WMAP analysis mask (optional)
   download_wmap_data(data_type='ANALYSIS_MASK', output_dir='wmap_data')

Alternatively, you can use the command-line interface:

.. code-block:: bash

   python -m wmap_data.wmap_data_handler --data-type POWER_SPECTRUM --output-dir wmap_data

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **ImportError: No module named healpy**

   Healpy can be challenging to install on some systems due to its C dependencies. Try:

   .. code-block:: bash

      pip install --no-binary :all: healpy

   Or on Ubuntu/Debian:

   .. code-block:: bash

      sudo apt-get install libhealpix-cxx-dev
      pip install healpy

2. **RuntimeError: module compiled against API version XX but this version of numpy is YY**

   This indicates a version mismatch between the NumPy used to compile a module and the one installed. Try:

   .. code-block:: bash

      pip uninstall numpy
      pip install numpy==1.16.6
      pip install -r requirements.txt

3. **Timeout or hanging during analysis**

   The framework includes timeout mechanisms to prevent indefinite hanging. If you encounter timeouts:

   - Increase the timeout value: `--timeout-seconds 120`
   - Reduce the number of simulations: `--num-simulations 20`
   - Disable parallel processing: `--no-parallel`

4. **Memory errors during analysis**

   For large datasets, you might encounter memory errors. Try:

   - Reduce the data size: `--data-size 2048`
   - Process data in chunks by modifying the code
   - Increase your system's swap space

System-Specific Notes
~~~~~~~~~~~~~~~~~~~

**Windows:**

- You may need to install Microsoft Visual C++ Build Tools for some packages
- Use backslashes in file paths or raw strings: r"C:\path\to\data"

**macOS:**

- You might need to install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, ensure you're using Python packages compiled for ARM architecture

**Linux:**

- Install system dependencies for HEALPix: `sudo apt-get install libhealpix-cxx-dev`
- For headless servers, use the Agg backend for matplotlib: `matplotlib.use('Agg')`
