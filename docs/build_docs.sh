#!/bin/bash
# Script to build the Sphinx documentation

# Ensure we're in the docs directory
cd "$(dirname "$0")"

# Create the source/_static directory if it doesn't exist
mkdir -p source/_static

# Install Sphinx and the Read the Docs theme if not already installed
pip install sphinx sphinx_rtd_theme

# Build the HTML documentation
echo "Building HTML documentation..."
make html

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "You can view the documentation by opening: $(pwd)/build/html/index.html"
    
    # Open the documentation in the default browser (macOS)
    open build/html/index.html
else
    echo "Error building documentation. Please check the error messages above."
fi
