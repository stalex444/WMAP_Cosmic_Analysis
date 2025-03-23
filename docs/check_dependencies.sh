#!/bin/bash
# Script to check for Xcode Command Line Tools and install Sphinx dependencies

# Check if Xcode Command Line Tools are installed
if ! xcode-select -p &> /dev/null; then
    echo "Xcode Command Line Tools are not installed."
    echo "Please install them by running: xcode-select --install"
    echo "After installation completes, run this script again."
    exit 1
fi

echo "Xcode Command Line Tools are installed."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip first."
    exit 1
fi

echo "Installing Sphinx and required dependencies..."
pip install sphinx sphinx_rtd_theme

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
    echo "You can now build the documentation using: ./build_docs.sh"
else
    echo "Error installing dependencies. Please check the error messages above."
fi
