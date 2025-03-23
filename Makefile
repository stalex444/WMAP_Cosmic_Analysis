# Makefile for WMAP Cosmic Analysis Framework

.PHONY: all clean test docs install lint

# Default target
all: install test docs

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python -m unittest discover -s tests

# Build documentation
docs:
	cd docs && make html

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/build/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Run linting
lint:
	flake8 .

# Run analysis with default configuration
run:
	python run_wmap_analysis.py

# Run analysis with visualization
run-viz:
	python run_wmap_analysis.py --visualize

# Run optimized analysis with early stopping
run-optimized:
	python run_wmap_analysis.py --timeout-seconds 60 --num-simulations 30 --early-stopping

# Generate test data
generate-test-data:
	python utils/generate_test_data.py

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Install dependencies, run tests, and build docs"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run tests"
	@echo "  docs           - Build documentation"
	@echo "  clean          - Clean build artifacts"
	@echo "  lint           - Run linting"
	@echo "  run            - Run analysis with default configuration"
	@echo "  run-viz        - Run analysis with visualization"
	@echo "  run-optimized  - Run analysis with optimized parameters"
	@echo "  generate-test-data - Generate test data"
