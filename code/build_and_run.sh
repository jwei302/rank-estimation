#!/bin/bash

# Build and run script for C++ Rank Estimation Pipeline

set -e  # Exit on any error

echo "==================================="
echo "C++ Rank Estimation Pipeline Build"
echo "==================================="

# Check if required dependencies are available
echo "Checking dependencies..."

# Check if cmake is available
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake is required but not installed."
    echo "Please install cmake: brew install cmake (on macOS) or apt-get install cmake (on Ubuntu)"
    exit 1
fi

# Check if a C++ compiler is available
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "ERROR: No C++ compiler found (g++ or clang++)."
    echo "Please install a C++ compiler."
    exit 1
fi

# Check if Eigen3 is available (try pkg-config first)
if ! pkg-config --exists eigen3 2>/dev/null; then
    echo "WARNING: Eigen3 not found via pkg-config."
    echo "The build system will try to find it via CMake or you may need to install it:"
    echo "  macOS: brew install eigen"
    echo "  Ubuntu: sudo apt-get install libeigen3-dev"
fi

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building the project..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ ! -f "rank_pipeline" ]; then
    echo "ERROR: Build failed - executable not found."
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

# Run the pipeline
echo "====================================="
echo "Running C++ Rank Estimation Pipeline"
echo "====================================="
echo ""

# Record start time
start_time=$(date)
echo "Started at: $start_time"
echo ""

# Run the executable
./rank_pipeline

echo ""
echo "====================================="
echo "Pipeline Completed"
echo "====================================="
echo "Started at: $start_time"
echo "Finished at: $(date)"

# Check if results file was created
if [ -f "rank_results.json" ]; then
    echo ""
    echo "Results saved to: $(pwd)/rank_results.json"
    echo "File size: $(du -h rank_results.json | cut -f1)"
    echo ""
    echo "Sample of results:"
    head -n 20 rank_results.json
    echo "..."
    echo "(Full results in rank_results.json)"
else
    echo "WARNING: Results file rank_results.json not found."
fi

cd .. 