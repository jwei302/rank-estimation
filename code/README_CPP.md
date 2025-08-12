# C++ Rank Estimation Pipeline

This is a high-performance C++ implementation of the rank estimation pipeline, converted from the original Python code. It includes parallelization, optimized algorithms, and efficient memory management.

## Features

- **Parallel Processing**: Multi-threaded computation using all available CPU cores
- **Optimized Algorithms**: 
  - Optimized Levi-Civita sign computation using cycle counting
  - Lookup table optimization for subset operations
  - Efficient sparse-to-dense matrix conversion
- **Memory Efficient**: Smart memory management with shared pointers
- **High Performance**: Native C++ with Eigen for linear algebra operations
- **JSON Output**: Structured results in JSON format for easy analysis

## Dependencies

### Required
- **CMake** (≥ 3.12): Build system
- **C++ Compiler**: g++ or clang++ with C++17 support
- **Eigen3**: Linear algebra library for matrix operations and SVD
- **nlohmann/json**: JSON library (automatically downloaded if not found)

### Installation

#### macOS (using Homebrew)
```bash
brew install cmake eigen
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake libeigen3-dev build-essential
```

#### Other Systems
- Eigen3: Download from https://eigen.tuxfamily.org/
- CMake: Download from https://cmake.org/

## Building and Running

### Quick Start
```bash
./build_and_run.sh
```

This script will:
1. Check dependencies
2. Configure the build with CMake
3. Compile the optimized executable
4. Run the pipeline for N=4 to N=20 (steps of 2)
5. Save results to `rank_results.json`

### Manual Build
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc)

# Run
./rank_pipeline
```

## Algorithm Details

### Core Functionality
1. **J Tensor Computation**: Generates J1 and J2 tensors using optimized implementation 2
2. **Jacobian Matrix**: Combines J1 and J2 into full Jacobian matrix
3. **Rank Calculation**: Uses SVD with threshold (1e-5) to determine matrix rank
4. **Parallel Processing**: Distributes computation across multiple threads

### Optimizations

#### Levi-Civita Sign Optimization
- Uses cycle counting algorithm instead of determinant computation
- O(n) complexity instead of O(n!) for permutation generation
- Significant speedup for larger N values

#### Parallel Processing Strategy
- Splits valid combinations into chunks for each thread
- Uses local matrices to minimize lock contention
- Combines results efficiently at the end

#### Memory Management
- Smart pointers for automatic memory management
- Efficient matrix operations using Eigen
- Minimal memory allocations in hot loops

## Output Format

Results are saved to `rank_results.json` with the following structure:

```json
{
    "metadata": {
        "description": "Rank estimation pipeline results",
        "algorithm": "optimized_implementation_2_parallel",
        "levi_civita": "optimized_cycle_counting",
        "threshold": 1e-05,
        "total_computation_time_seconds": 123
    },
    "results": [
        {
            "N": 4,
            "jacobian_shape": [36, 12],
            "matrix_rank": 12,
            "singular_values": [...],
            "computation_time_seconds": 0.045,
            "num_I": 6,
            "num_a": 6,
            "psi_state": "random"
        },
        ...
    ]
}
```

## Performance Characteristics

### Expected Performance (compared to Python)
- **5-20x faster** for larger N values (N ≥ 12)
- **2-10x faster** for core computation
- **Significantly reduced memory usage**
- **Linear scaling** with number of CPU cores

### Computational Complexity
- **Time**: O(num_I² × num_a²) for main computation
- **Space**: O(num_I² × num_a) for matrices
- **Parallelization**: Nearly linear speedup with thread count

### System Requirements
- **RAM**: ~1GB for N=20, scales as O(N⁶)
- **CPU**: Benefits from multiple cores (4+ recommended)
- **Storage**: ~10MB for JSON output

## Comparison with Python Implementation

| Aspect | Python | C++ |
|--------|---------|-----|
| Development Time | Days | Days (already implemented) |
| Performance | Baseline | 5-20x faster |
| Memory Usage | High | Optimized |
| Parallelization | Limited | Full multi-threading |
| Dependencies | PyTorch, NumPy | Eigen3, STL |
| Debugging | Easy | Moderate |
| Maintenance | Easy | Moderate |

## Troubleshooting

### Common Issues

1. **CMake not found**
   ```bash
   # macOS
   brew install cmake
   # Ubuntu
   sudo apt-get install cmake
   ```

2. **Eigen3 not found**
   ```bash
   # macOS
   brew install eigen
   # Ubuntu
   sudo apt-get install libeigen3-dev
   ```

3. **Compiler errors**
   - Ensure C++17 support: `g++ --version` should be ≥ 7.0
   - Update compiler if needed

4. **Memory issues for large N**
   - Monitor memory usage: `top` or `htop`
   - Consider reducing N range or using a machine with more RAM

### Performance Tuning

1. **Compiler Optimization**
   - Release build uses `-O3 -march=native`
   - For specific CPU: add `-mtune=native`

2. **Thread Count**
   - Automatically uses all available cores
   - Modify `thread::hardware_concurrency()` in code if needed

3. **Memory Optimization**
   - For memory-constrained systems, reduce chunk size in parallel processing
   - Consider chunked processing for very large N

## Files

- `rank_pipeline.cpp`: Main C++ implementation
- `CMakeLists.txt`: Build configuration
- `build_and_run.sh`: Automated build and run script
- `README_CPP.md`: This documentation
- `build/`: Build directory (created automatically)
- `rank_results.json`: Output file (created after running)

## License

Same as the original Python implementation. 