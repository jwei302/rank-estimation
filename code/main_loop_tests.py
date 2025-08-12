import torch
import itertools
import numpy as np
from scipy.linalg import det
import time
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import multiprocessing as mp
from functools import partial

def get_I_a_list(N):
    n = N//2
    I_list = list(itertools.combinations(range(1, N+1), 2))
    a_list = list(itertools.combinations(range(1, N+1), n))
    num_I = len(I_list)
    num_a = len(a_list)
    return I_list, a_list, num_I, num_a

# Original Levi-Civita sign function (from notebook)
def levi_civita_sign(big_set, sub_set):
    """
    Compute the sign of the permutation from sub_set + rest to big_set
    """
    perm = list(sub_set) + [x for x in big_set if x not in sub_set]
    idxs = [big_set.index(x) for x in perm]
    parity = np.array(idxs)
    sign = np.linalg.det(np.eye(len(parity))[parity])
    return int(round(sign))

# Optimized Levi-Civita sign function
def levi_civita_sign_optimized(big_set, sub_set):
    perm = list(sub_set) + [x for x in big_set if x not in sub_set]
    
    # Create a mapping from value to position
    value_to_pos = {val: pos for pos, val in enumerate(perm)}
    
    # Count cycles
    visited = [False] * len(perm)
    cycles = 0
    
    for i in range(len(perm)):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = value_to_pos[big_set[j]]
    
    # Sign is (-1)^(n - cycles) where n is the length
    return 1 if (len(perm) - cycles) % 2 == 0 else -1

# Removed vectorized Levi-Civita function as it was only used by inaccurate implementations

# Precompute lookup tables for faster set operations
def create_lookup_tables(I_list, a_list):
    """
    Create lookup tables to speed up set operations
    """
    # Precompute which I's are subsets of which a's
    I_subset_of_a = {}
    for i, I in enumerate(I_list):
        I_subset_of_a[i] = []
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                I_subset_of_a[i].append(a_idx)
    
    # Precompute Jp values (complements)
    Jp_values = {}
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                Jp = tuple(sorted(set(a) - set(I)))
                Jp_values[(i, a_idx)] = Jp
    
    return I_subset_of_a, Jp_values

# Original main loop implementation (from notebook)
def original_main_loop(I_list, a_list, psi):
    """
    Original main loop implementation from the notebook
    """
    num_I = len(I_list)
    num_a = len(a_list)
    
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    start_time = time.time()
    
    for i, I in enumerate(I_list):
        for ip, Ip in enumerate(I_list):
            for a_idx, a in enumerate(a_list):
                if set(I).issubset(a):
                    Jp = tuple(sorted(set(I) ^ set(a)))
                    for ap_idx, ap in enumerate(a_list):
                        if set(Ip).issubset(ap):
                            Jpp = tuple(sorted(set(Ip) ^ set(ap)))
                            if Jp == Jpp:
                                s1 = levi_civita_sign_optimized(a, I)
                                s2 = levi_civita_sign_optimized(ap, Ip)
                                J1[i,ip,a_idx] += s1 * s2 * psi[ap_idx]
                                J2[i,ip,ap_idx] += s1 * s2 * psi[a_idx]
    
    end_time = time.time()
    print(f"Original main loop time: {end_time - start_time:.4f} seconds")
    
    return J1, J2

# Original implementation for comparison (using optimized levi_civita)
def original_implementation(I_list, a_list, psi):
    num_I = len(I_list)
    num_a = len(a_list)
    
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    start_time = time.time()
    
    for i, I in enumerate(I_list):
        for ip, Ip in enumerate(I_list):
            for a_idx, a in enumerate(a_list):
                if set(I).issubset(a):
                    Jp = tuple(sorted(set(I) ^ set(a)))
                    for ap_idx, ap in enumerate(a_list):
                        if set(Ip).issubset(ap):
                            Jpp = tuple(sorted(set(Ip) ^ set(ap)))
                            if Jp == Jpp:
                                s1 = levi_civita_sign_optimized(a, I)
                                s2 = levi_civita_sign_optimized(ap, Ip)
                                J1[i,ip,a_idx] += s1 * s2 * psi[ap_idx]
                                J2[i,ip,ap_idx] += s1 * s2 * psi[a_idx]
    
    end_time = time.time()
    print(f"Original implementation time: {end_time - start_time:.4f} seconds")
    
    return J1, J2

# Optimized implementation 1: Use lookup tables
def optimized_implementation_1(I_list, a_list, psi):
    num_I = len(I_list)
    num_a = len(a_list)
    
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    # Create lookup tables
    I_subset_of_a, Jp_values = create_lookup_tables(I_list, a_list)
    
    start_time = time.time()
    
    for i, I in enumerate(I_list):
        for ip, Ip in enumerate(I_list):
            # Use precomputed subsets
            for a_idx in I_subset_of_a[i]:
                a = a_list[a_idx]
                Jp = Jp_values[(i, a_idx)]
                
                for ap_idx in I_subset_of_a[ip]:
                    ap = a_list[ap_idx]
                    Jpp = Jp_values[(ip, ap_idx)]
                    
                    if Jp == Jpp:
                        s1 = levi_civita_sign_optimized(a, I)
                        s2 = levi_civita_sign_optimized(ap, Ip)
                        J1[i,ip,a_idx] += s1 * s2 * psi[ap_idx]
                        J2[i,ip,ap_idx] += s1 * s2 * psi[a_idx]
    
    end_time = time.time()
    print(f"Optimized implementation 1 time: {end_time - start_time:.4f} seconds")
    
    return J1, J2

# Optimized implementation 2: Vectorized operations
def optimized_implementation_2(I_list, a_list, psi):
    num_I = len(I_list)
    num_a = len(a_list)
    
    # Convert to tensors for vectorized operations
    I_tensor = torch.tensor(I_list, dtype=torch.long)
    a_tensor = torch.tensor(a_list, dtype=torch.long)
    
    # Create sparse representation
    J1_data = []
    J1_indices = []
    J2_data = []
    J2_indices = []
    
    start_time = time.time()
    
    # Precompute all valid combinations
    valid_combinations = []
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                Jp = tuple(sorted(set(a) - set(I)))
                valid_combinations.append((i, a_idx, Jp))
    
    # Process valid combinations
    for i, a_idx, Jp in valid_combinations:
        for ip, ap_idx, Jpp in valid_combinations:
            if Jp == Jpp:
                a = a_list[a_idx]
                ap = a_list[ap_idx]
                I = I_list[i]
                Ip = I_list[ip]
                
                s1 = levi_civita_sign_optimized(a, I)
                s2 = levi_civita_sign_optimized(ap, Ip)
                
                J1_data.append(s1 * s2 * psi[ap_idx])
                J1_indices.append((i, ip, a_idx))
                
                J2_data.append(s1 * s2 * psi[a_idx])
                J2_indices.append((i, ip, ap_idx))
    
    # Convert to dense tensors
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    for (i, ip, a_idx), val in zip(J1_indices, J1_data):
        J1[i, ip, a_idx] = val
    
    for (i, ip, ap_idx), val in zip(J2_indices, J2_data):
        J2[i, ip, ap_idx] = val
    
    end_time = time.time()
    print(f"Optimized implementation 2 time: {end_time - start_time:.4f} seconds")
    
    return J1, J2

# Optimized implementation 3: Memory-efficient processing
def optimized_implementation_3(I_list, a_list, psi, chunk_size=1000):
    num_I = len(I_list)
    num_a = len(a_list)
    
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    # Create lookup tables
    I_subset_of_a, Jp_values = create_lookup_tables(I_list, a_list)
    
    start_time = time.time()
    
    # Process in chunks to reduce memory usage
    total_combinations = sum(len(I_subset_of_a[i]) for i in range(num_I))
    print(f"Total valid combinations: {total_combinations}")
    
    processed = 0
    for i in range(0, num_I, chunk_size):
        i_end = min(i + chunk_size, num_I)
        for ip in range(0, num_I, chunk_size):
            ip_end = min(ip + chunk_size, num_I)
            
            # Process this chunk
            for ii in range(i, i_end):
                for iip in range(ip, ip_end):
                    for a_idx in I_subset_of_a[ii]:
                        a = a_list[a_idx]
                        Jp = Jp_values[(ii, a_idx)]
                        
                        for ap_idx in I_subset_of_a[iip]:
                            ap = a_list[ap_idx]
                            Jpp = Jp_values[(iip, ap_idx)]
                            
                            if Jp == Jpp:
                                s1 = levi_civita_sign_optimized(a, I_list[ii])
                                s2 = levi_civita_sign_optimized(ap, I_list[iip])
                                J1[ii,iip,a_idx] += s1 * s2 * psi[ap_idx]
                                J2[ii,iip,ap_idx] += s1 * s2 * psi[a_idx]
                                processed += 1
    
    end_time = time.time()
    print(f"Optimized implementation 3 time: {end_time - start_time:.4f} seconds")
    print(f"Processed {processed} combinations")
    
    return J1, J2

# Removed vectorized implementation due to inaccuracy

# Removed ultra optimized implementation due to inaccuracy

# Multiprocessing implementation - bring back the discarded method
def process_chunk_multiprocessing(chunk_data):
    """Process a chunk of valid combinations in a separate process"""
    valid_combinations_chunk, valid_combinations_all, I_list, a_list, psi = chunk_data
    
    # Local results for this chunk
    J1_data = []
    J1_indices = []
    J2_data = []
    J2_indices = []
    
    # Process this chunk against ALL valid combinations
    for i, a_idx, Jp in valid_combinations_chunk:
        for ip, ap_idx, Jpp in valid_combinations_all:
            if Jp == Jpp:
                a = a_list[a_idx]
                ap = a_list[ap_idx]
                I = I_list[i]
                Ip = I_list[ip]
                
                s1 = levi_civita_sign_optimized(a, I)
                s2 = levi_civita_sign_optimized(ap, Ip)
                
                J1_data.append(s1 * s2 * psi[ap_idx])
                J1_indices.append((i, ip, a_idx))
                
                J2_data.append(s1 * s2 * psi[a_idx])
                J2_indices.append((i, ip, ap_idx))
    
    return J1_data, J1_indices, J2_data, J2_indices

def multiprocessing_implementation(I_list, a_list, psi, num_processes=None):
    """
    Multiprocessing implementation - parallelizes the valid combinations processing
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Limit to 8 to avoid overwhelming the system
    
    num_I = len(I_list)
    num_a = len(a_list)
    
    print(f"Using {num_processes} processes for multiprocessing implementation")
    
    start_time = time.time()
    
    # Precompute all valid combinations (same as opt2)
    valid_combinations = []
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                Jp = tuple(sorted(set(a) - set(I)))
                valid_combinations.append((i, a_idx, Jp))
    
    print(f"Found {len(valid_combinations)} valid combinations")
    
    # Split valid combinations into chunks for multiprocessing
    chunk_size = max(1, len(valid_combinations) // num_processes)
    chunks = [valid_combinations[i:i + chunk_size] 
              for i in range(0, len(valid_combinations), chunk_size)]
    
    print(f"Split into {len(chunks)} chunks")
    
    # Prepare data for each process
    chunk_data = [(chunk, valid_combinations, I_list, a_list, psi) for chunk in chunks]
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk_multiprocessing, chunk_data)
    
    # Combine results from all processes
    J1_data_all = []
    J1_indices_all = []
    J2_data_all = []
    J2_indices_all = []
    
    for J1_data, J1_indices, J2_data, J2_indices in results:
        J1_data_all.extend(J1_data)
        J1_indices_all.extend(J1_indices)
        J2_data_all.extend(J2_data)
        J2_indices_all.extend(J2_indices)
    
    # Convert to dense tensors
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    for (i, ip, a_idx), val in zip(J1_indices_all, J1_data_all):
        J1[i, ip, a_idx] = val
    
    for (i, ip, ap_idx), val in zip(J2_indices_all, J2_data_all):
        J2[i, ip, ap_idx] = val
    
    end_time = time.time()
    print(f"Multiprocessing implementation time: {end_time - start_time:.4f} seconds")
    print(f"Computed {len(J1_data_all)} non-zero elements")
    
    return J1, J2

# Test function to verify accuracy
def test_accuracy(N=6):
    print(f"Testing accuracy with N={N}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    # Run original implementation
    J1_orig, J2_orig = original_implementation(I_list, a_list, psi)
    
    # Run optimized implementations
    J1_opt1, J2_opt1 = optimized_implementation_1(I_list, a_list, psi)
    J1_opt2, J2_opt2 = optimized_implementation_2(I_list, a_list, psi)
    J1_opt3, J2_opt3 = optimized_implementation_3(I_list, a_list, psi)
    
    # Run multiprocessing implementation
    J1_mp, J2_mp = multiprocessing_implementation(I_list, a_list, psi)
    
    # Check accuracy
    print(f"Max difference opt1 vs original: {torch.max(torch.abs(J1_opt1 - J1_orig)):.2e}")
    print(f"Max difference opt2 vs original: {torch.max(torch.abs(J1_opt2 - J1_orig)):.2e}")
    print(f"Max difference opt3 vs original: {torch.max(torch.abs(J1_opt3 - J1_orig)):.2e}")
    print(f"Max difference multiprocessing vs original: {torch.max(torch.abs(J1_mp - J1_orig)):.2e}")
    
    # Check if all are close
    tolerance = 1e-10
    opt1_correct = torch.allclose(J1_opt1, J1_orig, atol=tolerance) and torch.allclose(J2_opt1, J2_orig, atol=tolerance)
    opt2_correct = torch.allclose(J1_opt2, J1_orig, atol=tolerance) and torch.allclose(J2_opt2, J2_orig, atol=tolerance)
    opt3_correct = torch.allclose(J1_opt3, J1_orig, atol=tolerance) and torch.allclose(J2_opt3, J2_orig, atol=tolerance)
    mp_correct = torch.allclose(J1_mp, J1_orig, atol=tolerance) and torch.allclose(J2_mp, J2_orig, atol=tolerance)
    
    print(f"Optimization 1 correct: {opt1_correct}")
    print(f"Optimization 2 correct: {opt2_correct}")
    print(f"Optimization 3 correct: {opt3_correct}")
    print(f"Multiprocessing correct: {mp_correct}")
    
    return opt1_correct and opt2_correct and opt3_correct and mp_correct

# Comprehensive benchmark function
def benchmark_all_implementations(N=8):
    print(f"Benchmarking all implementations with N={N}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    print(f"num_I={num_I}, num_a={num_a}")
    
    implementations = [
        ("Optimized 2 (Sparse)", optimized_implementation_2),
        ("Optimized 3 (Memory Efficient)", optimized_implementation_3),
        ("Multiprocessing", multiprocessing_implementation)
    ]
    
    results = {}
    
    for name, func in implementations:
        print(f"\nRunning {name}...")
        try:
            start_time = time.time()
            J1, J2 = func(I_list, a_list, psi)
            end_time = time.time()
            results[name] = {
                'time': end_time - start_time,
                'J1': J1,
                'J2': J2,
                'success': True
            }
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY for N={N}")
    print(f"{'='*60}")
    for name, result in results.items():
        if result['success']:
            print(f"{name:30}: {result['time']:.4f} seconds")
        else:
            error_msg = result.get('error', 'Skipped')
            print(f"{name:30}: {error_msg}")
    
    return results

# Special benchmark function for N=12 and N=14 without original loop
def benchmark_large_N(N_values=[12, 14]):
    """Benchmark implementations for larger N values, excluding slow original implementations"""
    
    for N in N_values:
        print(f"\n{'='*80}")
        print(f"BENCHMARKING N={N} (excluding slow original implementations)")
        print(f"{'='*80}")
        
        I_list, a_list, num_I, num_a = get_I_a_list(N)
        psi = torch.rand(num_a)
        psi /= psi.norm()
        
        print(f"num_I={num_I:,}, num_a={num_a:,}")
        print(f"Expected total operations: ~{num_I**2 * num_a:,}")
        
        # Only test fast implementations for large N
        implementations = [
            ("Optimized 2 (Sparse)", optimized_implementation_2),
            ("Optimized 3 (Memory Efficient)", optimized_implementation_3),
            ("Multiprocessing", multiprocessing_implementation)
        ]
        
        results = {}
        
        for name, func in implementations:
            print(f"\nRunning {name} for N={N}...")
            try:
                start_time = time.time()
                J1, J2 = func(I_list, a_list, psi)
                end_time = time.time()
                elapsed = end_time - start_time
                results[name] = {
                    'time': elapsed,
                    'success': True
                }
                print(f"{name} completed in {elapsed:.4f} seconds")
                
                # Check matrix properties
                print(f"  J1 shape: {J1.shape}, J2 shape: {J2.shape}")
                print(f"  J1 non-zero elements: {torch.count_nonzero(J1):,}")
                print(f"  J2 non-zero elements: {torch.count_nonzero(J2):,}")
                
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        # Print summary for this N
        print(f"\nSUMMARY for N={N}:")
        print("-" * 50)
        for name, result in results.items():
            if result['success']:
                print(f"{name:30}: {result['time']:.4f} seconds")
            else:
                error_msg = result.get('error', 'Failed')
                print(f"{name:30}: {error_msg}")

# Simple test for N=16 specifically
def test_N16_performance():
    """Test performance specifically for N=16"""
    N = 16
    print(f"\n{'='*80}")
    print(f"TESTING N={N} PERFORMANCE")
    print(f"{'='*80}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    print(f"num_I={num_I:,}, num_a={num_a:,}")
    print(f"Expected matrix size: {num_I**2 * num_a:,} elements")
    
    # Test only the fastest implementations to save time
    implementations = [
        ("Optimized 2 (Sparse)", optimized_implementation_2),
        ("Multiprocessing", multiprocessing_implementation)
    ]
    
    results = {}
    
    for name, func in implementations:
        print(f"\nRunning {name} for N={N}...")
        try:
            start_time = time.time()
            J1, J2 = func(I_list, a_list, psi)
            end_time = time.time()
            elapsed = end_time - start_time
            results[name] = {
                'time': elapsed,
                'success': True
            }
            print(f"{name} completed in {elapsed:.4f} seconds")
            
            # Check matrix properties
            print(f"  J1 shape: {J1.shape}, J2 shape: {J2.shape}")
            print(f"  J1 non-zero elements: {torch.count_nonzero(J1):,}")
            print(f"  J2 non-zero elements: {torch.count_nonzero(J2):,}")
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*50}")
    print(f"N={N} PERFORMANCE COMPARISON:")
    print(f"{'='*50}")
    for name, result in results.items():
        if result['success']:
            print(f"{name:30}: {result['time']:.4f} seconds")
        else:
            error_msg = result.get('error', 'Failed')
            print(f"{name:30}: {error_msg}")
    
    # Show speedup/slowdown
    if all(r['success'] for r in results.values()):
        opt2_time = results["Optimized 2 (Sparse)"]['time']
        mp_time = results["Multiprocessing"]['time']
        if mp_time < opt2_time:
            speedup = opt2_time / mp_time
            print(f"\nMultiprocessing is {speedup:.2f}x FASTER than Opt2!")
        else:
            slowdown = mp_time / opt2_time
            print(f"\nMultiprocessing is {slowdown:.2f}x SLOWER than Opt2")
    
    return results

if __name__ == "__main__":
    # Test accuracy first with a smaller N
    print("Testing accuracy...")
    accuracy_ok = test_accuracy(N=6)
    
    # if accuracy_ok:
    #     print("\nAll implementations pass accuracy test!")
    #     print("\nRunning benchmark for large N values...")
    #     benchmark_large_N([12, 14])
    # else:
    #     print("\nSome implementations failed accuracy test!")

    # Test N=16 performance specifically
    print("Testing N=16 performance...")
    test_N16_performance() 