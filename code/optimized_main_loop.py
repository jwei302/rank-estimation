import torch
import itertools
import numpy as np
from scipy.linalg import det
import time

def get_I_a_list(N):
    n = N//2
    I_list = list(itertools.combinations(range(1, N+1), 2))
    a_list = list(itertools.combinations(range(1, N+1), n))
    num_I = len(I_list)
    num_a = len(a_list)
    return I_list, a_list, num_I, num_a

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

# Original implementation for comparison
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
    
    # Check accuracy
    print(f"Max difference opt1 vs original: {torch.max(torch.abs(J1_opt1 - J1_orig)):.2e}")
    print(f"Max difference opt2 vs original: {torch.max(torch.abs(J1_opt2 - J1_orig)):.2e}")
    print(f"Max difference opt3 vs original: {torch.max(torch.abs(J1_opt3 - J1_orig)):.2e}")
    
    # Check if all are close
    tolerance = 1e-10
    opt1_correct = torch.allclose(J1_opt1, J1_orig, atol=tolerance) and torch.allclose(J2_opt1, J2_orig, atol=tolerance)
    opt2_correct = torch.allclose(J1_opt2, J1_orig, atol=tolerance) and torch.allclose(J2_opt2, J2_orig, atol=tolerance)
    opt3_correct = torch.allclose(J1_opt3, J1_orig, atol=tolerance) and torch.allclose(J2_opt3, J2_orig, atol=tolerance)
    
    print(f"Optimization 1 correct: {opt1_correct}")
    print(f"Optimization 2 correct: {opt2_correct}")
    print(f"Optimization 3 correct: {opt3_correct}")
    
    return opt1_correct and opt2_correct and opt3_correct

# Benchmark function
def benchmark_implementations(N=8):
    print(f"Benchmarking with N={N}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    print(f"num_I={num_I}, num_a={num_a}")
    
    # Only run optimized versions for larger N
    if N <= 8:
        J1_opt1, J2_opt1 = optimized_implementation_1(I_list, a_list, psi)
        J1_opt2, J2_opt2 = optimized_implementation_2(I_list, a_list, psi)
        J1_opt3, J2_opt3 = optimized_implementation_3(I_list, a_list, psi)
    else:
        print("Running only optimized implementation 3 for larger N")
        J1_opt3, J2_opt3 = optimized_implementation_3(I_list, a_list, psi)
    
    return J1_opt3, J2_opt3

if __name__ == "__main__":
    # Test accuracy first
    print("=== ACCURACY TEST ===")
    test_accuracy(N=6)
    
    print("\n=== BENCHMARK ===")
    # Benchmark with different N values
    for N in [6, 8, 10]:
        try:
            J1, J2 = benchmark_implementations(N)
            print(f"Successfully completed N={N}")
        except Exception as e:
            print(f"Failed for N={N}: {e}")
            break 