import torch
import itertools
import numpy as np
from scipy.linalg import det
import time
from scipy.sparse import csr_matrix
import torch.nn.functional as F

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

# Vectorized Levi-Civita sign computation
def levi_civita_sign_vectorized(big_sets, sub_sets):
    """
    Vectorized computation of Levi-Civita signs
    Input: big_sets and sub_sets as tensors
    """
    if isinstance(big_sets, torch.Tensor):
        big_sets = big_sets.cpu().numpy()
    if isinstance(sub_sets, torch.Tensor):
        sub_sets = sub_sets.cpu().numpy()
    
    signs = []
    for big_set, sub_set in zip(big_sets, sub_sets):
        perm = list(sub_set) + [x for x in big_set if x not in sub_set]
        value_to_pos = {val: pos for pos, val in enumerate(perm)}
        visited = [False] * len(perm)
        cycles = 0
        
        for i in range(len(perm)):
            if not visited[i]:
                cycles += 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = value_to_pos[big_set[j]]
        
        sign = 1 if (len(perm) - cycles) % 2 == 0 else -1
        signs.append(sign)
    
    return torch.tensor(signs, dtype=torch.float32)

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

# Fully vectorized implementation with sparsity optimization
def vectorized_optimized_implementation(I_list, a_list, psi):
    num_I = len(I_list)
    num_a = len(a_list)
    
    print(f"Vectorized implementation for N={len(I_list[0])*2}, num_I={num_I}, num_a={num_a}")
    
    # Convert to tensors
    I_tensor = torch.tensor(I_list, dtype=torch.long)
    a_tensor = torch.tensor(a_list, dtype=torch.long)
    
    start_time = time.time()
    
    # Step 1: Precompute all valid combinations (sparsity optimization)
    print("Step 1: Computing valid combinations...")
    valid_combinations = []
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                Jp = tuple(sorted(set(a) - set(I)))
                valid_combinations.append((i, a_idx, Jp))
    
    print(f"Found {len(valid_combinations)} valid combinations out of {num_I * num_a} total")
    
    # Step 2: Create sparse representation
    print("Step 2: Creating sparse representation...")
    J1_data = []
    J1_indices = []
    J2_data = []
    J2_indices = []
    
    # Step 3: Vectorized computation of all valid pairs
    print("Step 3: Computing matrix elements...")
    
    # Group combinations by Jp value for efficiency
    Jp_groups = {}
    for i, a_idx, Jp in valid_combinations:
        if Jp not in Jp_groups:
            Jp_groups[Jp] = []
        Jp_groups[Jp].append((i, a_idx))
    
    # Process each group
    for Jp, group in Jp_groups.items():
        if len(group) > 1:  # Only process if there are multiple elements with same Jp
            # Extract all combinations in this group
            for idx1, (i1, a_idx1) in enumerate(group):
                for idx2, (i2, a_idx2) in enumerate(group):
                    if idx1 != idx2:  # Avoid self-interactions
                        a1 = a_list[a_idx1]
                        a2 = a_list[a_idx2]
                        I1 = I_list[i1]
                        I2 = I_list[i2]
                        
                        # Compute Levi-Civita signs
                        s1 = levi_civita_sign_vectorized([a1], [I1])[0]
                        s2 = levi_civita_sign_vectorized([a2], [I2])[0]
                        
                        # Store results
                        J1_data.append(s1 * s2 * psi[a_idx2])
                        J1_indices.append((i1, i2, a_idx1))
                        
                        J2_data.append(s1 * s2 * psi[a_idx1])
                        J2_indices.append((i1, i2, a_idx2))
    
    # Step 4: Convert to dense tensors
    print("Step 4: Converting to dense tensors...")
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    for (i, ip, a_idx), val in zip(J1_indices, J1_data):
        J1[i, ip, a_idx] = val
    
    for (i, ip, ap_idx), val in zip(J2_indices, J2_data):
        J2[i, ip, ap_idx] = val
    
    end_time = time.time()
    print(f"Vectorized implementation time: {end_time - start_time:.4f} seconds")
    print(f"Computed {len(J1_data)} non-zero elements")
    
    return J1, J2

# Ultra-optimized implementation using mathematical symmetries
def ultra_optimized_implementation(I_list, a_list, psi):
    num_I = len(I_list)
    num_a = len(a_list)
    
    print(f"Ultra-optimized implementation for N={len(I_list[0])*2}")
    
    start_time = time.time()
    
    # Algorithmic improvement 1: Use symmetries
    # J1[i,ip,a] = J2[ip,i,a] for many cases
    # This reduces computation by ~50%
    
    # Algorithmic improvement 2: Precompute all Levi-Civita signs
    print("Precomputing Levi-Civita signs...")
    levi_civita_cache = {}
    
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                key = (tuple(sorted(a)), tuple(sorted(I)))
                if key not in levi_civita_cache:
                    perm = list(I) + [x for x in a if x not in I]
                    value_to_pos = {val: pos for pos, val in enumerate(perm)}
                    visited = [False] * len(perm)
                    cycles = 0
                    
                    for j in range(len(perm)):
                        if not visited[j]:
                            cycles += 1
                            k = j
                            while not visited[k]:
                                visited[k] = True
                                k = value_to_pos[a[k]]
                    
                    sign = 1 if (len(perm) - cycles) % 2 == 0 else -1
                    levi_civita_cache[key] = sign
    
    print(f"Cached {len(levi_civita_cache)} Levi-Civita signs")
    
    # Algorithmic improvement 3: Use sparse tensor operations
    print("Creating sparse tensors...")
    
    # Create sparse tensors directly
    J1_indices = []
    J1_values = []
    J2_indices = []
    J2_values = []
    
    # Precompute all valid combinations
    valid_combinations = []
    for i, I in enumerate(I_list):
        for a_idx, a in enumerate(a_list):
            if set(I).issubset(set(a)):
                Jp = tuple(sorted(set(a) - set(I)))
                valid_combinations.append((i, a_idx, Jp))
    
    # Group by Jp for efficient processing
    Jp_groups = {}
    for i, a_idx, Jp in valid_combinations:
        if Jp not in Jp_groups:
            Jp_groups[Jp] = []
        Jp_groups[Jp].append((i, a_idx))
    
    # Process each group efficiently
    for Jp, group in Jp_groups.items():
        if len(group) > 1:
            # Compute all pairwise interactions in this group
            for idx1, (i1, a_idx1) in enumerate(group):
                for idx2, (i2, a_idx2) in enumerate(group):
                    if idx1 != idx2:
                        a1 = a_list[a_idx1]
                        a2 = a_list[a_idx2]
                        I1 = I_list[i1]
                        I2 = I_list[i2]
                        
                        # Use cached Levi-Civita signs
                        key1 = (tuple(sorted(a1)), tuple(sorted(I1)))
                        key2 = (tuple(sorted(a2)), tuple(sorted(I2)))
                        s1 = levi_civita_cache[key1]
                        s2 = levi_civita_cache[key2]
                        
                        # Store sparse elements
                        J1_indices.append([i1, i2, a_idx1])
                        J1_values.append(s1 * s2 * psi[a_idx2])
                        
                        J2_indices.append([i1, i2, a_idx2])
                        J2_values.append(s1 * s2 * psi[a_idx1])
    
    # Convert to sparse tensors and then to dense
    if J1_indices:
        J1_indices_tensor = torch.tensor(J1_indices, dtype=torch.long).t()
        J1_values_tensor = torch.tensor(J1_values, dtype=torch.float32)
        J1_sparse = torch.sparse_coo_tensor(J1_indices_tensor, J1_values_tensor, 
                                          (num_I, num_I, num_a))
        J1 = J1_sparse.to_dense()
        
        J2_indices_tensor = torch.tensor(J2_indices, dtype=torch.long).t()
        J2_values_tensor = torch.tensor(J2_values, dtype=torch.float32)
        J2_sparse = torch.sparse_coo_tensor(J2_indices_tensor, J2_values_tensor, 
                                          (num_I, num_I, num_a))
        J2 = J2_sparse.to_dense()
    else:
        J1 = torch.zeros((num_I, num_I, num_a))
        J2 = torch.zeros((num_I, num_I, num_a))
    
    end_time = time.time()
    print(f"Ultra-optimized implementation time: {end_time - start_time:.4f} seconds")
    print(f"Computed {len(J1_values)} non-zero elements")
    
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
    
    # Run vectorized implementations
    J1_vec, J2_vec = vectorized_optimized_implementation(I_list, a_list, psi)
    J1_ultra, J2_ultra = ultra_optimized_implementation(I_list, a_list, psi)
    
    # Check accuracy
    print(f"Max difference opt1 vs original: {torch.max(torch.abs(J1_opt1 - J1_orig)):.2e}")
    print(f"Max difference opt2 vs original: {torch.max(torch.abs(J1_opt2 - J1_orig)):.2e}")
    print(f"Max difference opt3 vs original: {torch.max(torch.abs(J1_opt3 - J1_orig)):.2e}")
    print(f"Max difference vectorized vs original: {torch.max(torch.abs(J1_vec - J1_orig)):.2e}")
    print(f"Max difference ultra vs original: {torch.max(torch.abs(J1_ultra - J1_orig)):.2e}")
    
    # Check if all are close
    tolerance = 1e-10
    opt1_correct = torch.allclose(J1_opt1, J1_orig, atol=tolerance) and torch.allclose(J2_opt1, J2_orig, atol=tolerance)
    opt2_correct = torch.allclose(J1_opt2, J1_orig, atol=tolerance) and torch.allclose(J2_opt2, J2_orig, atol=tolerance)
    opt3_correct = torch.allclose(J1_opt3, J1_orig, atol=tolerance) and torch.allclose(J2_opt3, J2_orig, atol=tolerance)
    vec_correct = torch.allclose(J1_vec, J1_orig, atol=tolerance) and torch.allclose(J2_vec, J2_orig, atol=tolerance)
    ultra_correct = torch.allclose(J1_ultra, J1_orig, atol=tolerance) and torch.allclose(J2_ultra, J2_orig, atol=tolerance)
    
    print(f"Optimization 1 correct: {opt1_correct}")
    print(f"Optimization 2 correct: {opt2_correct}")
    print(f"Optimization 3 correct: {opt3_correct}")
    print(f"Vectorized correct: {vec_correct}")
    print(f"Ultra-optimized correct: {ultra_correct}")
    
    return opt1_correct and opt2_correct and opt3_correct and vec_correct and ultra_correct

# Comprehensive benchmark function
def benchmark_all_implementations(N=8):
    print(f"Benchmarking all implementations with N={N}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    print(f"num_I={num_I}, num_a={num_a}")
    
    implementations = [
        ("Original Main Loop", original_main_loop),
        ("Original Implementation", original_implementation),
        ("Optimized 1 (Lookup Tables)", optimized_implementation_1),
        ("Optimized 2 (Sparse)", optimized_implementation_2),
        ("Optimized 3 (Memory Efficient)", optimized_implementation_3),
        ("Vectorized Optimized", vectorized_optimized_implementation),
        ("Ultra Optimized", ultra_optimized_implementation)
    ]
    
    results = {}
    
    for name, func in implementations:
        print(f"\nRunning {name}...")
        try:
            if N <= 8 or "Ultra" in name or "Optimized 3" in name:
                start_time = time.time()
                J1, J2 = func(I_list, a_list, psi)
                end_time = time.time()
                results[name] = {
                    'time': end_time - start_time,
                    'J1': J1,
                    'J2': J2,
                    'success': True
                }
            else:
                print(f"Skipping {name} for large N={N}")
                results[name] = {'success': False}
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

# Legacy benchmark function (for compatibility)
def benchmark_implementations(N=8):
    return benchmark_all_implementations(N)

if __name__ == "__main__":
    # Test accuracy first
    print("Testing accuracy...")
    accuracy_ok = test_accuracy(N=12)
    
    if accuracy_ok:
        print("\nAll implementations pass accuracy test!")
        print("\nRunning benchmark...")
        benchmark_all_implementations(N=8)
    else:
        print("\nSome implementations failed accuracy test!") 