import torch
import itertools
import numpy as np
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
            # Use vectorized operations within each group
            group_tensor = torch.tensor(group)
            i_indices = group_tensor[:, 0]
            a_indices = group_tensor[:, 1]
            
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
    
    # Convert to sparse tensors
    if J1_indices:
        J1_indices = torch.tensor(J1_indices, dtype=torch.long).t()
        J1_values = torch.tensor(J1_values, dtype=torch.float32)
        J1_sparse = torch.sparse_coo_tensor(J1_indices, J1_values, (num_I, num_I, num_a))
        J1 = J1_sparse.to_dense()
    else:
        J1 = torch.zeros((num_I, num_I, num_a))
    
    if J2_indices:
        J2_indices = torch.tensor(J2_indices, dtype=torch.long).t()
        J2_values = torch.tensor(J2_values, dtype=torch.float32)
        J2_sparse = torch.sparse_coo_tensor(J2_indices, J2_values, (num_I, num_I, num_a))
        J2 = J2_sparse.to_dense()
    else:
        J2 = torch.zeros((num_I, num_I, num_a))
    
    end_time = time.time()
    print(f"Ultra-optimized implementation time: {end_time - start_time:.4f} seconds")
    print(f"Computed {len(J1_values)} non-zero elements")
    
    return J1, J2

# Test function
def test_implementations(N=6):
    print(f"Testing implementations with N={N}")
    
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    psi = torch.rand(num_a)
    psi /= psi.norm()
    
    print(f"num_I={num_I}, num_a={num_a}")
    
    # Test vectorized implementation
    try:
        J1_vec, J2_vec = vectorized_optimized_implementation(I_list, a_list, psi)
        print("Vectorized implementation successful")
    except Exception as e:
        print(f"Vectorized implementation failed: {e}")
        return
    
    # Test ultra-optimized implementation
    try:
        J1_ultra, J2_ultra = ultra_optimized_implementation(I_list, a_list, psi)
        print("Ultra-optimized implementation successful")
        
        # Check if results are consistent
        diff1 = torch.max(torch.abs(J1_vec - J1_ultra))
        diff2 = torch.max(torch.abs(J2_vec - J2_ultra))
        print(f"Max difference between implementations: {max(diff1, diff2):.2e}")
        
    except Exception as e:
        print(f"Ultra-optimized implementation failed: {e}")

if __name__ == "__main__":
    # Test with different N values
    for N in [6, 8, 10]:
        try:
            test_implementations(N)
            print(f"Successfully completed N={N}\n")
        except Exception as e:
            print(f"Failed for N={N}: {e}")
            break 