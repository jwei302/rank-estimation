import torch
import itertools
import numpy as np
from scipy.linalg import det
import time
import matplotlib.pyplot as plt
import argparse
import threading
import signal
import sys
from datetime import datetime, timedelta
import os


def get_I_a_list(N):
    """Generate I_list and a_list for given N"""
    n = N//2
    I_list = list(itertools.combinations(range(1, N+1), 2))
    a_list = list(itertools.combinations(range(1, N+1), n))
    num_I = len(I_list)
    num_a = len(a_list)
    return I_list, a_list, num_I, num_a


def create_images_folder(N, psi_state_name):
    """Create organized folder structure for saving images"""
    base_folder = "images"
    subfolder = f"N_{N}_{psi_state_name}"
    full_path = os.path.join(base_folder, subfolder)
    
    # Create the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


# Define psi states (only the two you requested)
def psi_product_state(num_a):
    """Product state: maximally unentangled"""
    psi = torch.zeros(num_a)
    psi[0] = 1.
    return psi


def psi_random(num_a):
    """Random state: moderate entanglement"""
    psi = 2 * torch.rand(num_a) - 1.
    psi /= psi.norm()
    return psi


# Global variables for progress tracking
progress_data = {
    'current': 0,
    'total': 0,
    'start_time': 0,
    'stop_flag': False,
    'last_update_time': 0,
    'update_interval': 30  # seconds
}


def progress_monitor():
    """Monitor progress and print updates every 30 seconds"""
    while not progress_data['stop_flag']:
        current_time = time.time()
        if (current_time - progress_data['last_update_time']) >= progress_data['update_interval']:
            if progress_data['total'] > 0 and progress_data['current'] > 0:
                elapsed = current_time - progress_data['start_time']
                progress = progress_data['current'] / progress_data['total']
                eta_seconds = (elapsed / progress) - elapsed if progress > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {progress_data['current']:,}/{progress_data['total']:,} "
                      f"({progress*100:.1f}%) | Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta}")
                
            progress_data['last_update_time'] = current_time
        time.sleep(5)  # Check every 5 seconds


def start_progress_monitor():
    """Start the progress monitoring thread"""
    progress_data['stop_flag'] = False
    progress_data['last_update_time'] = time.time()
    monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


def stop_progress_monitor():
    """Stop the progress monitoring thread"""
    progress_data['stop_flag'] = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nInterrupted by user. Cleaning up...')
    stop_progress_monitor()
    sys.exit(0)


def levi_civita_sign(big_set, sub_set):
    """
    Compute the sign of the permutation from sub_set + rest to big_set
    """
    perm = list(sub_set) + [x for x in big_set if x not in sub_set]
    idxs = [big_set.index(x) for x in perm]
    parity = np.array(idxs)
    sign = np.linalg.det(np.eye(len(parity))[parity])
    return int(round(sign))


def levi_civita_sign_optimized(big_set, sub_set):
    """Optimized Levi-Civita sign computation using cycle counting"""
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


def optimized_implementation_2(I_list, a_list, psi, verbose=True):
    """
    Optimized implementation using sparse representation with progress tracking
    """
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
    
    if verbose:
        print(f"Found {len(valid_combinations)} valid combinations")
        print(f"Total operations to process: {len(valid_combinations)**2:,}")
    
    # Setup progress tracking
    progress_data['current'] = 0
    progress_data['total'] = len(valid_combinations)**2
    progress_data['start_time'] = start_time
    start_progress_monitor()
    
    # Process valid combinations
    processed = 0
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
            
            processed += 1
            progress_data['current'] = processed
    
    stop_progress_monitor()
    
    # Convert to dense tensors
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    for (i, ip, a_idx), val in zip(J1_indices, J1_data):
        J1[i, ip, a_idx] = val
    
    for (i, ip, ap_idx), val in zip(J2_indices, J2_data):
        J2[i, ip, ap_idx] = val
    
    end_time = time.time()
    if verbose:
        print(f"\nOptimized implementation 2 time: {end_time - start_time:.4f} seconds")
    
    return J1, J2


def optimized_implementation_3(I_list, a_list, psi, chunk_size=1000, verbose=True):
    """
    Memory-efficient processing using chunks with progress tracking
    """
    num_I = len(I_list)
    num_a = len(a_list)
    
    J1 = torch.zeros((num_I, num_I, num_a))
    J2 = torch.zeros((num_I, num_I, num_a))
    
    # Create lookup tables
    I_subset_of_a, Jp_values = create_lookup_tables(I_list, a_list)
    
    start_time = time.time()
    
    # Calculate total operations for progress tracking
    total_combinations = sum(len(I_subset_of_a[i]) for i in range(num_I))
    total_operations = total_combinations ** 2
    
    if verbose:
        print(f"Total valid combinations: {total_combinations:,}")
        print(f"Total operations to process: {total_operations:,}")
    
    # Setup progress tracking
    progress_data['current'] = 0
    progress_data['total'] = total_operations
    progress_data['start_time'] = start_time
    start_progress_monitor()
    
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
                            progress_data['current'] = processed
    
    stop_progress_monitor()
    
    end_time = time.time()
    if verbose:
        print(f"\nOptimized implementation 3 time: {end_time - start_time:.4f} seconds")
        print(f"Processed {processed:,} combinations")
    
    return J1, J2


def plot_spectrum(S, threshold=1e-5, title="Spectrum of Jacobian Matrix", 
                 N=None, psi_state_name=None, implementation=None):
    """
    Plot the spectrum of singular values and return the rank
    """
    # Count number of singular values greater than threshold
    count_above_threshold = np.sum(S > threshold)
    
    plt.figure(figsize=(8, 6))
    x_vals = np.arange(1, 1 + len(S))
    plt.semilogy(x_vals, S, 'o-', markersize=4, linewidth=1.5)
    plt.xlabel('Singular value index', fontsize=12)
    plt.ylabel('Singular value (log scale)', fontsize=12)
    plt.title(title, fontsize=14)

    # Show at most 10 ticks
    tick_count = min(len(S), 10)
    tick_locs = np.linspace(1, len(S), num=tick_count, dtype=int)
    plt.xticks(tick_locs)

    # Add text in bottom-left with more info
    info_text = f'Rank: {count_above_threshold}'
    if N is not None:
        info_text += f'\nN: {N}'
    if psi_state_name is not None:
        info_text += f'\nPsi: {psi_state_name}'
    if implementation is not None:
        info_text += f'\nImpl: {implementation}'
    
    plt.text(0.05, 0.05, info_text,
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8))

    # Add horizontal dashed line at threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
               label=f'Threshold: {threshold}')

    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    if N is not None and psi_state_name is not None:
        folder_path = create_images_folder(N, psi_state_name)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        impl_suffix = f"_{implementation}" if implementation else ""
        filename = f"spectrum_N{N}_{psi_state_name}{impl_suffix}_rank{count_above_threshold}_{timestamp}.png"
        filepath = os.path.join(folder_path, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved to: {filepath}")
    
    plt.show()
    
    return count_above_threshold


def compute_jacobian_matrix(J1, J2, num_I, num_a):
    """
    Combine J1 and J2 into the full Jacobian matrix
    """
    J1_reshaped = J1.reshape(num_I**2, num_a)
    J2_reshaped = J2.reshape(num_I**2, num_a)
    J = torch.cat((J1_reshaped, J2_reshaped), dim=1)
    return J


def compute_rank(J, threshold=1e-5):
    """
    Compute the rank of matrix J using SVD
    """
    s = torch.linalg.svdvals(J)
    rank = np.array(s)
    return rank, s


def run_pipeline(N=8, psi_state_func=psi_random, implementation='opt2'):
    """
    Run the complete pipeline for given parameters
    
    Args:
        N: System size
        psi_state_func: Function to generate psi state
        implementation: 'opt2' or 'opt3'
        
    Returns:
        dict with results: rank, singular_values, J_matrix, timing
    """
    threshold = 1e-5
    
    print(f"Running pipeline with N={N}, implementation={implementation}")
    print(f"Psi state: {psi_state_func.__name__}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    start_total = time.time()
    I_list, a_list, num_I, num_a = get_I_a_list(N)
    
    print(f"num_I: {num_I:,}, num_a: {num_a:,}")
    print(f"Expected memory usage: ~{(num_I**2 * num_a * 8 / 1e9):.1f} GB")
    
    # Generate psi state
    psi = psi_state_func(num_a)
    
    # Compute J1, J2
    if implementation == 'opt2':
        J1, J2 = optimized_implementation_2(I_list, a_list, psi, verbose=True)
    elif implementation == 'opt3':
        J1, J2 = optimized_implementation_3(I_list, a_list, psi, verbose=True)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    # Compute Jacobian matrix
    print("Computing Jacobian matrix...")
    J = compute_jacobian_matrix(J1, J2, num_I, num_a)
    
    print(f"Jacobian matrix shape: {J.shape}")
    print("Computing SVD...")
    
    # Compute rank and singular values
    rank, s = compute_rank(J, threshold=threshold)
    
    # Plot spectrum and save
    plot_spectrum(np.array(s), threshold=threshold, 
                 N=N, psi_state_name=psi_state_func.__name__, 
                 implementation=implementation)
    
    end_total = time.time()
    
    print(f"Total pipeline time: {timedelta(seconds=int(end_total - start_total))}")
    print(f"Matrix rank: {rank}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'rank': rank,
        'singular_values': s,
        'jacobian_matrix': J,
        'total_time': end_total - start_total,
        'N': N,
        'num_I': num_I,
        'num_a': num_a,
        'psi_state': psi_state_func.__name__
    }


def get_psi_function_by_name(name):
    """Get psi function by string name"""
    psi_functions = {
        'product': psi_product_state,
        'random': psi_random
    }
    
    if name not in psi_functions:
        available = list(psi_functions.keys())
        raise ValueError(f"Unknown psi state '{name}'. Available: {available}")
    
    return psi_functions[name]


def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run rank estimation pipeline')
    parser.add_argument('--N', type=int, default=8, help='System size (default: 8)')
    parser.add_argument('--psi', type=str, default='random', 
                        choices=['product', 'random'],
                        help='Psi state function (default: random)')
    parser.add_argument('--impl', type=str, default='opt2', 
                        choices=['opt2', 'opt3'],
                        help='Implementation to use (default: opt2)')
    
    args = parser.parse_args()
    
    try:
        psi_func = get_psi_function_by_name(args.psi)
        result = run_pipeline(
            N=args.N,
            psi_state_func=psi_func,
            implementation=args.impl
        )
        
        print(f"\nFinal Result: Rank = {result['rank']}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
