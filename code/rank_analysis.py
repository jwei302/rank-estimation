import torch
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading compatibility
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import argparse
from main_loop_tests import get_I_a_list, optimized_implementation_2

def create_psi_state(psi_type, num_a):
    """Create different types of psi states"""
    if psi_type == "product":
        # Product state - all coefficients equal
        psi = torch.ones(num_a)
        psi /= psi.norm()
        return psi
    elif psi_type == "random":
        # Random state
        psi = torch.rand(num_a)
        psi /= psi.norm()
        return psi
    else:
        raise ValueError(f"Unknown psi_type: {psi_type}")

def estimate_rank_from_spectrum(singular_values, threshold=1e-5):
    """Estimate rank from singular values using threshold"""
    return torch.sum(singular_values > threshold).item()

def plot_spectrum(S, threshold=1e-5, title_suffix=""):
    """Original plot_spectrum function from the notebook"""
    plt.figure(figsize=(6, 4))
    x_vals = np.arange(1, 1 + len(S))
    plt.semilogy(x_vals, S, 'o-')
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value (log scale)')
    plt.title(f'Spectrum of Jacobian Matrix{title_suffix}')

    # Show at most 10 ticks
    tick_count = min(len(S), 10)
    tick_locs = np.linspace(1, len(S), num=tick_count, dtype=int)
    plt.xticks(tick_locs)

    # Count number of singular values greater than threshold
    count_above_threshold = np.sum(S > threshold)

    # Add text in bottom-left
    plt.text(0.05, 0.05, f'rank: {count_above_threshold}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')

    # Add horizontal dashed line at threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1)

    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    return count_above_threshold

def analyze_single_N(N, psi_type, threshold=1e-5):
    """Analyze a single N value and return results"""
    print(f"Analyzing N={N}, psi_type={psi_type}")
    
    try:
        # Get combinatorial data
        I_list, a_list, num_I, num_a = get_I_a_list(N)
        
        # Create psi state
        psi = create_psi_state(psi_type, num_a)
        
        # Run optimization
        start_time = time.time()
        J1, J2 = optimized_implementation_2(I_list, a_list, psi)
        computation_time = time.time() - start_time
        
        # Reshape and concatenate exactly like in the notebook
        J1_reshaped = J1.reshape(num_I**2, num_a)
        J2_reshaped = J2.reshape(num_I**2, num_a)
        J = torch.cat((J1_reshaped, J2_reshaped), dim=1)
        
        # Print the torch.Size() as requested
        print(f"J.shape: {J.shape}")
        
        # Compute SVD values
        s = torch.linalg.svdvals(J)
        
        # Use the original plot_spectrum function
        title_suffix = f" (N={N}, ψ={psi_type})"
        rank = plot_spectrum(np.array(s.detach()), threshold, title_suffix)
        
        # Save the plot
        os.makedirs('images', exist_ok=True)
        filename = f"spectrum_{N}_{psi_type}.png"
        filepath = os.path.join('images', filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'N': N,
            'psi_type': psi_type,
            'rank': rank,
            'computation_time': computation_time,
            'num_I': num_I,
            'num_a': num_a,
            'J_shape': J.shape,
            'singular_values': s.detach().numpy(),
            'filename': filename,
            'success': True
        }
        
    except Exception as e:
        print(f"Error analyzing N={N}, psi_type={psi_type}: {e}")
        return {
            'N': N,
            'psi_type': psi_type,
            'success': False,
            'error': str(e)
        }

def run_sequential_analysis(N_values, psi_types, threshold=1e-5):
    """Run analysis sequentially to avoid matplotlib threading issues"""
    results = []
    total_tasks = len(N_values) * len(psi_types)
    current_task = 0
    
    print(f"Running {total_tasks} analyses sequentially...")
    
    for N in N_values:
        for psi_type in psi_types:
            current_task += 1
            print(f"Progress: {current_task}/{total_tasks}")
            result = analyze_single_N(N, psi_type, threshold)
            results.append(result)
    
    return results

def create_summary_plot(results, psi_types):
    """Create summary plots showing rank vs N for different psi types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, psi_type in enumerate(psi_types):
        # Filter results for this psi_type
        psi_results = [r for r in results if r['success'] and r['psi_type'] == psi_type]
        if not psi_results:
            continue
            
        N_vals = [r['N'] for r in psi_results]
        ranks = [r['rank'] for r in psi_results]
        times = [r['computation_time'] for r in psi_results]
        
        color = colors[i % len(colors)]
        
        # Plot rank vs N
        ax1.plot(N_vals, ranks, 'o-', color=color, label=f'ψ = {psi_type}', markersize=6)
        
        # Plot computation time vs N
        ax2.plot(N_vals, times, 's-', color=color, label=f'ψ = {psi_type}', markersize=6)
    
    ax1.set_xlabel('System Size (N)')
    ax1.set_ylabel('Estimated Rank')
    ax1.set_title('Rank vs System Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('System Size (N)')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Computation Time vs System Size')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = 'images/rank_analysis_summary.png'
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {summary_filename}")

def save_results_csv(results):
    """Save results to CSV file"""
    import csv
    
    csv_filename = 'images/rank_analysis_results.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['N', 'psi_type', 'rank', 'computation_time', 'num_I', 'num_a', 'J_shape', 'filename', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if result['success']:
                writer.writerow({
                    'N': result['N'],
                    'psi_type': result['psi_type'],
                    'rank': result['rank'],
                    'computation_time': result['computation_time'],
                    'num_I': result['num_I'],
                    'num_a': result['num_a'],
                    'J_shape': str(result['J_shape']),
                    'filename': result['filename'],
                    'success': result['success']
                })
    
    print(f"Results saved to: {csv_filename}")

def print_results_summary(results):
    """Print a summary of all results"""
    print("\n" + "="*80)
    print("RANK ANALYSIS SUMMARY")
    print("="*80)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Total analyses: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if failed_results:
        print("\nFailed analyses:")
        for result in failed_results:
            print(f"  N={result['N']}, psi={result['psi_type']}: {result['error']}")
    
    if successful_results:
        print(f"\n{'N':>3} {'ψ':>10} {'Rank':>6} {'Time(s)':>8} {'num_I':>8} {'num_a':>8}")
        print("-" * 50)
        for result in sorted(successful_results, key=lambda x: (x['N'], x['psi_type'])):
            print(f"{result['N']:>3} {result['psi_type']:>10} {result['rank']:>6} "
                  f"{result['computation_time']:>8.2f} {result['num_I']:>8} {result['num_a']:>8}")

def main():
    parser = argparse.ArgumentParser(description='Run rank estimation analysis across multiple N values')
    parser.add_argument('--N-start', type=int, default=4, help='Starting N value (default: 4)')
    parser.add_argument('--N-end', type=int, default=20, help='Ending N value (default: 20)')
    parser.add_argument('--N-step', type=int, default=2, help='N increment (default: 2)')
    parser.add_argument('--psi-types', nargs='*', default=['random'], 
                        choices=['product', 'random'], help='Psi state types to analyze')
    parser.add_argument('--threshold', type=float, default=1e-5, help='Rank threshold (default: 1e-5)')
    
    args = parser.parse_args()
    
    # Generate N values
    N_values = list(range(args.N_start, args.N_end + 1, args.N_step))
    
    print(f"Running rank analysis:")
    print(f"  N values: {N_values}")
    print(f"  Psi types: {args.psi_types}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Total analyses: {len(N_values) * len(args.psi_types)}")
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Run sequential analysis
    start_time = time.time()
    results = run_sequential_analysis(N_values, args.psi_types, args.threshold)
    total_time = time.time() - start_time
    
    print(f"\nTotal analysis time: {total_time:.2f} seconds")
    
    # Print summary
    print_results_summary(results)
    
    # Save results to CSV
    save_results_csv(results)
    
    # Create summary plots
    create_summary_plot(results, args.psi_types)
    
    print(f"\nAll individual spectrum plots saved in 'images/' directory")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 