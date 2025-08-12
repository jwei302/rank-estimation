#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <future>
#include <memory>

// Include Eigen for matrix operations and SVD
#include <Eigen/Dense>
#include <Eigen/SVD>

// Include nlohmann/json for JSON output
#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

// Type definitions for better performance
using MatrixXdPtr = shared_ptr<MatrixXd>;
using VectorXdPtr = shared_ptr<VectorXd>;

// Structure to hold results
struct ComputationResult {
    int N;
    vector<int> jacobian_shape;
    int matrix_rank;
    vector<double> singular_values;
    double computation_time;
    int num_I;
    int num_a;
    string psi_state;
};

// Generate I_list and a_list for given N
pair<vector<vector<int>>, vector<vector<int>>> get_I_a_list(int N) {
    int n = N / 2;
    vector<vector<int>> I_list, a_list;
    
    // Generate I_list: all combinations of 2 elements from {1, 2, ..., N}
    for (int i = 1; i <= N; i++) {
        for (int j = i + 1; j <= N; j++) {
            I_list.push_back({i, j});
        }
    }
    
    // Generate a_list: all combinations of n elements from {1, 2, ..., N}
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 1);
    
    // Generate all combinations of size n
    vector<bool> selector(N);
    fill(selector.begin(), selector.begin() + n, true);
    
    do {
        vector<int> combination;
        for (int i = 0; i < N; i++) {
            if (selector[i]) {
                combination.push_back(indices[i]);
            }
        }
        a_list.push_back(combination);
    } while (prev_permutation(selector.begin(), selector.end()));
    
    return make_pair(I_list, a_list);
}

// Optimized Levi-Civita sign computation using cycle counting
int levi_civita_sign_optimized(const vector<int>& big_set, const vector<int>& sub_set) {
    vector<int> perm = sub_set;
    
    // Add remaining elements from big_set using unordered_set for O(1) lookup
    unordered_set<int> sub_set_set(sub_set.begin(), sub_set.end());
    for (int x : big_set) {
        if (sub_set_set.find(x) == sub_set_set.end()) {
            perm.push_back(x);
        }
    }
    
    // Create mapping from value to position using unordered_map for O(1) lookup
    unordered_map<int, int> value_to_pos;
    for (int i = 0; i < perm.size(); i++) {
        value_to_pos[perm[i]] = i;
    }
    
    // Count cycles
    vector<bool> visited(perm.size(), false);
    int cycles = 0;
    
    for (int i = 0; i < perm.size(); i++) {
        if (!visited[i]) {
            cycles++;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = value_to_pos[big_set[j]];
            }
        }
    }
    
    // Sign is (-1)^(n - cycles) where n is the length
    return ((perm.size() - cycles) % 2 == 0) ? 1 : -1;
}

// Create lookup tables for optimization
pair<map<int, vector<int>>, map<pair<int, int>, vector<int>>> create_lookup_tables(
    const vector<vector<int>>& I_list, 
    const vector<vector<int>>& a_list) {
    
    map<int, vector<int>> I_subset_of_a;
    map<pair<int, int>, vector<int>> Jp_values;
    
    for (int i = 0; i < I_list.size(); i++) {
        I_subset_of_a[i] = vector<int>();
        set<int> I_set(I_list[i].begin(), I_list[i].end());
        
        for (int a_idx = 0; a_idx < a_list.size(); a_idx++) {
            set<int> a_set(a_list[a_idx].begin(), a_list[a_idx].end());
            
            // Check if I is subset of a
            bool is_subset = true;
            for (int elem : I_set) {
                if (a_set.find(elem) == a_set.end()) {
                    is_subset = false;
                    break;
                }
            }
            
            if (is_subset) {
                I_subset_of_a[i].push_back(a_idx);
                
                // Compute Jp = a - I (set difference)
                vector<int> Jp;
                for (int elem : a_set) {
                    if (I_set.find(elem) == I_set.end()) {
                        Jp.push_back(elem);
                    }
                }
                sort(Jp.begin(), Jp.end());
                Jp_values[make_pair(i, a_idx)] = Jp;
            }
        }
    }
    
    return make_pair(I_subset_of_a, Jp_values);
}

// Generate random psi state
VectorXd psi_random(int num_a) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    VectorXd psi(num_a);
    for (int i = 0; i < num_a; i++) {
        psi(i) = dis(gen);
    }
    
    // Normalize
    psi.normalize();
    return psi;
}

// Generate product state
VectorXd psi_product_state(int num_a) {
    VectorXd psi = VectorXd::Zero(num_a);
    psi(0) = 1.0;
    return psi;
}

// Ultra-optimized direct matrix implementation - eliminates all allocation overhead
pair<MatrixXdPtr, MatrixXdPtr> optimized_implementation_2_parallel(
    const vector<vector<int>>& I_list,
    const vector<vector<int>>& a_list,
    const VectorXd& psi,
    int num_threads = thread::hardware_concurrency()) {
    
    int num_I = I_list.size();
    int num_a = a_list.size();
    
    // Create matrices directly - no sparse representation overhead
    auto J1 = make_shared<MatrixXd>(MatrixXd::Zero(num_I * num_I, num_a));
    auto J2 = make_shared<MatrixXd>(MatrixXd::Zero(num_I * num_I, num_a));
    
    // Group valid combinations by their Jp values and precompute everything
    map<vector<int>, vector<tuple<int, int, int>>> Jp_groups;
    
    for (int i = 0; i < I_list.size(); i++) {
        const auto& I = I_list[i];
        for (int a_idx = 0; a_idx < a_list.size(); a_idx++) {
            const auto& a = a_list[a_idx];
            
            // Fast subset check using sorted vectors
            bool is_subset = includes(a.begin(), a.end(), I.begin(), I.end());
            
            if (is_subset) {
                // Compute Jp = a - I using set_difference (very fast)
                vector<int> Jp;
                Jp.reserve(a.size() - I.size());
                set_difference(a.begin(), a.end(), I.begin(), I.end(), back_inserter(Jp));
                
                // Precompute Levi-Civita sign
                int s1 = levi_civita_sign_optimized(a, I);
                Jp_groups[Jp].push_back(make_tuple(i, a_idx, s1));
            }
        }
    }
    
    // Count total operations
    size_t total_operations = 0;
    for (const auto& group : Jp_groups) {
        size_t group_size = group.second.size();
        total_operations += group_size * group_size;
    }
    
    cout << "  Found " << Jp_groups.size() << " unique Jp groups" << endl;
    cout << "  Total operations to process: " << total_operations << endl;
    
    // Convert groups to vector for parallel processing
    vector<pair<vector<int>, vector<tuple<int, int, int>>>> group_vec(Jp_groups.begin(), Jp_groups.end());
    
    // Mutex for thread safety on matrices
    mutex matrix_mutex;
    
    // Parallel processing function - direct matrix updates
    auto process_chunk = [&](int start_idx, int end_idx) {
        // Local matrices to minimize lock time
        MatrixXd local_J1 = MatrixXd::Zero(num_I * num_I, num_a);
        MatrixXd local_J2 = MatrixXd::Zero(num_I * num_I, num_a);
        
        for (int group_idx = start_idx; group_idx < end_idx; group_idx++) {
            const auto& group = group_vec[group_idx].second;
            
            // Process all pairs within this group (same Jp) - direct matrix access
            for (const auto& combo1 : group) {
                int i = get<0>(combo1);
                int a_idx = get<1>(combo1);
                int s1 = get<2>(combo1);
                
                for (const auto& combo2 : group) {
                    int ip = get<0>(combo2);
                    int ap_idx = get<1>(combo2);
                    int s2 = get<2>(combo2);
                    
                    double factor = s1 * s2;
                    int linear_idx = i * num_I + ip;
                    
                    // Direct matrix assignment - no allocations
                    local_J1(linear_idx, a_idx) = factor * psi(ap_idx);
                    local_J2(linear_idx, ap_idx) = factor * psi(a_idx);
                }
            }
        }
        
        // Quick matrix addition - much faster than vector operations
        lock_guard<mutex> lock(matrix_mutex);
        *J1 += local_J1;
        *J2 += local_J2;
    };
    
    // Launch parallel tasks
    vector<future<void>> futures;
    int chunk_size = max(1, (int)group_vec.size() / num_threads);
    
    for (int t = 0; t < num_threads; t++) {
        int start_idx = t * chunk_size;
        int end_idx = (t == num_threads - 1) ? group_vec.size() : (t + 1) * chunk_size;
        
        if (start_idx < group_vec.size()) {
            futures.push_back(async(launch::async, process_chunk, start_idx, end_idx));
        }
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return make_pair(J1, J2);
}

// Compute Jacobian matrix
MatrixXd compute_jacobian_matrix(const MatrixXd& J1, const MatrixXd& J2) {
    MatrixXd J(J1.rows(), J1.cols() + J2.cols());
    J << J1, J2;
    return J;
}

// Compute rank using SVD
pair<int, VectorXd> compute_rank(const MatrixXd& J, double threshold = 1e-5) {
    JacobiSVD<MatrixXd> svd(J, ComputeThinU | ComputeThinV);
    VectorXd s = svd.singularValues();
    
    int rank = 0;
    for (int i = 0; i < s.size(); i++) {
        if (s(i) > threshold) {
            rank++;
        }
    }
    
    return make_pair(rank, s);
}

// Run pipeline for a single N value
ComputationResult run_pipeline_single(int N, const string& psi_state_name = "random") {
    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "Processing N=" << N << " with " << psi_state_name << " state..." << endl;
    
    // Get I and a lists
    auto lists = get_I_a_list(N);
    auto& I_list = lists.first;
    auto& a_list = lists.second;
    
    int num_I = I_list.size();
    int num_a = a_list.size();
    
    cout << "  num_I: " << num_I << ", num_a: " << num_a << endl;
    
    // Generate psi state
    VectorXd psi;
    if (psi_state_name == "random") {
        psi = psi_random(num_a);
    } else {
        psi = psi_product_state(num_a);
    }
    
    // Compute J1, J2 using parallel optimized implementation
    auto J_result = optimized_implementation_2_parallel(I_list, a_list, psi);
    auto& J1 = *J_result.first;
    auto& J2 = *J_result.second;
    
    // Compute Jacobian matrix
    MatrixXd J = compute_jacobian_matrix(J1, J2);
    
    cout << "  Jacobian shape: " << J.rows() << " x " << J.cols() << endl;
    
    // Compute rank and singular values
    auto rank_result = compute_rank(J);
    int matrix_rank = rank_result.first;
    VectorXd singular_values = rank_result.second;
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    double computation_time = duration.count() / 1000.0;
    
    cout << "  Rank: " << matrix_rank << ", Time: " << computation_time << "s" << endl;
    
    // Prepare result
    ComputationResult result;
    result.N = N;
    result.jacobian_shape = {(int)J.rows(), (int)J.cols()};
    result.matrix_rank = matrix_rank;
    result.singular_values = vector<double>(singular_values.data(), 
                                          singular_values.data() + singular_values.size());
    result.computation_time = computation_time;
    result.num_I = num_I;
    result.num_a = num_a;
    result.psi_state = psi_state_name;
    
    return result;
}

// Convert results to JSON
json results_to_json(const vector<ComputationResult>& results) {
    json output;
    output["metadata"] = {
        {"description", "Rank estimation pipeline results"},
        {"algorithm", "optimized_implementation_2_parallel"},
        {"levi_civita", "optimized_cycle_counting"},
        {"threshold", 1e-5}
    };
    
    json data = json::array();
    for (const auto& result : results) {
        json entry = {
            {"N", result.N},
            {"jacobian_shape", result.jacobian_shape},
            {"matrix_rank", result.matrix_rank},
            {"singular_values", result.singular_values},
            {"computation_time_seconds", result.computation_time},
            {"num_I", result.num_I},
            {"num_a", result.num_a},
            {"psi_state", result.psi_state}
        };
        data.push_back(entry);
    }
    
    output["results"] = data;
    return output;
}

int main() {
    cout << "C++ Rank Estimation Pipeline" << endl;
    cout << "=============================" << endl;
    cout << "Using " << thread::hardware_concurrency() << " threads" << endl << endl;
    
    auto total_start = chrono::high_resolution_clock::now();
    
    vector<ComputationResult> all_results;
    
    // Run for N = 4 to 20 in steps of 2
    for (int N = 4; N <= 20; N += 2) {
        try {
            // Run with random state only
            ComputationResult result_random = run_pipeline_single(N, "random");
            all_results.push_back(result_random);
            
        } catch (const exception& e) {
            cerr << "Error processing N=" << N << ": " << e.what() << endl;
            break;
        }
        
        cout << endl;
    }
    
    auto total_end = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(total_end - total_start);
    
    cout << "Total computation time: " << total_duration.count() << " seconds" << endl;
    
    // Save results to JSON
    json output = results_to_json(all_results);
    output["metadata"]["total_computation_time_seconds"] = total_duration.count();
    
    ofstream file("rank_results.json");
    file << output.dump(4);
    file.close();
    
    cout << "Results saved to rank_results.json" << endl;
    
    return 0;
} 