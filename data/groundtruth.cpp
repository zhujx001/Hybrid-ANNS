#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <future>
#include <numeric>
#include <map>
#include <queue>
#include <filesystem>
#include <cstring>
#include <functional>
#include <atomic>
#include <sstream>
#include <cmath>
#include <iomanip>

// SIMD support for faster vector computations
#ifdef __AVX__
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

// Class to show progress bar
class ProgressBar {
private:
    unsigned int total_;
    unsigned int current_;
    unsigned int width_;
    std::string desc_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point start_time_;

public:
    ProgressBar(unsigned int total, unsigned int width = 50, const std::string& desc = "Progress") 
        : total_(total), current_(0), width_(width), desc_(desc), 
          start_time_(std::chrono::steady_clock::now()) {}

    void update(unsigned int n = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_ += n;
        if (current_ > total_) current_ = total_;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        float progress = static_cast<float>(current_) / total_;
        int pos = static_cast<int>(width_ * progress);
        
        std::ostringstream oss;
        oss << "\r" << desc_ << " [";
        for (int i = 0; i < width_; ++i) {
            if (i < pos) oss << "=";
            else if (i == pos) oss << ">";
            else oss << " ";
        }
        
        float percent = progress * 100.0;
        oss << "] " << std::fixed << std::setprecision(1) << percent << "% ";
        oss << current_ << "/" << total_;
        
        if (elapsed > 0 && progress > 0) {
            double items_per_sec = static_cast<double>(current_) / elapsed;
            double remaining_secs = (total_ - current_) / items_per_sec;
            oss << " [" << elapsed << "s elapsed, ~" << static_cast<int>(remaining_secs) << "s left]";
        }
        
        std::cout << oss.str() << std::flush;
    }

    void finish() {
        update(total_ - current_);
        std::cout << std::endl;
    }
};

// Configuration structure
struct DatasetConfig {
    std::string base_dir;
    std::string label_dir;
    std::string gt_dir;
    std::string base_file;
    std::string query_file;
};

struct QuerySetConfig {
    std::string name;
    std::vector<int> attrs;
    std::string suffix;
};

// Utility for reading and writing binary files
class VecsIO {
public:
    // Read fvecs file
    static std::vector<std::vector<float>> read_fvecs(const std::string& filename, bool use_mmap = false) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<std::vector<float>> vectors;
        
        while (file.good() && !file.eof()) {
            int d;
            file.read(reinterpret_cast<char*>(&d), sizeof(int));
            if (file.eof()) break;
            
            std::vector<float> vec(d);
            file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
            vectors.push_back(vec);
        }
        
        return vectors;
    }

    // Write ivecs file
    static void write_ivecs(const std::string& filename, const std::vector<std::vector<int>>& data) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        for (const auto& row : data) {
            int k = static_cast<int>(row.size());
            file.write(reinterpret_cast<const char*>(&k), sizeof(int));
            file.write(reinterpret_cast<const char*>(row.data()), k * sizeof(int));
        }
    }
};

// Vector operations with SIMD optimizations
class VectorOps {
public:
    // Compute squared Euclidean distance between two vectors
    static float compute_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector dimensions do not match!");
        }
        
        float distance = 0.0f;
        
#ifdef __AVX__
        // Using AVX instructions if available
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        
        // Process 8 elements at a time
        for (; i + 7 < vec1.size(); i += 8) {
            __m256 v1 = _mm256_loadu_ps(&vec1[i]);
            __m256 v2 = _mm256_loadu_ps(&vec2[i]);
            __m256 diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }
        
        // Horizontal sum of the 8 floats in sum
        float result[8];
        _mm256_storeu_ps(result, sum);
        distance = result[0] + result[1] + result[2] + result[3] + 
                   result[4] + result[5] + result[6] + result[7];
        
        // Process remaining elements
        for (; i < vec1.size(); ++i) {
            float diff = vec1[i] - vec2[i];
            distance += diff * diff;
        }
#else
        // Non-AVX implementation
        for (size_t i = 0; i < vec1.size(); ++i) {
            float diff = vec1[i] - vec2[i];
            distance += diff * diff;
        }
#endif

        return distance;
    }

    // Find indices of top K smallest elements in array
    static std::vector<int> find_top_k(const std::vector<float>& distances, int k) {
        if (distances.empty()) return {};
        
        int n = static_cast<int>(distances.size());
        if (n < k) k = n;

        // Use a max-heap for efficient top-k
        std::priority_queue<std::pair<float, int>> max_heap;
        
        for (int i = 0; i < n; ++i) {
            if (max_heap.size() < static_cast<size_t>(k)) {
                max_heap.push({distances[i], i});
            } else if (distances[i] < max_heap.top().first) {
                max_heap.pop();
                max_heap.push({distances[i], i});
            }
        }
        
        // Extract indices in reverse order (smallest distances first)
        std::vector<int> indices(k);
        for (int i = k - 1; i >= 0; --i) {
            indices[i] = max_heap.top().second;
            max_heap.pop();
        }
        
        return indices;
    }
};

// Load base attributes from file
std::vector<std::vector<int>> load_base_attributes(const std::string& filename) {
    std::vector<std::vector<int>> attrs;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> values;
        std::istringstream iss(line);
        int value;
        while (iss >> value) {
            values.push_back(value);
        }
        attrs.push_back(values);
    }
    
    return attrs;
}

// Load queries from file
std::tuple<int, int, std::vector<std::vector<int>>> load_queries(const std::string& filename) {
    std::vector<std::vector<int>> queries;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> values;
        std::istringstream iss(line);
        int value;
        while (iss >> value) {
            values.push_back(value);
        }
        queries.push_back(values);
    }
    
    if (queries.empty()) {
        throw std::runtime_error("Query file is empty");
    }
    
    int Nq = static_cast<int>(queries.size());
    int L = static_cast<int>(queries[0].size());
    
    return {Nq, L, queries};
}

// Worker task for processing one query
struct QueryTask {
    int query_id;
    std::vector<int> query_vals;
    std::vector<int> query_attr_indices;
    const std::vector<std::vector<int>>& base_attrs;
    const std::vector<std::vector<float>>& base_vecs;
    const std::vector<float>& query_vec;
    int top_k;
    
    std::vector<int> execute() const {
        // Create initial mask - all true
        std::vector<bool> mask(base_attrs.size(), true);
        int valid_count = mask.size();
        
        // Apply filter for each attribute
        for (size_t i = 0; i < query_vals.size(); ++i) {
            int val = query_vals[i];
            int attr_idx = query_attr_indices[i];
            
            for (size_t j = 0; j < base_attrs.size(); ++j) {
                if (mask[j] && base_attrs[j][attr_idx] != val) {
                    mask[j] = false;
                    valid_count--;
                }
            }
            
            if (valid_count == 0) {
                return {};  // No valid points left
            }
        }
        
        // Extract valid indices
        std::vector<int> valid_indices;
        valid_indices.reserve(valid_count);
        for (size_t j = 0; j < mask.size(); ++j) {
            if (mask[j]) {
                valid_indices.push_back(j);
            }
        }
        
        if (valid_indices.empty()) {
            return {};
        }
        
        // Compute distances for valid points
        std::vector<float> distances(valid_indices.size());
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            distances[i] = VectorOps::compute_distance(base_vecs[valid_indices[i]], query_vec);
        }
        
        // Find top k indices
        std::vector<int> top_idx = VectorOps::find_top_k(distances, top_k);
        
        // Map back to global indices
        std::vector<int> result(top_idx.size());
        for (size_t i = 0; i < top_idx.size(); ++i) {
            result[i] = valid_indices[top_idx[i]];
        }
        
        return result;
    }
};

// Main function for computing ground truth with k-NN filtering
void compute_ground_truth_with_knn(
    const std::string& base_attr_file,
    const std::string& base_fvecs_file,
    const std::string& query_attr_file,
    const std::string& query_fvecs_file,
    const std::string& output_ivecs_file,
    int top_k = 100,
    const std::vector<int>& label_mapping = {},
    int num_threads = 0,
    bool use_mmap = false,
    int chunk_size = 20
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Determine number of threads if not specified
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    std::cout << "[INFO] System has " << std::thread::hardware_concurrency() << " hardware threads\n";
    std::cout << "[INFO] Using " << num_threads << " threads\n";
    
    // 1) Read base data
    std::cout << "[INFO] Reading base data...\n";
    std::vector<std::vector<int>> base_attrs = load_base_attributes(base_attr_file);
    std::vector<std::vector<float>> base_vecs = VecsIO::read_fvecs(base_fvecs_file, use_mmap);
    int Nb = static_cast<int>(base_vecs.size());
    std::cout << "[INFO] Base data loaded: " << Nb << " vectors, dimension " << base_vecs[0].size() << "\n";
    
    // 2) Read query data
    std::cout << "[INFO] Reading query data...\n";
    auto [Nq, L, query_data] = load_queries(query_attr_file);
    std::vector<std::vector<float>> query_vecs = VecsIO::read_fvecs(query_fvecs_file, use_mmap);
    if (query_vecs.size() != static_cast<size_t>(Nq)) {
        throw std::runtime_error("Query vectors count does not match query lines count");
    }
    std::cout << "[INFO] Query data loaded: " << Nq << " queries, " << L << " labels\n";
    
    // 3) Validate label mapping
    std::vector<int> effective_label_mapping;
    if (label_mapping.empty()) {
        std::cout << "[WARNING] No label mapping provided, using all labels\n";
        effective_label_mapping.resize(L);
        std::iota(effective_label_mapping.begin(), effective_label_mapping.end(), 0);
    } else {
        effective_label_mapping = label_mapping;
    }
    
    for (int label : effective_label_mapping) {
        if (label >= static_cast<int>(base_attrs[0].size())) {
            throw std::runtime_error("Label mapping exceeds base data label columns");
        }
    }
    
    std::cout << "[INFO] Using label mapping: [";
    for (size_t i = 0; i < effective_label_mapping.size(); ++i) {
        std::cout << effective_label_mapping[i];
        if (i < effective_label_mapping.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // 4) Process queries in batches
    std::vector<std::vector<int>> results(Nq);
    std::mutex results_mutex;
    
    auto process_batch = [&](int start_idx, int end_idx) {
        for (int i = start_idx; i < end_idx; ++i) {
            std::vector<int> query_attr_indices;
            query_attr_indices.reserve(query_data[i].size());
            
            for (size_t j = 0; j < query_data[i].size() && j < effective_label_mapping.size(); ++j) {
                query_attr_indices.push_back(effective_label_mapping[j]);
            }
            
            QueryTask task{
                i,
                query_data[i],
                query_attr_indices,
                base_attrs,
                base_vecs,
                query_vecs[i],
                top_k
            };
            
            std::vector<int> result = task.execute();
            
            // Fill with -1 if result is too small
            if (result.size() < static_cast<size_t>(top_k)) {
                result.resize(top_k, -1);
            } else if (result.size() > static_cast<size_t>(top_k)) {
                result.resize(top_k);
            }
            
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                results[i] = std::move(result);
            }
        }
    };
    
    int batch_size = std::max(1, Nq / (num_threads * 2));
    int num_batches = (Nq + batch_size - 1) / batch_size;
    
    std::cout << "[INFO] Processing " << Nq << " queries in " << num_batches << " batches\n";
    ProgressBar progress_bar(num_batches, 50, "Processing batches");
    
    std::vector<std::thread> threads;
    std::atomic<int> completed_batches = 0;
    
    auto batch_worker = [&]() {
        while (true) {
            int batch_idx = completed_batches.fetch_add(1);
            if (batch_idx >= num_batches) break;
            
            int start_idx = batch_idx * batch_size;
            int end_idx = std::min(start_idx + batch_size, Nq);
            
            process_batch(start_idx, end_idx);
            progress_bar.update(1);
        }
    };
    
    // Start all worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(batch_worker);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    progress_bar.finish();
    
    // 5) Write results
    VecsIO::write_ivecs(output_ivecs_file, results);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    std::cout << "[INFO] Ground truth computation completed, result saved to: " << output_ivecs_file << "\n";
    std::cout << "[INFO] Total time: " << duration << " seconds\n";
}

// Get all dataset and query configurations
std::pair<std::map<std::string, DatasetConfig>, std::map<std::string, QuerySetConfig>> get_query_config() {
    std::map<std::string, DatasetConfig> datasets = {
       
        {"msong", {
            "/data/zjxdata/data/msong",
            "/data/filter_data/label/msong_label",
            "/data/filter_data/gt/msong",
            "msong_base.fvecs",
            "msong_query.fvecs"
        }},
        {"audio", {
            "/data/zjxdata/data/audio",
            "/data/filter_data/label/audio_label",
            "/data/filter_data/gt/audio",
            "audio_base.fvecs",
            "audio_query.fvecs"
        }},
        {"enron", {
            "/data/zjxdata/data/enron",
            "/data/filter_data/label/enron_label",
            "/data/filter_data/gt/enron",
            "enron_base.fvecs",
            "enron_query.fvecs"
        }},
        {"gist", {
            "/data/zjxdata/data/gist",
            "/data/filter_data/label/gist_label",
            "/data/filter_data/gt/gist",
            "gist_base.fvecs",
            "gist_query.fvecs"
        }},
        {"glove-100", {
            "/data/zjxdata/data/glove-100",
            "/data/filter_data/label/glove-100_label",
            "/data/filter_data/gt/glove-100",
            "glove-100_base.fvecs",
            "glove-100_query.fvecs"
        }},
         {"sift", {
            "/data/zjxdata/data/sift",
            "/data/filter_data/label/sift_label",
            "/data/filter_data/gt/sift",
            "sift_base.fvecs",
            "sift_query.fvecs"
        }}
    };
    
    std::map<std::string, QuerySetConfig> query_sets = {
        {"1", {
            "基本实验单属性",
            {0},
            "query_set_1"
        }},
        {"2-1", {
            "多属性构建单标签搜索",
            {0},
            "query_set_2_1"
        }},
        {"3-1", {
            "1%选择性实验",
            {5},
            "query_set_3_1"
        }},
        {"3-2", {
            "25%选择性实验",
            {5},
            "query_set_3_2"
        }},
        {"3-3", {
            "50%选择性实验",
            {5},
            "query_set_3_3"
        }},
        {"3-4", {
            "75%选择性实验",
            {5},
            "query_set_3_4"
        }},
        {"4", {
            "1%选择性实验",
            {7},
            "query_set_4"
        }},
        {"5-1", {
            "长尾分布标签实验",
            {1},
            "query_set_5_1"
        }},
        {"5-2", {
            "正态分布标签实验",
            {2},
            "query_set_5_2"
        }},
        {"5-3", {
            "幂律分布标签实验",
            {4},
            "query_set_5_3"
        }},
        {"5-4", {
            "均匀分布标签实验",
            {0},
            "query_set_5_4"
        }},
        {"6", {
            "三标签搜索",
            {0, 8, 9},
            "query_set_6"
        }},
        {"7-1", {
            "多标签1%选择性",
            {0, 8, 5},
            "query_set_7_1"
        }},
        {"7-3", {
            "多标签50%选择性",
            {0, 8, 5},
            "query_set_7_3"
        }},
        {"7-4", {
            "多标签75%选择性",
            {0, 8, 5},
            "query_set_7_4"
        }}
    };
    
    return {datasets, query_sets};
}

// Process all datasets and query sets
void process_all_datasets() {
    auto [datasets, query_sets] = get_query_config();
    
    for (const auto& [dataset_name, dataset_config] : datasets) {
        std::cout << "\n[INFO] Processing dataset: " << dataset_name << "\n";
        
        std::string base_fvecs = dataset_config.base_dir + "/" + dataset_config.base_file;
        std::string query_fvecs = dataset_config.base_dir + "/" + dataset_config.query_file;
        std::string base_attr = dataset_config.label_dir + "/labels.txt";
        
        for (const auto& [query_id, query_config] : query_sets) {
            std::cout << "\n[INFO] Processing query set: " << query_config.name << " (ID: " << query_id << ")\n";
            
            // Construct query file path
            std::string query_attr = dataset_config.label_dir + "/query_" + query_config.suffix + ".txt";
            
            // Construct output file path
            std::string output_dir = dataset_config.gt_dir;
            fs::create_directories(output_dir);
            std::string output_file = output_dir + "/gt-" + query_config.suffix + ".ivecs";
            
            try {
                // Check if files exist
                if (!fs::exists(base_fvecs) || !fs::exists(query_fvecs) || 
                    !fs::exists(base_attr) || !fs::exists(query_attr)) {
                    std::cout << "[WARNING] Skipping " << dataset_name << "-" << query_id << ": Some files don't exist\n";
                    continue;
                }
                
                std::cout << "[INFO] Starting ground truth computation: " << output_file << "\n";
                compute_ground_truth_with_knn(
                    base_attr,
                    base_fvecs,
                    query_attr,
                    query_fvecs,
                    output_file,
                    100,
                    query_config.attrs,
                    96,  // Use 96 threads as in the original code
                    true,
                    20
                );
                std::cout << "[INFO] Completed query set " << query_id << "\n";
                
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Error processing " << dataset_name << "-" << query_id << ": " << e.what() << "\n";
                continue;
            }
        }
    }
}

int main() {
    try {
        process_all_datasets();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
