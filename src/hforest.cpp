#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <H5Cpp.h>
#include <omp.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

namespace py = pybind11;

#include "assert.hpp"
#include "fast_vector.hpp"

using Candidate = std::tuple<int, int>;
using Candidates = fast_vector<Candidate>;

#include "htree.hpp"
#include "hsort.hpp"
#include "hsearch.hpp"
#include "timing.hpp"
#include "tree_encoder.hpp"
#include "forest_utils.hpp"
#include "progress_bar.hpp"
#include "utils.hpp"

// Bidirectional mapping between original IDs and pre-Hilbert sorted indices
struct IDMapping {
    std::vector<int> id_to_preidx;
    std::vector<int> preidx_to_id;
};

class HilbertForest {
private:
    int dimensions = -1;
    int bitDepth = 8;          // 8 bits = 0-255 range
    int ntrees_total;          // Total number of trees built during training
    int ntrees;                // Number of trees to use during search (can be <= ntrees_total)
    int odd_candidates = 11;   // Number of candidates for odd-level nodes
    int even_candidates = 10;  // Number of candidates for even-level nodes
    int dist_candidates = 100; // Number of candidate points for distance calculation
    int hops = 1;              // Search range for neighboring points (pre_idx ±hops)
    int leaf_size = 1;         // Stop splitting when node size falls below this
    bool is_trained = false;
    int verbose = 1;           // 0=silent, 1=normal, 2=verbose
    float rate = 500.0f;       // Quantization rate
    
    // Quantized data storage (in pre-Hilbert sort order)
    uint8_t* points_data = nullptr;
    int num_points = 0;
    
    // Sketch data storage (in pre-Hilbert sort order)
    uint8_t* sketches_data = nullptr;
    void* sketches_memory = nullptr;
    
    IDMapping id_mapping;
    ForestTiming timing;
    
    struct TreeInfo {
        HilbertTree* tree;  // mmap'ed memory pointer
        
        TreeInfo() : tree(nullptr) {}
        
        void clear() {
            if (tree != nullptr) {
                TreeEncoder::unmapTree(tree);
                tree = nullptr;
            }
        }
    };
    
    std::vector<TreeInfo> trees;
    std::string dbPath = "db";
    std::mt19937 rng;
    
    void clearTrees() {
        for (auto& tree_info : trees) {
            tree_info.clear();
        }
        trees.clear();
        
        if (points_data != nullptr) {
            delete[] points_data;
            points_data = nullptr;
            num_points = 0;
        }
        
        if (sketches_memory != nullptr) {
            free(sketches_memory);
            sketches_memory = nullptr;
        }
    }
    
    void ensureDbDirectory() {
        struct stat st;
        if (stat(dbPath.c_str(), &st) != 0) {
            int result __attribute__((unused)) = mkdir(dbPath.c_str(), 0755);
            assert(result == 0 && "Failed to create db directory");
        } else {
            assert(S_ISDIR(st.st_mode) && "db must be a directory");
        }
    }
    
public:
    HilbertForest(const std::string& db_path = "db", int ntrees = 10, int leaf_size = 1, int verbose_level = 1, int seed = -1)
        : ntrees_total(ntrees), ntrees(ntrees), leaf_size(leaf_size),
          verbose(verbose_level), timing(), dbPath(db_path) {
        assert(leaf_size > 0);
        assert(verbose_level >= 0 && verbose_level <= 2);
        if (seed == -1) {
            rng = std::mt19937(std::random_device{}());
        } else {
            rng = std::mt19937(seed);
        }
    }

    ~HilbertForest() {
        clearTrees();
    }

    // Preload data and complete quantization and pre-Hilbert sorting
    void preload(py::object data) {
        timing.reset();
        
        py::object shape = data.attr("shape");
        int num_points = py::cast<int>(shape[py::int_(0)]);
        int data_dimensions = py::cast<int>(shape[py::int_(1)]);
        
        if (dimensions == -1) {
            dimensions = data_dimensions;
            // Calculate rate once when dimensions are set
            rate = sqrt(sqrt(dimensions)) * 128.0f;
        } else {
            assert(data_dimensions == dimensions && "Data dimensions do not match previously set dimensions");
        }
        
        assert(dimensions > 0);
        
        clearTrees();
        
        const int batch_size = 1000;

        this->num_points = num_points;
        
        // Prepare temporary buffer for quantization (before ID sorting)
        size_t temp_data_size = size_t(dimensions) * num_points;
        uint8_t* temp_points_data = new uint8_t[temp_data_size];
        
        id_mapping.id_to_preidx.resize(num_points);
        id_mapping.preidx_to_id.resize(num_points);
        
        std::vector<int> all_points;
        all_points.reserve(num_points);
        
        py::gil_scoped_acquire acquire;
        
        {
            ProgressBar progress_bar(num_points, "Quantizing data points", verbose);
            double data_fetch_time = 0.0;
            double numpy_convert_time = 0.0;
            double point_transform_time = 0.0;
            
            for (int start_idx = 0; start_idx < num_points; start_idx += batch_size) {
                int current_batch = std::min(batch_size, num_points - start_idx);
                
                // Get batch data
                {
                    ScopedTimer timer(data_fetch_time);
                    py::slice batch_slice = py::slice(start_idx, start_idx + current_batch, 1);
                    py::object batch = data.attr("__getitem__")(batch_slice);
                    timer.stop();
                
                    // Convert to NumPy array
                    ScopedTimer timer2(numpy_convert_time);
                    py::array_t<float> batch_array = py::array::ensure(batch);
                    py::buffer_info buf = batch_array.request();
                    float* ptr = (float*)(void*)(buf.ptr);
                    timer2.stop();
                
                    // Process each data point
                    ScopedTimer timer3(point_transform_time);
                    
                    int num_threads = std::min(omp_get_max_threads(), 4);
                    
                    // Prepare local buffers for each thread
                    std::vector<std::vector<int>> thread_points(num_threads);
                    for (int t = 0; t < num_threads; t++) {
                        thread_points[t].reserve(current_batch / num_threads + 1);
                    }
                    
                    // Parallel processing with OpenMP
                    #pragma omp parallel num_threads(num_threads)
                    {
                        int thread_id = omp_get_thread_num();
                        
                        #pragma omp for
                        for (int i = 0; i < current_batch; i++) {
                            int original_idx = start_idx + i;
                            
                            // Calculate pointer to the point in temporary buffer (using 64-bit integer calculation)
                            uint8_t* point_ptr = temp_points_data + size_t(original_idx) * dimensions;
                            
                            // Transform and add to point cloud (ID is stored externally)
                            ForestUtils::transform_point(ptr + i * dimensions, point_ptr, dimensions, rate);
                            
                            // Add index to thread-specific buffer
                            thread_points[thread_id].push_back(original_idx);
                        }
                    }
                    
                    // Merge local buffers from each thread
                    for (int t = 0; t < num_threads; t++) {
                        all_points.insert(all_points.end(), thread_points[t].begin(), thread_points[t].end());
                        thread_points[t].clear();
                    }
                }
                
                progress_bar.update(current_batch);
            }
            
            progress_bar.complete();
            
            // Display timing results
            if (verbose) {
                std::cerr << "  Data fetch time: " << data_fetch_time << "s" << std::endl;
                std::cerr << "  Numpy array conversion time: " << numpy_convert_time << "s" << std::endl;
                std::cerr << "  Point transformation time: " << point_transform_time << "s" << std::endl;
            }
        }
        
        // Pre-Hilbert sort
        {
            ProgressBar progress_bar(1, "Pre-Hilbert sort", verbose);
            std::vector<int> pre_hilbert_axes(dimensions);
            for (int i = 0; i < dimensions; i++) {
                pre_hilbert_axes[i] = i;
            }
            HilbertSort pre_sorter(pre_hilbert_axes, bitDepth, rng, temp_points_data, dimensions, 1);
            pre_sorter.sort(all_points);
            progress_bar.complete();
        }
        
        // Create ID <-> PRE_IDX mapping based on sort results
        for (int pre_idx = 0; pre_idx < num_points; pre_idx++) {
            int original_id = all_points[pre_idx];
            id_mapping.preidx_to_id[pre_idx] = original_id;
            id_mapping.id_to_preidx[original_id] = pre_idx;
        }
        
        // Create new memory layout based on pre-Hilbert sort order
        size_t point_size = size_t(dimensions);
        size_t total_size = point_size * num_points;
        points_data = new uint8_t[total_size];
        
        // Place points in PRE_IDX order
        for (int pre_idx = 0; pre_idx < num_points; pre_idx++) {
            int original_id = id_mapping.preidx_to_id[pre_idx];
            uint8_t* src_ptr = temp_points_data + size_t(original_id) * dimensions;
            uint8_t* dst_ptr = points_data + size_t(pre_idx) * dimensions;
            
            std::memcpy(dst_ptr, src_ptr, dimensions);
        }
        
        // Release temporary buffer
        delete[] temp_points_data;
        
        // Generate sketches from quantized data (stored in PRE_IDX order)
        {
            std::stringstream ss;
            int sketch_bytes = (dimensions + 7) >> 3;
            ss << "Generating sketches (" << sketch_bytes << " bytes/point, total: " << sketch_bytes * num_points << " bytes)";
            ProgressBar progress_bar(1, ss.str(), verbose);

            // Generate sketches for entire dataset
            size_t sketch_qword_bytes = (size_t((dimensions + 7) >> 3) + 7) & ~7;
            sketches_memory = malloc(sketch_qword_bytes * num_points + 8);
            sketches_data = (uint8_t*)(void*)(((size_t)sketches_memory + 7) & ~7);
            ForestUtils::generate_sketches_batch(points_data, num_points, dimensions, sketches_data, timing);

            progress_bar.complete();
        }
    }

    // Build index from dataset
    void fit(py::object data = py::none()) {
        // If data is not specified, use preloaded data
        if (!data.is_none()) {
            // If new data is specified, perform preload
            preload(data);
        }
        assert(points_data != nullptr && "No preloaded data. Call preload() first or specify data in fit()");
        
        timing.reset();
        
        ensureDbDirectory();
        
        // Build trees
        trees.resize(ntrees_total);
        
        size_t total_node_count = 0;
        
        ProgressBar tree_build_bar(ntrees_total, "Building trees", verbose);
        
        // Reusable index array
        std::vector<int> sorted_points;
        sorted_points.reserve(num_points);
        
        for (int tree_idx = 0; tree_idx < ntrees_total; tree_idx++) {
            // Create formal tree file path
            std::string tree_file_path = dbPath + "/tree_" + std::to_string(tree_idx) + ".bin";
            
            // First check for existing file
            HilbertTree* existing_tree = TreeEncoder::loadTreeFile(tree_file_path);
            
            if (existing_tree != nullptr) {
                // If existing tree file found, load it
                trees[tree_idx].tree = existing_tree;
                
                // Estimate node count for statistics
                size_t estimated_nodes = existing_tree->pointIdOffset / existing_tree->nodeBits;
                total_node_count += estimated_nodes;
            } else {
                // If tree file not found, create new one
                
                sorted_points.clear();
                
                // Prepare indices for representative points of each group
                for (int pre_idx = 0; pre_idx < num_points; ++pre_idx) {
                    // Use PRE_IDX directly as index
                    sorted_points.push_back(pre_idx);
                }
                
                // Generate axis indices from 0 to dimensions-1
                std::vector<int> shuffled_axes(dimensions);
                for (int i = 0; i < dimensions; i++) {
                    shuffled_axes[i] = i;
                }
                // Randomly shuffle axes
                std::shuffle(shuffled_axes.begin(), shuffled_axes.end(), rng);
                //shuffled_axes.resize(dimensions - int(dimensions * 0.1));

                // Split nodes with configured minimum leaf size
                HilbertSort sorter(shuffled_axes, bitDepth, rng, points_data, dimensions, leaf_size);
                int rootNodeId __attribute__((unused)) = sorter.sort(sorted_points);
                assert(rootNodeId == 0);

                // Count nodes
                size_t node_count = sorter.getNodeCount();
                total_node_count += node_count;
                
                // Get constructed tree structure
                const std::vector<int>& treeNodes = sorter.getTreeNodes();
                
                // Get internal node information (axis * bitDepth + bitPos)
                const std::vector<int>& nodeInfo = sorter.getNodeInfo();
                
                // Encode tree structure in binary format
                TreeEncoder encoder(treeNodes, nodeInfo, sorted_points, bitDepth, dimensions);
                encoder.encode();
                
                // Create temporary file name
                std::string temp_tree_file_path = dbPath + "/temp_tree_" + std::to_string(tree_idx) + "_" + 
                                                 std::to_string(std::random_device{}()) + ".bin.tmp";
                
                // Save tree to temporary file and mmap it
                HilbertTree* temp_tree = encoder.createTreeFile(temp_tree_file_path);
                
                // Unmap temporary file
                TreeEncoder::unmapTree(temp_tree);
                
                // Rename temporary file to formal file name
                int rename_result __attribute__((unused)) = rename(temp_tree_file_path.c_str(), tree_file_path.c_str());
                assert(rename_result == 0 && "Failed to rename temporary file");
                
                // Load formal file with mmap for reading
                trees[tree_idx].tree = TreeEncoder::loadTreeFile(tree_file_path);
                assert(trees[tree_idx].tree != nullptr && "Failed to load formal file");
            }

            tree_build_bar.update();
        }
        
        tree_build_bar.complete();
        
        is_trained = true;

        if (verbose) {
            // Display statistics (after progress bar completion)
            double avg_node_count = double(total_node_count) / ntrees_total;
            std::cerr << "  Average node count: " << avg_node_count << std::endl;
        }
    }
    
    // Get whether trained or not
    bool is_trained_getter() const {
        return is_trained;
    }
    
    // Getters and setters for number of trees to use
    int get_ntrees() const {
        return ntrees;
    }
    
    void set_ntrees(int nt) {
        assert(nt > 0 && nt <= ntrees_total);
        ntrees = nt;
    }
    
    // Getters and setters for candidate numbers
    int get_odd_candidates() const {
        return odd_candidates;
    }
    
    void set_odd_candidates(int n) {
        assert(n > 0);
        odd_candidates = n;
    }
    
    int get_even_candidates() const {
        return even_candidates;
    }
    
    void set_even_candidates(int n) {
        assert(n > 0);
        even_candidates = n;
    }
    
    // Getters and setters for number of candidate points for distance calculation
    int get_dist_candidates() const {
        return dist_candidates;
    }
    
    void set_dist_candidates(int n) {
        assert(n > 0);
        dist_candidates = n;
    }
    
    // Getters and setters for hops
    int get_hops() const {
        return hops;
    }
    
    void set_hops(int n) {
        assert(n >= 0);
        hops = n;
    }

    // Nearest neighbor search (optimized: duplicate counting, scoring and lazy distance calculation)
    std::tuple<py::array_t<float>, py::array_t<int>> search(py::array_t<float> queries, int k) {
        timing.reset();
        
        assert(is_trained && "HilbertForest must be fitted before searching");
        
        // Verify query dimensions
        py::buffer_info buf = queries.request();
        assert(buf.ndim == 2 && buf.shape[1] == dimensions && "Query dimensions must match index dimensions");
        
        int num_queries = buf.shape[0];
        if(verbose) std::cerr << "Searching: queries=" << num_queries << ", k=" << k << std::endl;

        // Create result arrays
        py::array_t<float> D(py::array::ShapeContainer({num_queries, k}));
        py::array_t<int> I(py::array::ShapeContainer({num_queries, k}));
        
        auto d_ptr = D.mutable_unchecked<2>();
        auto i_ptr = I.mutable_unchecked<2>();
        
        // Number of trees to actually use (smaller of ntrees and ntrees_total)
        int used_trees = std::min(ntrees, ntrees_total);
        
        // Data structures for per-query processing
        // Arrays to store overall statistics (maintain statistics even with per-query processing for memory efficiency)
        std::vector<int> total_candidates_per_query(num_queries, 0);
        std::vector<int> filtered_candidates_per_query(num_queries, 0);
        
        // Quantize all queries once in advance
        std::vector<uint8_t> quantized_queries;
        {
            ProgressBar progress_bar(1, "Quantizing query points", verbose);
            quantized_queries = ForestUtils::quantize_queries_batch(queries, dimensions, rate, timing);
            progress_bar.complete();
        }
        
        // Prepare pre-quantized queries according to axis mapping
        std::vector<uint8_t> prepared_queries;
        {
            ProgressBar progress_bar(1, "Encoding queries", verbose);
            prepared_queries = ForestUtils::prepare_queries_batch(
                quantized_queries, num_queries, dimensions, timing);
            progress_bar.complete();
        }
            
        int num_threads = omp_get_max_threads();
        if (verbose) std::cerr << "OpenMP threads: " << num_threads << std::endl;

        size_t block_size = 65536;
        size_t block_mask = block_size - 1;

        // Prepare memory for tree processing (allocate in upper scope)
        size_t tree_mem_size = sizeof(int) * num_queries;
        size_t tree_mem_qword_size = (tree_mem_size + block_mask) & ~block_mask;
        uint8_t* tree_mem_memory = (uint8_t*)malloc(tree_mem_qword_size * used_trees + block_size);// Actually +block_mask would be sufficient, but that would always allocate odd bytes which feels awkward, and allocating 1 extra byte (not -1) might result in better memory reuse in malloc/free cycles
        uint8_t* tree_mem_base = (uint8_t*)((size_t(tree_mem_memory) + block_mask) & ~block_mask);

        // Create progress bar for tree processing
        ProgressBar tree_progress_bar(used_trees, "Processing trees", verbose);
        {
            int progress_omp_counter = 0;

            // Prepare memory for each thread
            size_t work_size2 = 0;

            // Memory for index array
            size_t indices_bytes = sizeof(int) * num_queries;
            size_t indices_qword_bytes = (indices_bytes + 7) & ~7;
            work_size2 += indices_qword_bytes;

            work_size2 = (work_size2 + block_mask) & ~block_mask;
            uint8_t * work_memory = (uint8_t *)malloc(work_size2 * num_threads + block_size);// Actually +block_mask would be sufficient, but that would always allocate odd bytes which feels awkward, and allocating 1 extra byte (not -1) might result in better memory reuse in malloc/free cycles
            uint8_t * whole_base = (uint8_t *)((size_t(work_memory) + block_mask) & ~block_mask);
            
            // Query data is read-only so can be shared
            uint8_t * queries = prepared_queries.data();
            
            // Parallel processing with OpenMP
            #pragma omp parallel num_threads(num_threads)
            {
                int thread_id = omp_get_thread_num();
                assert(0 <= thread_id && thread_id < num_threads);

                uint8_t * work_base2 = whole_base + work_size2 * thread_id;

                // Initialize index array
                int* query_indices = (int*)(void*)work_base2;
                for (int i = 0; i < num_queries; i++) {
                    query_indices[i] = i;
                }
                work_base2 += indices_qword_bytes;

                // Batch process for each tree
                #pragma omp for schedule(dynamic)
                for (int t = 0; t < used_trees; t++) {
                    // Pointer to result storage array
                    int* results = (int*)(tree_mem_base + tree_mem_qword_size * t);

                    // Search all at once using HilbertTreeSearch
                    HilbertTreeSearch searcher(trees[t].tree);
                    // Calculate query bit count
                    int totalQueryBits = dimensions * 8;
                    searcher.search(queries, num_queries, totalQueryBits, results, query_indices);
                    
                    // Update per-tree progress
                    if (2<=verbose) {
                        #pragma omp critical(progress_bar)
                        {
                            ++progress_omp_counter;
                            if (thread_id==0) {
                                tree_progress_bar.update(progress_omp_counter);
                                progress_omp_counter = 0;
                            }
                        }
                    }
                }
            }

            // Free thread memory
            free(work_memory);
        }
        tree_progress_bar.complete();
        
        // Process each query in parallel
        {
            ProgressBar progress_bar(num_queries, "Processing queries", verbose);
            int progress_omp_counter = 0;
            
            // Prepare sketch buffer for each thread
            size_t work_size = 0;

            // Timing must always be placed first
            size_t timing_bytes = sizeof(ForestTiming);
            size_t timing_qword_bytes = (timing_bytes + 7) & ~7;
            work_size += timing_qword_bytes;

            // Calculate sketch size
            size_t sketch_bytes = (dimensions + 7) >> 3;
            size_t sketch_qword_bytes = (sketch_bytes + 7) & ~7;
            work_size += sketch_qword_bytes;

            // Set threshold for early filtering (4 * dist_candidates)
            size_t scores_capacity = dist_candidates * 4;
            size_t scores_bytes = sizeof(Candidates) + sizeof(Candidate) * scores_capacity;
            size_t scores_qword_bytes = (scores_bytes + 7) & ~7;
            work_size += scores_qword_bytes;

            size_t filtered_capacity = dist_candidates * (1 + hops + hops);
            size_t filtered_bytes = sizeof(Candidates) + sizeof(Candidate) * filtered_capacity;
            size_t filtered_qword_bytes = (filtered_bytes + 7) & ~7;
            work_size += filtered_qword_bytes;

            work_size = (work_size + block_mask) & ~block_mask;
            uint8_t * work_memory = (uint8_t *)malloc(work_size * num_threads + block_size);// Actually +block_mask would be sufficient, but that would always allocate odd bytes which feels awkward, and allocating 1 extra byte (not -1) might result in better memory reuse in malloc/free cycles
            uint8_t * whole_base = (uint8_t *)((size_t(work_memory) + block_mask) & ~block_mask);
            
            // Parallel processing with OpenMP
            #pragma omp parallel num_threads(num_threads)
            {
                int thread_id = omp_get_thread_num();
                assert(0 <= thread_id && thread_id < num_threads);

                uint8_t * work_base = whole_base + work_size * thread_id;

                ForestTiming & timing = *(ForestTiming*)(void*)work_base;
                timing.reset();
                work_base += timing_qword_bytes;

                uint8_t * const query_sketch = work_base;
                work_base += sketch_qword_bytes;

                Candidates & scores = *(Candidates*)(void*)work_base;
                scores.init(scores_capacity);
                work_base += scores_qword_bytes;

                Candidates & filtered = *(Candidates*)(void*)work_base;
                filtered.init(filtered_capacity);
                work_base += filtered_qword_bytes;
                
                #pragma omp for schedule(dynamic)
                for (size_t q = 0; q < size_t(num_queries); q++) {
                    // Clear per-thread work buffers
                    scores.clear();
                    filtered.clear();
                
                    const uint8_t* q_ptr;
                    // Generate sketch from query
                    {
                        ScopedTimer timer(timing.search_sketch);
                        const size_t query_offset = dimensions * q;
                        const size_t end_offset __attribute__((unused)) = query_offset + dimensions;
                        assert(end_offset <= quantized_queries.size() && "Out of bounds access to quantized_queries");
                        q_ptr = quantized_queries.data() + query_offset;
                        ForestUtils::generate_sketch_from_quantized(q_ptr, query_sketch, dimensions, sketch_qword_bytes);
                    }
                    // Get pointer to quantized query
                    {
                        ScopedTimer timer(timing.search_pickup);
                        // Collect results from each tree for this query
                        assert(0 <= q && q < size_t(num_queries));
                        
                        // Maximum acceptable score (initially infinite)
                        int max_acceptable_score = dimensions;
                        int total_candidates = 0;
                        
                        for (int tree_id = 0; tree_id < used_trees; tree_id++) {
                            // Read results directly from tree_mem_base
                            int* tree_results = (int*)(tree_mem_base + tree_mem_qword_size * tree_id);
                            const int leafValue = tree_results[q];
                            // Parse leafValue to get index and flag
                            const int index = leafValue >> 1;
                            const bool flag = leafValue & 1;
                            
                            // Determine number of candidates (odd if odd flag, even if even flag)
                            const int candidates = flag ? odd_candidates : even_candidates;
                            assert(candidates > 0);
                            
                            // Calculate range of candidate indices
                            const int half_rangeL = candidates >> 1;
                            const int half_rangeR = (candidates+1) >> 1;
                            const int start_idx = std::max(0, index - half_rangeL);
                            const int end_idx = std::min(index + half_rangeR, num_points);
                            total_candidates += end_idx - start_idx;
                            
                            // Process each candidate
                            assert(0 <= tree_id && tree_id < int(trees.size()));
                            assert(trees[tree_id].tree != nullptr);
                            const uint8_t* tree_data = trees[tree_id].tree->getData();
                            size_t pointIdPos = trees[tree_id].tree->pointIdOffset + size_t(start_idx) * trees[tree_id].tree->pointIdBits;
                            for (int idx = start_idx; idx < end_idx; idx++, pointIdPos += trees[tree_id].tree->pointIdBits) {
                                assert(0 <= idx && idx < num_points);
                                // Get GRP_IDX from leaf node encoding
                                
                                assert(tree_data != nullptr);
                                size_t pre_idx = readBits(tree_data, pointIdPos, trees[tree_id].tree->pointIdBits);
                                assert(pre_idx < size_t(num_points));
                                
                                // Get data point sketch and calculate sketch distance
                                size_t offset = pre_idx * sketch_qword_bytes;
                                assert(offset + sketch_qword_bytes <= sketch_qword_bytes * num_points);
                                uint8_t* data_sketch = sketches_data + offset;
                                
                                // Use Hamming distance directly (smaller means higher similarity)
                                int distance = SketchUtils::hammingDistance(query_sketch, data_sketch, sketch_qword_bytes, max_acceptable_score);
        
                                // Only add if current score is less than max acceptable score
                                if (distance <= max_acceptable_score) {
                                    // Use distance directly as score (smaller is better)
                                    // Store PRE_IDX in scores
                                    scores.emplace_back(distance, pre_idx);
                                    
                                    // Perform early filtering if candidate count exceeds threshold
                                    if (scores.full()) {
                                        ScopedTimer sort_timer(timing.search_sort);
                                        
                                        // Sort all and filter
                                        std::sort(scores.begin(), scores.end());
                                        scores.resize(std::min((int)std::distance(scores.begin(), std::unique(scores.begin(), scores.end())), dist_candidates));
                                        
                                        // Update maximum acceptable score
                                        if(scores.size() == dist_candidates) {
                                            max_acceptable_score = std::get<0>(scores.back()) - 1;
                                        }
                                    }
                                }
                            }
                        }
                        total_candidates_per_query[q] += total_candidates;
                    }
                
                    {
                        ScopedTimer timer(timing.search_sort);

                        // Sort all and filter
                        std::sort(scores.begin(), scores.end());
                        scores.resize(std::min((int)std::distance(scores.begin(), std::unique(scores.begin(), scores.end())), dist_candidates));
                        filtered_candidates_per_query[q] = scores.size();
                    }
                
                    // Distance calculation for this query
                    {
                        ScopedTimer timer(timing.search_distance);
                    
                        // Process each candidate group
                        for (int j = 0; j < scores.size(); j++) {
                            // scores contains [score, PRE_IDX]
                            int base_pre_idx = std::get<1>(scores[j]);
                            assert(0 <= base_pre_idx && base_pre_idx < num_points);
                            
                            // [begin, end)
                            int begin = std::max(0, base_pre_idx - hops);
                            int end = std::min(base_pre_idx + 1 + hops, num_points);
                            
                            // Calculate distance for all points in group
                            for (int pre_idx = begin; pre_idx < end; ++pre_idx) {
                                
                                // Use quantized data directly (in PRE_IDX order)
                                uint8_t* point_ptr = points_data + size_t(dimensions) * pre_idx;
                            
                                // Calculate squared Euclidean distance
                                int distance2 = 0;
                                for (int d = 0; d < dimensions; d++) {
                                    int diff = (int)q_ptr[d] - (int)point_ptr[d];
                                    distance2 += diff * diff;
                                }
                                
                                // Use squared Euclidean distance as score (smaller is better)
                                filtered.emplace_back(distance2, pre_idx);
                            }
                        }
                    }
                
                    // Process final results
                    {
                        ScopedTimer timer(timing.search_topk);
                        // Sort by score (ascending distance)
                        std::sort(filtered.begin(), filtered.end());
                        
                        // Remove duplicates
                        filtered.erase_to_end(std::unique(filtered.begin(), filtered.end()));
                        
                        // Store k results (verify sufficient candidates)
                        assert(int(filtered.size()) >= k);
                        
                        for (int i = 0; i < k; i++) {
                            d_ptr(q, i) = std::get<0>(filtered[i]);
                            // Convert PRE_IDX to original ID and store
                            i_ptr(q, i) = id_mapping.preidx_to_id[std::get<1>(filtered[i])];
                        }
                    }
                    
                    // Progress bar update temporarily disabled
                    if (2<=verbose) {
                        #pragma omp critical(progress_bar)
                        {
                            ++progress_omp_counter;
                            if (thread_id==0) {
                                progress_bar.update(progress_omp_counter);
                                progress_omp_counter = 0;
                            }
                        }
                    }
                }
            }
            
            progress_bar.complete();
            
            // Merge timing information from each thread
            if (2<=verbose) {
                for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
                    timing.merge(*(ForestTiming*)(void*)(whole_base + (thread_id * work_size)));
                }
            }
            
            // Free thread memory
            free(work_memory);
        }
        
        if (2<=verbose) {
            float rate = 1.0 / num_threads;
            std::cerr << "  Sketch creation time: " << timing.search_sketch * rate << "s" << std::endl;
            std::cerr << "  Candidate extraction time: " << timing.search_pickup * rate << "s" << std::endl;
            std::cerr << "  Candidate sorting time: " << timing.search_sort * rate << "s" << std::endl;
            std::cerr << "  Final result selection time: " << timing.search_topk * rate << "s" << std::endl;
            std::cerr << "  Distance calculation time: " << timing.search_distance * rate << "s" << std::endl;

            // Candidate statistics
            double avg_total = 0.0, avg_filtered = 0.0;
            for (int q = 0; q < num_queries; q++) {
                avg_total += total_candidates_per_query[q];
                avg_filtered += filtered_candidates_per_query[q];
            }
            avg_total /= num_queries;
            avg_filtered /= num_queries;

            std::cerr << "Statistics:" << std::endl;
            std::cerr << "  Average candidates (all trees, including duplicates): " << avg_total << std::endl;
            std::cerr << "  Average filtered candidates: " << avg_filtered << std::endl;
            std::cerr << "  Filtering rate: " << avg_filtered / avg_total * 100.0 << "%" << std::endl;
        }
        
        free(tree_mem_memory);
        
        return std::make_tuple(D, I);
    }
    
    py::dict get_timing_info() const {
        return timing.get_timing_info();
    }
    
    // Get index properties
    py::dict get_properties() const {
        py::dict properties;
        properties["dimensions"] = dimensions;
        properties["bitDepth"] = bitDepth;
        properties["ntrees_total"] = ntrees_total;
        properties["ntrees"] = ntrees;
        properties["odd_candidates"] = odd_candidates;
        properties["even_candidates"] = even_candidates;
        properties["is_trained"] = is_trained;
        properties["leaf_size"] = leaf_size;
        
        // Add candidate count parameters
        properties["dist_candidates"] = dist_candidates;
        properties["hops"] = hops;
        
        // Also add timing information
        properties["timings"] = get_timing_info();
        
        return properties;
    }
    
    // Getters and setters for leaf size
    int get_leaf_size() const {
        return leaf_size;
    }
    
    void set_leaf_size(int size) {
        assert(size > 0);
        leaf_size = size;
    }
    
};

// Create index
HilbertForest create_index(const std::string& db_path = "db", int ntrees = 10, int leaf_size = 1, int verbose_level = 1, int seed = -1) {
    return HilbertForest(db_path, ntrees, leaf_size, verbose_level, seed);
}

PYBIND11_MODULE(hforest, m) {
    m.doc() = "Hilbert Forest: A spatial indexing library using Hilbert curves"; 
    
    py::class_<HilbertForest>(m, "HilbertForest")
        .def(py::init<const std::string&, int, int, int, int>(),
             py::arg("db_path") = "db",
             py::arg("ntrees") = 10,
             py::arg("leaf_size") = 1,
             py::arg("verbose") = 1,
             py::arg("seed") = -1)
        .def("preload", &HilbertForest::preload, 
             py::arg("data"),
             "Preload dataset and perform quantization and pre-Hilbert sorting")
        .def("fit", &HilbertForest::fit, 
             py::arg("data") = py::none(),
             "Build index. If data is not specified, use data preloaded with preload()")
        .def("search", &HilbertForest::search, 
             py::arg("queries"), 
             py::arg("k") = 10,
             "Search k nearest neighbors for query points")
        .def_property_readonly("is_trained", &HilbertForest::is_trained_getter)
        .def_property("ntrees", 
                     &HilbertForest::get_ntrees, 
                     &HilbertForest::set_ntrees)
        .def_property("odd_candidates",
                     &HilbertForest::get_odd_candidates,
                     &HilbertForest::set_odd_candidates)
        .def_property("even_candidates",
                     &HilbertForest::get_even_candidates,
                     &HilbertForest::set_even_candidates)
        .def_property("dist_candidates",
                     &HilbertForest::get_dist_candidates,
                     &HilbertForest::set_dist_candidates)
        .def_property("leaf_size",
                     &HilbertForest::get_leaf_size,
                     &HilbertForest::set_leaf_size)
        .def_property("hops",
                     &HilbertForest::get_hops,
                     &HilbertForest::set_hops)
        .def("get_properties", &HilbertForest::get_properties)
        .def("get_timing_info", &HilbertForest::get_timing_info, 
             "Get detailed timing information");
    
    m.def("create_index", &create_index,
          py::arg("db_path") = "db",
          py::arg("ntrees") = 10,
          py::arg("leaf_size") = 1,
          py::arg("verbose") = false,
          py::arg("seed") = -1,
          "Create new HilbertForest index");
}