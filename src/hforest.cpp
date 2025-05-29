// Standard library headers
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// System headers (POSIX/Linux)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Third-party library headers
#include <immintrin.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "assert.hpp"
#include "fast_vector.hpp"
#include "utils.hpp"

using Candidate = std::tuple<int, int>;
using Candidates = fast_vector<Candidate>;

#include "timing.hpp"
#include "progress_bar.hpp"
#include "forest_utils.hpp"
#include "bit_quantize.hpp"
#include "htree.hpp"
#include "hsort.hpp"
#include "tree_encoder.hpp"
#include "hsearch.hpp"

// Bidirectional mapping between original IDs and pre-Hilbert sorted indices
struct IDMapping {
    std::vector<int> id_to_preidx;
    std::vector<int> preidx_to_id;
};

class HilbertForest {
private:
    int dimensions = -1;
    int bitDepth = 4;          // Bit depth for quantization (e.g., 3 for 3-bit, 8 for 8-bit)
    int ntrees_total;          // Total number of trees built during training
    int ntrees;                // Number of trees to use during search (can be <= ntrees_total)
    int odd_candidates = 11;   // Number of candidates for leaf nodes with odd size
    int even_candidates = 10;  // Number of candidates for leaf nodes with even size
    int dist_candidates = 100; // Number of candidate points for distance calculation
    int hops = 1;              // Search range for neighboring points (pre_idx ±hops)
    int leaf_size = 1;         // Stop splitting when node size falls below this
    bool is_trained = false;
    int verbose = 1;           // 0=silent, 1=normal, 2=verbose
    float quantization_rate = 500.0f;       // Quantization rate
    int laneCount;             // Number of values that can be read at once (57 / bitDepth)
    
    // Quantized data storage (in pre-Hilbert sort order)
    uint8_t* points_data = nullptr;
    void* points_memory = nullptr;
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
        
        if (points_memory != nullptr) {
            free(points_memory);
            points_memory = nullptr;
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
    HilbertForest(const std::string& db_path = "db", int ntrees = 10, int leaf_size = 1, int verbose_level = 1, int seed = -1, int bit_depth = 4)
        : bitDepth(bit_depth), ntrees_total(ntrees), ntrees(ntrees), leaf_size(leaf_size),
          verbose(verbose_level), timing(), dbPath(db_path) {
        assert(leaf_size > 0);
        assert(verbose_level >= 0 && verbose_level <= 2);
        assert(bit_depth > 0 && bit_depth <= 32);
        // Calculate how many values can be read at once
        laneCount = 57 / bit_depth;
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
            // Calculate quantization_rate once when dimensions are set
            // For bitDepth bits, the midpoint is 2^(bitDepth-1)
            float midpoint = (float)(1LL << (bitDepth - 1));
            quantization_rate = sqrt(sqrt(dimensions)) * midpoint;
        } else {
            assert(data_dimensions == dimensions && "Data dimensions do not match previously set dimensions");
        }
        
        assert(dimensions > 0);
        
        clearTrees();
        
        const int batch_size = 1000;

        this->num_points = num_points;
        
        // Prepare BitWriter for all data
        BitWriter dataWriter;
        
        
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
                    
                    // Transform batch to bit-packed format
                    BitQuantizer::transformToBits(ptr, dataWriter, current_batch, dimensions, quantization_rate, bitDepth);
                    
                    // Add indices for all points in batch
                    for (int i = 0; i < current_batch; i++) {
                        all_points.push_back(start_idx + i);
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
        
        // Finalize bit-packed data
        dataWriter.finalizeBuffer();
        
        // Create temporary buffer for bit-packed data
        size_t temp_data_size = dataWriter.size();
        uint8_t* temp_points_data = (uint8_t*)malloc(temp_data_size);
        assert(temp_points_data != nullptr && "Failed to allocate memory for temporary points data");
        std::memcpy(temp_points_data, dataWriter.data(), temp_data_size);
        
        // Pre-Hilbert sort
        {
            ProgressBar progress_bar(1, "Pre-Hilbert sort", verbose);
            std::vector<int> pre_hilbert_axes(dimensions);
            for (int i = 0; i < dimensions; i++) {
                pre_hilbert_axes[i] = i;
            }
            HilbertSort pre_sorter(pre_hilbert_axes, bitDepth, rng, temp_points_data, dimensions, 1, laneCount);
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
        // For bit-packed data, we need to reorder the data
        BitWriter sortedDataWriter;
        
        // Process points in PRE_IDX order
        for (int pre_idx = 0; pre_idx < num_points; pre_idx++) {
            int original_id = id_mapping.preidx_to_id[pre_idx];
            
            // Copy bit-packed data for this point
            for (int d = 0; d < dimensions; d++) {
                size_t bitPos = BitQuantizer::calculateBitPosition(original_id, d, dimensions, bitDepth);
                uint64_t value = readBits(temp_points_data, bitPos, bitDepth);
                sortedDataWriter.writeBits(value, bitDepth);
            }
        }
        
        // Finalize sorted data
        sortedDataWriter.finalizeBuffer();
        size_t sorted_data_size = sortedDataWriter.size();
        points_data = (uint8_t*)malloc(sorted_data_size);
        assert(points_data != nullptr && "Failed to allocate memory for points data");
        std::memcpy(points_data, sortedDataWriter.data(), sorted_data_size);
        
        // Release temporary buffer
        free(temp_points_data);
        
        // Generate sketches from quantized data (stored in PRE_IDX order)
        {
            std::stringstream ss;
            int sketch_bytes = (dimensions + 7) >> 3;
            ss << "Generating sketches (" << sketch_bytes << " bytes/point, total: " << sketch_bytes * num_points << " bytes)";
            ProgressBar progress_bar(1, ss.str(), verbose);

            // Generate sketches for entire dataset
            size_t sketch_qword_bytes = (size_t((dimensions + 7) >> 3) + 7) & ~7;
            // Add extra 8 bytes at the end for safe batch reading with readBits
            sketches_memory = malloc(sketch_qword_bytes * num_points + 16);
            assert(sketches_memory != nullptr && "Failed to allocate memory for sketches");
            sketches_data = (uint8_t*)(void*)(((size_t)sketches_memory + 7) & ~7);
            
            // Generate sketches from bit-packed data
            #pragma omp parallel for
            for (int i = 0; i < num_points; i++) {
                uint8_t* sketch_ptr = sketches_data + (size_t)i * sketch_qword_bytes;
                BitQuantizer::generateSketchFromBitPacked(points_data, sketch_ptr, i, dimensions, bitDepth, sketch_qword_bytes);
            }

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
                HilbertSort sorter(shuffled_axes, bitDepth, rng, points_data, dimensions, leaf_size, laneCount);
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
        
        // Compress database points to save memory (bitDepth -> bitDepth-1)
        // MSB information is stored in sketches
        {
            ProgressBar progress_bar(1, "Compressing database points", verbose);
            
            BitWriter compressedWriter;
            BitQuantizer::compressBitPackedData(points_data, compressedWriter, num_points, dimensions, bitDepth);
            compressedWriter.finalizeBuffer();
            
            // Replace original points_data with compressed version
            free(points_memory);
            size_t compressed_size = compressedWriter.size();
            points_memory = malloc(compressed_size + 8);
            assert(points_memory != nullptr && "Failed to allocate memory for compressed points");
            points_data = (uint8_t*)(void*)(((size_t)points_memory + 7) & ~7);
            std::memcpy(points_data, compressedWriter.data(), compressed_size);
            
            progress_bar.complete();
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
        
        int total_queries = buf.shape[0];
        if(verbose) std::cerr << "Searching: queries=" << total_queries << ", k=" << k << std::endl;

        // Create result arrays
        py::array_t<float> D(py::array::ShapeContainer({total_queries, k}));
        py::array_t<int> I(py::array::ShapeContainer({total_queries, k}));
        
        auto d_ptr = D.mutable_unchecked<2>();
        auto i_ptr = I.mutable_unchecked<2>();
        
        // Number of trees to actually use (smaller of ntrees and ntrees_total)
        int used_trees = std::min(ntrees, ntrees_total);
        
        // Batch processing
        const int batch_size = 12032;
        const int max_batch_queries = std::min(batch_size, total_queries);
        
        // Pre-allocate memory for largest batch
        int num_threads = omp_get_max_threads();
        if (verbose) std::cerr << "OpenMP threads: " << num_threads << std::endl;
        
        // Create batch progress bar if multiple batches
        ProgressBar batch_progress_bar(total_queries, "Processing batches", batch_size < total_queries ? this->verbose : 0);
        
        // Suppress verbose output within batch if multiple batches
        int verbose = batch_size < total_queries ? 0 : this->verbose;
        
        size_t cache_line_bits = 6;  // 2^6 = 64
        size_t cache_line_size = 1 << cache_line_bits;
        size_t cache_line_mask = cache_line_size - 1;
        
        size_t block_size = 65536;
        size_t block_mask = block_size - 1;
        
        // Pre-allocate quantized queries buffer
        size_t max_query_data_size = ((size_t)max_batch_queries * dimensions * bitDepth + 7) >> 3;
        uint8_t* quantized_queries = (uint8_t*)malloc(max_query_data_size + 8);
        assert(quantized_queries != nullptr && "Failed to allocate memory for quantized queries");
        
        // Pre-allocate query sketches memory
        size_t sketch_bytes = (dimensions + 7) >> 3;
        size_t sketch_qword_bytes = (sketch_bytes + 7) & ~7;
        size_t sketch_mem_size = sketch_qword_bytes * max_batch_queries;
        size_t sketch_mem_aligned_size = (sketch_mem_size + cache_line_mask) & ~cache_line_mask;
        void* query_sketches_memory = malloc(sketch_mem_aligned_size + cache_line_size);
        assert(query_sketches_memory != nullptr && "Failed to allocate memory for query sketches");
        uint8_t* query_sketches = (uint8_t*)((size_t(query_sketches_memory) + cache_line_mask) & ~cache_line_mask);
        
        // Pre-allocate tree work memory
        size_t work_size2 = 0;
        
        // Memory for per-query rankings (each thread needs rankings for all queries)
        // Layout: [acceptable_score (int)] [Candidates]
        size_t acceptable_score_bytes = sizeof(int);
        size_t acceptable_score_qword_bytes = (acceptable_score_bytes + 7) & ~7;
        
        size_t per_query_ranking_capacity = dist_candidates * 4;
        size_t per_query_ranking_bytes = sizeof(Candidates) + sizeof(Candidate) * per_query_ranking_capacity;
        size_t per_query_ranking_qword_bytes = (per_query_ranking_bytes + 7) & ~7;
        
        size_t per_query_total_bytes = acceptable_score_qword_bytes + per_query_ranking_qword_bytes;
        size_t all_query_rankings_bytes = per_query_total_bytes * max_batch_queries;
        work_size2 += all_query_rankings_bytes;

        // Memory for index array (placed after rankings)
        size_t indices_bytes = sizeof(int) * max_batch_queries;
        size_t indices_qword_bytes = (indices_bytes + 7) & ~7;
        work_size2 += indices_qword_bytes;

        work_size2 = (work_size2 + block_mask) & ~block_mask;
        uint8_t * tree_work_memory = (uint8_t *)malloc(work_size2 * num_threads + block_size);
        assert(tree_work_memory != nullptr && "Failed to allocate memory for thread work");
        uint8_t * tree_whole_base = (uint8_t *)((size_t(tree_work_memory) + block_mask) & ~block_mask);
        
        // Pre-allocate query processing work memory
        size_t work_size = 0;

        // Timing must always be placed first
        size_t timing_bytes = sizeof(ForestTiming);
        size_t timing_qword_bytes = (timing_bytes + 7) & ~7;
        work_size += timing_qword_bytes;

        // Set capacity to handle merged rankings from all threads
        size_t scores_capacity = dist_candidates * 4 * num_threads;
        size_t scores_bytes = sizeof(Candidates) + sizeof(Candidate) * scores_capacity;
        size_t scores_qword_bytes = (scores_bytes + 7) & ~7;
        work_size += scores_qword_bytes;

        size_t filtered_capacity = dist_candidates * (1 + hops + hops);
        size_t filtered_bytes = sizeof(Candidates) + sizeof(Candidate) * filtered_capacity;
        size_t filtered_qword_bytes = (filtered_bytes + 7) & ~7;
        work_size += filtered_qword_bytes;

        work_size = (work_size + block_mask) & ~block_mask;
        uint8_t * work_memory = (uint8_t *)malloc(work_size * num_threads + block_size);
        assert(work_memory != nullptr && "Failed to allocate memory for query processing");
        uint8_t * whole_base = (uint8_t *)((size_t(work_memory) + block_mask) & ~block_mask);
        
        for (int batch_start = 0; batch_start < total_queries; batch_start += batch_size) {
            int batch_end = std::min(batch_start + batch_size, total_queries);
            int num_queries = batch_end - batch_start;
        
            // Array to store filtered candidate counts for statistics
            std::vector<int> filtered_candidates_per_query(num_queries, 0);
            
            // Quantize all queries once in advance
            BitWriter queryWriter;
            {
                ProgressBar progress_bar(1, "Quantizing query points", verbose);
                
                float* query_ptr = (float*)buf.ptr + (size_t)batch_start * dimensions;
                
                BitQuantizer::transformToBits(query_ptr, queryWriter, num_queries, dimensions, quantization_rate, bitDepth);
                queryWriter.finalizeBuffer();
                
                progress_bar.complete();
            }
            
            // Copy quantized queries to pre-allocated buffer
            size_t query_data_size = queryWriter.size();
            assert(query_data_size <= max_query_data_size && "Query data size exceeds pre-allocated buffer");
            std::memcpy(quantized_queries, queryWriter.data(), query_data_size);
            
            {
                ProgressBar progress_bar(1, "Generating query sketches", verbose);
                
                // Process in cache_line_size chunks for better cache locality
                int num_chunks = (num_queries + cache_line_mask) >> cache_line_bits;
                
                #pragma omp parallel for schedule(static)
                for (int chunk = 0; chunk < num_chunks; chunk++) {
                    int q_start = chunk << cache_line_bits;
                    int q_end = std::min(q_start + (int)cache_line_size, num_queries);
                    uint8_t* sketch_ptr = query_sketches + (size_t)q_start * sketch_qword_bytes;
                    
                    for (int q = q_start; q < q_end; q++) {
                        BitQuantizer::generateSketchFromBitPacked(quantized_queries, sketch_ptr, q, dimensions, bitDepth, sketch_qword_bytes);
                        sketch_ptr += sketch_qword_bytes;
                    }
                }
                
                progress_bar.complete();
            }


            // Create progress bar for tree processing
            ProgressBar tree_progress_bar(used_trees, "Processing trees", verbose);
            {
                int progress_omp_counter = 0;
                
                // Query data is read-only so can be shared
                uint8_t * queries = quantized_queries;
                
                // Parallel processing with OpenMP
                #pragma omp parallel num_threads(num_threads)
                {
                    int thread_id = omp_get_thread_num();
                    assert(0 <= thread_id && thread_id < num_threads);

                    uint8_t * work_base2 = tree_whole_base + work_size2 * thread_id;
                    
                    // Initialize per-query rankings and acceptable scores
                    uint8_t* query_rankings_base = work_base2;
                    for (int i = 0; i < num_queries; i++) {
                        uint8_t* query_data = query_rankings_base + per_query_total_bytes * i;
                        
                        // Initialize acceptable score
                        int* acceptable_score = (int*)(void*)query_data;
                        *acceptable_score = dimensions;  // Initially accept all distances
                        
                        // Initialize ranking
                        Candidates& ranking = *(Candidates*)(void*)(query_data + acceptable_score_qword_bytes);
                        ranking.init(per_query_ranking_capacity);
                        ranking.clear();
                    }
                    work_base2 += all_query_rankings_bytes;

                    // Initialize index array (after rankings)
                    int* query_indices = (int*)(void*)work_base2;
                    for (int i = 0; i < num_queries; i++) {
                        query_indices[i] = i;
                    }
                    work_base2 += indices_qword_bytes;

                    // Batch process for each tree
                    #pragma omp for schedule(dynamic)
                    for (int t = 0; t < used_trees; t++) {
                        // Search all at once using HilbertTreeSearch
                        HilbertTreeSearch searcher(trees[t].tree);
                        // Calculate query bit count
                        int totalQueryBits = dimensions * bitDepth;
                        searcher.search(queries, num_queries, totalQueryBits, query_indices,
                                    query_sketches, sketches_data, sketch_qword_bytes,
                                    query_rankings_base, per_query_total_bytes, acceptable_score_qword_bytes,
                                    even_candidates, odd_candidates, dist_candidates, num_points, dimensions);
                        
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
            }
            tree_progress_bar.complete();
            
            // Process each query in parallel
            {
                ProgressBar progress_bar(num_queries, "Processing queries", verbose);
                int progress_omp_counter = 0;
                
                
                // Parallel processing with OpenMP
                #pragma omp parallel num_threads(num_threads)
                {
                    int thread_id = omp_get_thread_num();
                    assert(0 <= thread_id && thread_id < num_threads);

                    uint8_t * work_base = whole_base + work_size * thread_id;

                    ForestTiming & timing = *(ForestTiming*)(void*)work_base;
                    timing.reset();
                    work_base += timing_qword_bytes;

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
                    
                        {
                            ScopedTimer timer(timing.search_pickup);
                            // Merge rankings from all threads for this query
                            assert(0 <= q && q < size_t(num_queries));
                            
                            // Merge rankings from all threads
                            for (int tid = 0; tid < num_threads; tid++) {
                                // Get thread's ranking for this query
                                uint8_t* thread_work_base = tree_whole_base + work_size2 * tid;
                                uint8_t* thread_query_data = thread_work_base + per_query_total_bytes * q;
                                Candidates& thread_ranking = *(Candidates*)(void*)(thread_query_data + acceptable_score_qword_bytes);
                                
                                // Copy all candidates from thread's ranking to scores
                                for (int i = 0; i < thread_ranking.size(); i++) {
                                    scores.emplace_back(thread_ranking[i]);
                                }
                            }
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
                                    // Get sketch for this point
                                    size_t sketch_stride = (size_t((dimensions + 7) >> 3) + 7) & ~7;
                                    uint8_t* point_sketch = sketches_data + (size_t)pre_idx * sketch_stride;
                                    
                                    // Calculate squared distance using compressed data + sketch
                                    int distance2 = BitQuantizer::calculateSquaredDistanceCompressed(
                                        points_data, pre_idx, point_sketch, sketch_stride,
                                        quantized_queries, q, dimensions, bitDepth, laneCount);
                                    
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
                                d_ptr(batch_start + q, i) = std::get<0>(filtered[i]);
                                // Convert PRE_IDX to original ID and store
                                i_ptr(batch_start + q, i) = id_mapping.preidx_to_id[std::get<1>(filtered[i])];
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
            }
            if (2<=verbose) {
                float rate = 1.0 / num_threads;
                std::cerr << "  Candidate extraction time: " << timing.search_pickup * rate << "s" << std::endl;
                std::cerr << "  Candidate sorting time: " << timing.search_sort * rate << "s" << std::endl;
                std::cerr << "  Final result selection time: " << timing.search_topk * rate << "s" << std::endl;
                std::cerr << "  Distance calculation time: " << timing.search_distance * rate << "s" << std::endl;

                // Candidate statistics
                double avg_filtered = 0.0;
                for (int q = 0; q < num_queries; q++) {
                    avg_filtered += filtered_candidates_per_query[q];
                }
                avg_filtered /= num_queries;

                std::cerr << "Statistics:" << std::endl;
                std::cerr << "  Average filtered candidates: " << avg_filtered << std::endl;
            }
            
            batch_progress_bar.update(num_queries);
        }
        
        batch_progress_bar.complete();
        
        // Free all pre-allocated memory
        free(work_memory);
        free(tree_work_memory);
        free(quantized_queries);
        free(query_sketches_memory);
        
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
HilbertForest create_index(const std::string& db_path = "db", int ntrees = 10, int leaf_size = 1, int verbose_level = 1, int seed = -1, int bit_depth = 4) {
    return HilbertForest(db_path, ntrees, leaf_size, verbose_level, seed, bit_depth);
}

PYBIND11_MODULE(hforest, m) {
    m.doc() = "Hilbert Forest: A spatial indexing library using Hilbert curves"; 
    
    py::class_<HilbertForest>(m, "HilbertForest")
        .def(py::init<const std::string&, int, int, int, int, int>(),
             py::arg("db_path") = "db",
             py::arg("ntrees") = 10,
             py::arg("leaf_size") = 1,
             py::arg("verbose") = 1,
             py::arg("seed") = -1,
             py::arg("bit_depth") = 4)
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
          py::arg("bit_depth") = 4,
          "Create new HilbertForest index");
}