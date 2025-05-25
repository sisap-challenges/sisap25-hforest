#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Class to hold timing information
class ForestTiming {
public:
    ForestTiming() {
        reset();
    }

    void reset() {
        // fit related
        fit_prepare = 0.0;
        fit_sort = 0.0;
        fit_encode = 0.0;
        fit_sketch = 0.0;
        
        // search related
        search_pickup = 0.0;
        search_sort = 0.0;
        search_distance = 0.0;
        search_topk = 0.0;
        search_sketch = 0.0;
    }

    // Merge timing information from other
    void merge(const ForestTiming& other) {
        // fit related
        fit_prepare += other.fit_prepare;
        fit_sort += other.fit_sort;
        fit_encode += other.fit_encode;
        fit_sketch += other.fit_sketch;
        
        // search related
        search_pickup += other.search_pickup;
        search_sort += other.search_sort;
        search_distance += other.search_distance;
        search_topk += other.search_topk;
        search_sketch += other.search_sketch;
    }
    
    // Get timing information as dictionary for Python
    py::dict get_timing_info() const {
        py::dict timing;
        
        // fit related
        timing["fit_prepare"] = fit_prepare;
        timing["fit_sort"] = fit_sort;
        timing["fit_encode"] = fit_encode;
        timing["fit_sketch"] = fit_sketch;
        
        // search related
        timing["search_pickup"] = search_pickup;
        timing["search_sort"] = search_sort;
        timing["search_distance"] = search_distance;
        timing["search_topk"] = search_topk;
        timing["search_sketch"] = search_sketch;
        
        return timing;
    }
    
    // fit related timing
    double fit_prepare = 0.0;           // Data preparation and quantization time
    double fit_sort = 0.0;              // Hilbert sort time
    double fit_encode = 0.0;            // Tree encoding time
    double fit_sketch = 0.0;            // Sketch generation time
    
    // search related timing
    double search_pickup = 0.0;         // Candidate pickup time
    double search_sort = 0.0;           // Candidate sorting time
    double search_distance = 0.0;       // Distance calculation time
    double search_topk = 0.0;           // Final top-k extraction time
    double search_sketch = 0.0;         // Sketch generation and distance calculation time
};

// Timer class - starts timing in constructor, automatically stores result in specified variable in destructor
class ScopedTimer {
public:
    ScopedTimer(double& time_var) 
        : time_ptr(&time_var), start_time(std::chrono::high_resolution_clock::now()) {}
    
    void stop() {
        if(time_ptr) {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            *time_ptr += elapsed.count();
            time_ptr = NULL;
        }
    }
    ~ScopedTimer() {
        stop();
    }

private:
    double * time_ptr;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif // TIMING_HPP