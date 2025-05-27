CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O2 -march=native -fopenmp

# Build dependencies
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)

# Source and header files
SRC_DIR = src

# Build Python extension module
.PHONY: all clean

all: hforest 

# Header file list
HEADERS = $(SRC_DIR)/hsort.hpp $(SRC_DIR)/hsearch.hpp $(SRC_DIR)/htree.hpp \
          $(SRC_DIR)/utils.hpp $(SRC_DIR)/timing.hpp $(SRC_DIR)/tree_encoder.hpp \
          $(SRC_DIR)/forest_utils.hpp $(SRC_DIR)/progress_bar.hpp \
          $(SRC_DIR)/assert.hpp $(SRC_DIR)/fast_vector.hpp

# Define actual output filename (depends on Python version)
SO_FILE = $(shell python3 -c "import sysconfig; print('hforest' + sysconfig.get_config_var('EXT_SUFFIX'))")

# Build Python extension module
$(SO_FILE): $(SRC_DIR)/hforest.cpp $(HEADERS)
	rm -rf build
	rm -f *.so
	python3 setup.py build_ext --inplace

# Convenient alias
hforest: $(SO_FILE)

clean:
	rm -rf build
	rm -f *.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete