#!/usr/bin/env python3
import argparse
import hforest
import h5py
import numpy as np
import time
import os
import sys
from pathlib import Path

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def get_recall(I, gt, k):
    """
    Calculate k-NN recall rate
    """
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)

def gen_hyper_params(max_ntrees, example):
    while True:
        print(f"input ntrees, even, odd, dist, hops: (example ... {max_ntrees} {example[1]} {example[2]} {example[3]} {example[4]})")
        values = input().split()
        if len(values) >= 5:
            try:
                values = tuple(int(value) for value in values[:5])
                if 1 <= values[0] <= max_ntrees and 1 <= values[1] and 1 <= values[2] and 1 <= values[3] and 0 <= values[4]:
                    yield values
            except:
                pass


def run(task, verbose_level=1):
    """
    Run search with specified task
    """
    print(f'Running {task}')
    
    # Set dataset and parameters for each task
    if task[:5] == 'task1':
        dataset = 'pubmed23'
        k = 30
        bit_depth = 4
        
        # Load data (task1 loads all from the same file)
        f_data = h5py.File('data/benchmark-dev-pubmed23.h5', 'r')
        data = f_data['train'] # Cannot use np.array() - dataset too large for memory
        queries = np.array(f_data['otest']['queries'])
        gt_I = np.array(f_data['otest']['knns'])

        leaf_size = 100
        hyper_params = [
            (160, 1420, 1421, 370, 2),
            (160, 1420, 1421, 360, 2), # 73.0064% 12.968sec
            (160, 1300, 1301, 350, 2),
            (160, 1300, 1301, 340, 2), # 72.0154% 12.087sec
            (160, 1200, 1201, 330, 2),
            (160, 1200, 1201, 320, 2), # 71.0912% 11.341sec
            (160, 1100, 1101, 310, 2),
            (160, 1100, 1101, 300, 2), # 70.0600% 10.504sec
        ]
        
    elif task[:5] == 'task2':
        dataset = 'gooaq'
        k = 16
        bit_depth = 8
        
        # Load data (task2 loads from two files)
        f_data = h5py.File('data/benchmark-dev-gooaq.h5', 'r')
        queries = data = np.array(f_data['train'])
        
        f_gt = h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r')
        gt_I = np.array(f_gt['knns'])
        f_gt.close()

        leaf_size = 10
        hyper_params = [
            (340, 6, 7, 60, 0), # 80.8418% 109.721sec
            (400, 6, 7, 60, 0), # 82.7257% 126.354sec
            (450, 6, 7, 60, 0), # 83.9927% 142.453sec
            (500, 6, 7, 60, 0), # 85.0380% 157.165sec
            (550, 6, 7, 60, 0), # 85.9742% 171.756sec
            (600, 6, 7, 60, 0), # 86.7744% 185.8142sec
        ]
        if task == 'task2':
            hyper_params = hyper_params[:1]
    ntrees = max(hyper_param[0] for hyper_param in hyper_params)

    # Create index
    print("Creating index...")
    index = hforest.create_index(db_path=task[:5], ntrees=ntrees, leaf_size=leaf_size, verbose=verbose_level, bit_depth=bit_depth)
    
    # Build index
    start_time = time.time()
    
    if task == 'task1wf':
        # Use preload feature for task1wf
        print("Using preload feature...")
        index.preload(data)
        print(f"Data preloaded in {time.time() - start_time:.2f}s")

        # Close file
        f_data.close()
        os.remove('data/benchmark-dev-pubmed23.h5')

        # Simulate data release here
        print("Data source freed, now building index...")
        build_start = time.time()
        index.fit()  # Use preloaded data
        build_time = time.time() - build_start
    elif task == 'task2':
        build_time = 0
    else:
        # Build index with normal fit
        index.fit(data)
        build_time = time.time() - start_time

        # Close file
        f_data.close()
    
    print(f"Index built in {build_time}s")
    
    # Accept user input in interactive mode
    if sys.stdin.isatty():
        hyper_params = gen_hyper_params(ntrees if task != 'task2' else 10000, hyper_params[-1])
    for search_trees, even, odd, dist, hops in hyper_params:
        print(f"Starting search on {queries.shape} with ntrees={search_trees}, bitDepth={bit_depth}")
        start = time.time()
        index.ntrees = search_trees     # Number of trees to use
        index.even_candidates = even    # Candidates for even-level nodes
        index.odd_candidates = odd      # Candidates for odd-level nodes
        index.dist_candidates = dist    # Number of candidates for distance calculation
        index.hops = hops               # Search range for neighboring points (pre_idx ±hops)
        if task == 'task2':
            D, I = index.graph(queries, k)
        else:
            D, I = index.search(queries, k)
        if task[:5] == 'task2':
            pass
            #I += (I == np.arange(I.shape[0])[:, None]) * 1000000000
            #I.sort()
            #I = I[:, :k-1]
        I = I + 1 # Convert from 0-indexed to 1-indexed to match groundtruth
        # The +1 conversion should be included in contest timing measurement
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        identifier = f"index=(ntrees={ntrees},leaf_size={leaf_size}),query=(ntrees={search_trees},even={even},odd={odd},dist={dist},hops={hops})"
        store_results(os.path.join("results/", dataset, task, f"hforest_{identifier}.h5"), 
                     "hforest", dataset, task, D, I, build_time, elapsed_search, identifier)

        recall = get_recall(I, gt_I, k)
        print(f"Recall: {recall * 100.0}%")
        print(f"search_ntrees={search_trees}, even={even}, odd={odd}, dist={dist}, hops={hops}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HilbertForest approximate nearest neighbor search example')
    parser.add_argument('task', choices=['task1', 'task2', 'task1wf', 'task2old'],
                        help='Task type to execute (task1, task2, task1wf, task2old)')
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress (verbose_level=2)'
    )
    parser.add_argument(
        '--concise',
        action='store_true',
        help='Show standard progress (verbose_level=1)'
    )
    parser.add_argument(
        '--silent',
        action='store_true',
        help='Show minimal output only (verbose_level=0)'
    )
    
    args = parser.parse_args()
    
    # Set verbose level with priority: silent > verbose > concise
    verbose_level = 0  # Default value (silent)
    if args.silent:
        verbose_level = 0  # Highest priority
    elif args.verbose:
        verbose_level = 2  # Second priority
    elif args.concise:
        verbose_level = 1  # Lowest priority
    
    run(args.task, verbose_level)