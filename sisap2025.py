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
        need_self_loop_removal = False
        
        # Load data (task1 loads all from the same file)
        f_data = h5py.File('data/benchmark-dev-pubmed23.h5', 'r')
        data = f_data['train'] # Cannot use np.array() - dataset too large for memory
        queries = np.array(f_data['otest']['queries'])
        gt_I = np.array(f_data['otest']['knns'])

        ntrees = 400
        leaf_size = 10
        hyper_params = [
            (ntrees, 400, 401, 301, 2),
            (ntrees, 400, 401, 300, 2),
            (ntrees, 240, 241, 301, 2),
            (ntrees, 240, 241, 300, 2),
            (200, 840, 841, 301, 2),
            (200, 840, 841, 300, 2),
            (100, 3000, 3001, 301, 2),
            (100, 3000, 3001, 300, 2),
        ]
        
    elif task[:5] == 'task2':
        dataset = 'gooaq'
        k = 16
        need_self_loop_removal = True
        
        # Load data (task2 loads from two files)
        f_data = h5py.File('data/benchmark-dev-gooaq.h5', 'r')
        queries = data = np.array(f_data['train'])
        
        f_gt = h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r')
        gt_I = np.array(f_gt['knns'])
        f_gt.close()

        ntrees = 340
        leaf_size = 10
        hyper_params = [
            (ntrees, 12, 13, 61, 0),
            (ntrees, 12, 13, 60, 0),
            (ntrees, 10, 11, 61, 0),
            (ntrees, 10, 11, 60, 0),
            (ntrees, 8, 9, 61, 0),
            (ntrees, 8, 9, 60, 0),
            (ntrees, 6, 7, 61, 0),
            (ntrees, 6, 7, 60, 0)
        ]

    # Create index
    print("Creating index...")
    index = hforest.create_index(db_path=task[:5], ntrees=ntrees, leaf_size=leaf_size, verbose=verbose_level)
    
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
    else:
        # Build index with normal fit
        index.fit(data)
        build_time = time.time() - start_time

        # Close file
        f_data.close()
    
    print(f"Index built in {build_time}s")
    
    # Accept user input in interactive mode
    if sys.stdin.isatty():
        hyper_params = gen_hyper_params(ntrees, hyper_params[-1])
    for search_trees, even, odd, dist, hops in hyper_params:
        print(f"Starting search on {queries.shape} with ntrees={search_trees}")
        start = time.time()
        index.ntrees = search_trees     # Number of trees to use
        index.even_candidates = even    # Candidates for even-level nodes
        index.odd_candidates = odd      # Candidates for odd-level nodes
        index.dist_candidates = dist    # Number of candidates for distance calculation
        index.hops = hops               # Search range for neighboring points (pre_idx ±hops)
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
    parser.add_argument('task', choices=['task1', 'task2', 'task1wf'],
                        help='Task type to execute (task1, task2, task1wf)')
    
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress (verbose_level=2)'
    )
    verbosity_group.add_argument(
        '--silent',
        action='store_true',
        help='Show minimal output only (verbose_level=0)'
    )
    
    args = parser.parse_args()
    
    # Set verbose level: silent=0, normal=1, verbose=2
    verbose_level = 1  # Default value
    if args.verbose:
        verbose_level = 2
    elif args.silent:
        verbose_level = 0
    
    run(args.task, verbose_level)