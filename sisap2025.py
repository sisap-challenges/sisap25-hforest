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


def run(task, verbose_level, args):
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
            (160, 2024, 2025, 450, 2),
            (160, 2024, 2025, 440, 2), # 77.0294% 18.785sec
            (160, 1920, 1921, 430, 2),
            (160, 1920, 1921, 420, 2), # 76.0097% 17.005sec
            (160, 1740, 1741, 410, 2),
            (160, 1740, 1741, 400, 2), # 75.0791% 15.710sec
            (160, 1580, 1581, 390, 2),
            (160, 1580, 1581, 380, 2), # 74.1115% 14.706sec
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
        k = 15
        bit_depth = 8
        
        # Load data (task2 loads from two files)
        f_data = h5py.File('data/benchmark-dev-gooaq.h5', 'r')
        queries = data = np.array(f_data['train'])
        
       # f_gt = h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r')
       # gt_I = np.array(f_gt['knns'])
       # f_gt.close()

       # print(f"Processing ground truth: removing self-references and adjusting to k={k}")
       # gt_process_start = time.time()
       # processed_gt = []
       # for i in range(gt_I.shape[0]):
       #     # Create new result row without self-reference
       #     nearest_neighbors = []
       #     for j in gt_I[i]:
       #         if j != i+1:  # Skip self (1-indexed)
       #             nearest_neighbors.append(j)
       #             if len(nearest_neighbors) >= k:
       #                 break
       #     processed_gt.append(nearest_neighbors)
       # gt_I = np.array(processed_gt)
       # gt_process_end = time.time()
       # print(f"Ground truth processing completed in {gt_process_end - gt_process_start:.3f} seconds")

        leaf_size = 10
        # SISAP 2025 Python Example's 81.25% == Problem's 80.00%
        # Note: Our implementation correctly handles self-exclusion with the modified ground truth
        hyper_params_dict = {
            'task2': (80, 96, 97, 60, 0), # 81.7343%(80.7956%) 51.279sec
            'task2:85': (112, 106, 107, 75, 0), # 86.4794%(85.5781%) 74.1814sec
            'task2:90': (160, 130, 131, 100, 0), # 91.1381%(90.5473%) 116.435sec
            'task2:95': (280, 168, 169, 150, 0), # 95.8289%(95.5509%) 233.765sec
            'task2:98': (720, 170, 171, 300, 0), # 98.6168%(98.5246%) 611.313sec
        }
        if task in hyper_params_dict:
            hyper_params = [hyper_params_dict[task]]
        else:
            hyper_params = [hyper_params_dict[key] for key in hyper_params_dict]
    ntrees = max(hyper_param[0] for hyper_param in hyper_params)

    # Create index
    print("Creating index...")
    index = hforest.create_index(ntrees=ntrees, leaf_size=leaf_size, verbose=verbose_level, bit_depth=bit_depth)
    
    # Build index
    start_time = time.time()
    fitted = False
    
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
        fitted = True
        build_time = time.time() - build_start
    elif task == 'task2' or task[:6] == 'task2:':
        build_time = 0
    else:
        # Build index with normal fit
        index.fit(data)
        fitted = True
        build_time = time.time() - start_time

        # Close file
        f_data.close()
    
    print(f"Index built in {build_time}s")
    
    # Accept user input in interactive mode only if the interactive flag is set
    if args.interactive and sys.stdin.isatty():
        hyper_params = gen_hyper_params(ntrees if fitted else 10000, hyper_params[-1])
    for search_trees, even, odd, dist, hops in hyper_params:
        print(f"Starting search on {queries.shape} with ntrees={search_trees}, bitDepth={bit_depth}")
        start = time.time()
        index.ntrees = search_trees     # Number of trees to use
        index.even_candidates = even    # Candidates for even-level nodes
        index.odd_candidates = odd      # Candidates for odd-level nodes
        index.dist_candidates = dist    # Number of candidates for distance calculation
        index.hops = hops               # Search range for neighboring points (pre_idx ±hops)
        if fitted:
            if task[:5] == 'task2':
                D, I = index.search(queries, k + 1)
                I += (I == np.arange(I.shape[0])[:, None]) * 1000000000
                I.sort()
                I = I[:, :k]
            else:
                D, I = index.search(queries, k)
        else:
            D, I = index.graph(queries, k, include_self=False)
        I = I + 1 # Convert from 0-indexed to 1-indexed to match groundtruth
        # The +1 conversion should be included in contest timing measurement
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        identifier = f"index=(ntrees={ntrees},leaf_size={leaf_size}),query=(ntrees={search_trees},even={even},odd={odd},dist={dist},hops={hops})"
        store_results(os.path.join("results/", dataset, task, f"hforest_{identifier}.h5"), 
                     "hforest", dataset, task, D, I, build_time, elapsed_search, identifier)

        #recall = get_recall(I, gt_I, k)
        #print(f"Recall: {recall * 100.0}%")
        
        print(f"search_ntrees={search_trees}, even={even}, odd={odd}, dist={dist}, hops={hops}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HilbertForest approximate nearest neighbor search example')
    parser.add_argument('task', choices=['task1', 'task2', 'task1wf', 'task2old', 'task2:85', 'task2:90', 'task2:95', 'task2:98'],
                        help='Task type to execute (task1, task2, task1wf, task2old, task2:85, task2:90, task2:95, task2:98)')
    
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
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode to input hyperparameters'
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
    
    run(args.task, verbose_level, args)
