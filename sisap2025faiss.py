#!/usr/bin/env python3
import argparse
import faiss
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
    Calculate recall rate for k-NN results
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


def run(task):
    """
    Execute search with specified task
    """
    print(f'Running {task}')
    
    # Configure dataset and parameters for each task
    if task[:5] == 'task1':
        dataset = 'pubmed23'
        k = 30
        need_self_loop_removal = False
        
        # Load data (task1 loads everything from the same file)
        f_data = h5py.File('data/benchmark-dev-pubmed23.h5', 'r')
        data = f_data['train'] # Cannot use np.array() - dataset too large for memory
        queries = np.array(f_data['otest']['queries'])
        gt_I = np.array(f_data['otest']['knns'])
        
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

    # Create index
    print("Creating index...")
    index = faiss.index_factory(queries.shape[1], f"IVF1024,SQfp16")
    
    # Build index
    start_time = time.time()
    
    # Build index using standard fit method
    index.train(data)
    index.add(data)
    build_time = time.time() - start_time

    # Close files
    f_data.close()
    
    print(f"Index built in {build_time}s")
    
    # Run searches with different nprobe values
    for nprobe in [1, 2, 5, 10]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        if task[:5] == 'task2':
            pass
            #I += (I == np.arange(I.shape[0])[:, None]) * 1000000000
            #I.sort()
            #I = I[:, :k-1]
        I = I + 1 # Convert to 1-indexed to match groundtruth (hforest also uses 0-indexed)
        # The +1 conversion should be included in contest timing measurements
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        recall = get_recall(I, gt_I, k)
        print(f"Recall: {recall * 100.0}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example execution of FAISS approximate nearest neighbor search')
    parser.add_argument('task', choices=['task1', 'task2'],
                        help='Task type to execute (task1, task2)')
    
    args = parser.parse_args()
    
    run(args.task)