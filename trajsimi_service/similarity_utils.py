import math
import numpy as np
import multiprocessing as mp
import traj_dist.distance as tdist
import sys
import os

# add the path of core_cpu module
sys.path.append("/home/haitao/data/GoodST/code/baseline/TrajSimiMeasures")
import core_cpu
from functools import partial

from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
import sys

import redis, struct


# configure logging, output to standard output, and set the log level to WARNING
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# get the logger object
logger = logging.getLogger("my_fastapi_app")

# define the mapping from method to single byte encoding
METHOD_MAP = {'lcss': 0, 'edr': 1, 
            'erp': 2, 'dtw': 3,
            'dfrechet': 4, 'hausdorff': 5,
            'stedr': 6, 'cdds': 7,
            'stlcss': 8
            }

# define the mapping from dataset to single byte encoding
DATASET_MAP = {'xian': 0, 'porto': 1, 'geolife': 2, 'chengdu': 3, 'germany': 4}

# global Redis connection variable (initialized in the child process)
redis_conn = None
# global trajectory data variable (set in the main process, shared by child processes)
global_lst_trajs = None
global_lst_traj_ids = None

def init_worker(fn_name):
    """
    ProcessPoolExecutor initialization function, called in each child process,
    initialize the global Redis connection, avoid duplicate connection creation.
    """
    global redis_conn
    # only use the method to determine the database, not consider the dataset
    db_key = METHOD_MAP.get(fn_name, 0)
    redis_conn = redis.Redis(host='localhost', port=6379, db=db_key)
    logger.info(f"Worker process initialized Redis connection for method={fn_name}, db={db_key}.")

def get_pair_key(traj_id1, traj_id2, dataset_id):
    """
    construct the cache key: 1 byte dataset encoding + two 4 byte trajectory IDs (sorted lexicographically)
    use 4 bytes to store the trajectory ID, because the data size does not exceed one million
    """
    # sort lexicographically to ensure the fixed order of the key
    if traj_id1 > traj_id2:
        traj_id1, traj_id2 = traj_id2, traj_id1
    # use struct.pack to pack the data into binary: !B represents 1 byte unsigned integer, II represents two 4 byte unsigned integers
    key = struct.pack('!BII', dataset_id, traj_id1, traj_id2)
    return key


def simi_comp_operator(fn_name, fn, lst_trajs, lst_traj_ids, dataset_id, sub_idx, expire_sec=3600000, use_cache=True):
    """
    for each original row index _i in sub_idx, compute the similarity with all subsequent rows, using batch pipeline method:
      1. collect all pair keys to be calculated;
      2. use pipeline to batch get cached results;
      3. batch compute similarity for the pairs that miss the cache, and use pipeline to batch setex cache;
      4. restore the result for each row according to the pair (i,j).
      
    return format: list of tuples: (row_index, row_result), where row_result is a list containing the similarity with all subsequent rows.
    """
    results = {}  # used to store the similarity for each (i,j)
    pair_list = []  # each element in the list is (i, j, key)
    l = len(lst_trajs)

    # 1. collect all pairs to be queried: traverse each _i in sub_idx, and for each _j from _i+1 to l-1
    for _i in sub_idx:
        for _j in range(_i + 1, l):
            key = get_pair_key(lst_traj_ids[_i], lst_traj_ids[_j], dataset_id)
            pair_list.append((_i, _j, key))

    global redis_conn
    if redis_conn is None:
        # defensive check, only use the method to determine the database
        db_key = METHOD_MAP.get(fn_name, 0)
        redis_conn = redis.Redis(host='localhost', port=6379, db=db_key)

    # 2. use pipeline to batch get the cached results for all pairs
    if use_cache:
        pipe = redis_conn.pipeline()
        for (_, _, key) in pair_list:
            pipe.get(key)
        cached_results = pipe.execute()
    else:
        cached_results = [None] * len(pair_list)

    # 3. traverse pair_list and corresponding cached_results
    missing = []  # used to record the pairs that miss the cache
    for idx, (i, j, key) in enumerate(pair_list):
        cached = cached_results[idx]
        if cached is not None:
            try:
                results[(i, j)] = float(cached.decode('utf-8'))
            except Exception as e:
                logger.error(f"Cache parse error for key {key}: {e}")
                missing.append((i, j, key))
        else:
            missing.append((i, j, key))

    # 4. batch compute similarity for all pairs that miss the cache
    if missing:
        # compute one by one, if the computation is large, consider parallel, but here we simplify the batch writing, and compute in the same process
        for (i, j, key) in missing:
            try:
                sim = fn(lst_trajs[i], lst_trajs[j])
                sim = round(sim, 2)
                results[(i, j)] = sim
            except Exception as e:
                logger.error(f"Error computing similarity between {i} and {j}: {e}")
                results[(i, j)] = 0.0

        # 5. use pipeline to batch setex the pairs that miss the cache to Redis
        if use_cache:
            pipe = redis_conn.pipeline()
            for (i, j, key) in missing:
                if i==j or results[(i, j)] > 0:
                    pipe.setex(key, expire_sec, str(results[(i, j)]))
            pipe.execute()

    # 6. assemble the result for each row: for each _i in sub_idx, the row_result list contains the similarity with all subsequent rows from _i+1 to l-1
    ordered_results = []
    for _i in sub_idx:
        row_result = []
        for _j in range(_i + 1, l):
            row_result.append(results.get((_i, _j), 0.0))
        ordered_results.append((_i, row_result))

    return ordered_results


def _get_simi_fn(fn_name):
    fn =  {'lcss': tdist.lcss, 'edr': tdist.edr, 
            'erp': tdist.erp, 'dtw': tdist.dtw,
            'dfrechet': tdist.discret_frechet, 'hausdorff': tdist.hausdorff,
            'stedr': core_cpu.stedr, 'cdds': core_cpu.cdds,
            'stlcss': core_cpu.stlcss
            }.get(fn_name, None)
    
    if fn is None:
        return None
        
    if fn_name == 'erp': 
        fn = partial(fn, g = np.asarray([0, 0], dtype = np.float64))
    elif fn_name in ['stedr', 'stlcss']:
        fn = partial(fn, eps = 200, delta = 60)
    elif fn_name == 'cdds':
        fn = partial(fn, eps = 200)
    return fn

def compute_simi_matrix(fn_name, trajectories, traj_ids, dataset, use_cache, batch_size=50, processes=None):
    """
    compute the similarity matrix and return (2D list or np.array)
    trajectories: list[list[float]] or list[list[list]] depending on the actual situation
    traj_ids: list[int] list of trajectory IDs
    dataset: str dataset name
    """
    fn = _get_simi_fn(fn_name)
    if fn is None:
        logger.error(f"Unsupported method: {fn_name}")
        raise ValueError(f"Unsupported method: {fn_name}")
    
    if dataset not in DATASET_MAP:
        logger.error(f"Unsupported dataset: {dataset}")
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    dataset_id = DATASET_MAP[dataset]
    
    # convert all trajectories to numpy array
    np_trajs = [np.asarray(traj, dtype=np.float64) for traj in trajectories]

    l = len(np_trajs)
    if l == 0:
        return []

    if not processes:
        # default use CPU core number - 3
        processes = 1 #max(mp.cpu_count() - 3, 1)

    # split all indices into batches according to batch_size
    tasks = []
    total_batch = math.ceil(l / batch_size)
    for i in range(total_batch):
        start_i = batch_size * i
        end_i = min(batch_size * (i + 1), l)
        tasks.append((fn_name, fn, np_trajs, traj_ids, dataset_id, range(start_i, end_i), 3600000, use_cache))

    # use ProcessPoolExecutor to run tasks
    results_with_idx = []
    with ProcessPoolExecutor(max_workers=processes, initializer=init_worker, initargs=(fn_name,), mp_context=mp.get_context("spawn")) as executor:
        future_to_task = {executor.submit(simi_comp_operator, *task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                results_with_idx.extend(future.result())
            except Exception as exc:
                logger.error(f"Task generated an exception: {exc}")

    # sort the results according to the row index
    results_with_idx.sort(key=lambda x: x[0])
    # extract the ordered row_result list
    ordered_results = [row_result for _, row_result in results_with_idx]

    # construct the symmetric matrix
    simi_mat = [[0.0] * l for _ in range(l)]
    for i, row in enumerate(ordered_results):
        expected = l - i - 1
        if len(row) != expected:
            logger.debug(f"Warning: row {i} length = {len(row)}, but expected {expected}")
        for k, val in enumerate(row):
            j = i + 1 + k
            simi_mat[i][j] = val
            simi_mat[j][i] = val
    return simi_mat


def init_worker_with_trajs(fn_name, lst_trajs, lst_traj_ids):
    global redis_conn, global_lst_trajs, global_lst_traj_ids
    db_key = METHOD_MAP.get(fn_name, 0)
    redis_conn = redis.Redis(host='localhost', port=6379, db=db_key)
    global_lst_trajs = lst_trajs
    global_lst_traj_ids = lst_traj_ids
    logger.info(f"Worker process initialized with trajs, traj_ids, method={fn_name}, db={db_key}.")

def compute_batch_simi_matrices(fn_name, batch_trajectories, batch_traj_ids, dataset, use_cache=True, batch_size=50, processes=None):
    """
    batch compute similarity matrices for multiple trajectory groups
    batch_trajectories: list[list[list[list[float]]]] batch trajectory data
    batch_traj_ids: list[list[int]] batch trajectory ID data
    dataset: str dataset name
    """
    if len(batch_trajectories) != len(batch_traj_ids):
        raise ValueError("batch_trajectories and batch_traj_ids must have the same length")
    
    # if compute st similarity, check if the trajectory has the third dimension, if not, add the third dimension
    if fn_name in ['stedr', 'stlcss', 'cdds']:
        if len(batch_trajectories[0][0][0]) < 3:
            new_batch_trajectories = []
            for trajectories, traj_ids in zip(batch_trajectories, batch_traj_ids):
                new_trajectories = []
                for traj in trajectories:
                    new_trajectories.append([[point[0], point[1], float(idx)] for idx, point in enumerate(traj)])
                new_batch_trajectories.append(new_trajectories)
            batch_trajectories = new_batch_trajectories
        
    # parallel process each batch
    # assert processes == 4
    with ProcessPoolExecutor(max_workers=processes) as executor:
        # submit all tasks and save future and corresponding index
        future_to_index = {}
        for i, (trajectories, traj_ids) in enumerate(zip(batch_trajectories, batch_traj_ids)):
            future = executor.submit(compute_simi_matrix, fn_name, trajectories, traj_ids, dataset, use_cache, batch_size, processes)
            future_to_index[future] = i
        
        # initialize the result list
        batch_results = [None] * len(batch_trajectories)
        
        # collect the results in the order of completion, but keep the original order
        for future in as_completed(future_to_index):
            batch_index = future_to_index[future]
            try:
                result = future.result()
                # verify if the dimension of the returned matrix is consistent with the input batch length
                expected_dim = len(batch_trajectories[batch_index])
                if result is not None:
                    if len(result) != expected_dim or (len(result) > 0 and len(result[0]) != expected_dim):
                        raise ValueError(f"Batch {batch_index}: Matrix dimension mismatch. Expected {expected_dim}x{expected_dim}, got {len(result)}x{len(result[0]) if result else 0}")
                batch_results[batch_index] = result
            except Exception as exc:
                logger.error(f"Batch {batch_index} computation generated an exception: {exc}")
                batch_results[batch_index] = None
    
    return batch_results