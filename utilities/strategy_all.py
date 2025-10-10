# -*- encoding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import random
import math
import time
import numpy as np
from config import Config as Config
import os
import pickle
import logging
from utilities import tool_funcs
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial
import core_cpu
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from utilities.strategy_mcts_cluster import MCTSStrategy, _simi_matrix, _get_simi_fn, load_trajs_sims_from_api
from torch.nn.utils.rnn import pack_sequence, pad_sequence



# define the agent that extracts sub-trajectories from trajectories

# global variable for multiprocessing
_global_learner = None

def _topk_search_worker(args):
    """multi-process worker function, for executing topk search"""
    global _global_learner
    i, traj_id, batch_size = args
    try:
        topk_ids = _global_learner.mcts.rtree._top_k_search(traj_id, batch_size)
        return i, traj_id, topk_ids, None
    except Exception as e:
        return i, traj_id, [], str(e)

# randomly extract k sub-trajectories from each trajectory
def load_trajs_from_file(config, train_split_ratio = 0.7, eval_split_ratio = 0.8):
    
    with open(config.dataset_trajsimi_traj, "rb") as fh:
        dic_dataset = pickle.load(fh)
        trajs_merc = dic_dataset["trajs_merc"]
        trajs_ts = dic_dataset["trajs_ts"]
        

        train_trajs_merc = trajs_merc[:int(len(trajs_merc) * train_split_ratio * config.training_trajectory_ratio)]
        eval_trajs_merc = trajs_merc[int(len(trajs_merc) * train_split_ratio):int(len(trajs_merc) * eval_split_ratio)]
        test_trajs_merc = trajs_merc[int(len(trajs_merc) * eval_split_ratio):]
        
        if trajs_ts is not None:
            train_trajs_ts = trajs_ts[:int(len(trajs_ts) * train_split_ratio * config.training_trajectory_ratio)]
            eval_trajs_ts = trajs_ts[int(len(trajs_ts) * train_split_ratio):int(len(trajs_ts) * eval_split_ratio)]
            test_trajs_ts = trajs_ts[int(len(trajs_ts) * eval_split_ratio):]
        else:
            train_trajs_ts = None
            eval_trajs_ts = None
            test_trajs_ts = None
        logging.info("train_trajs_merc: {}, eval_trajs_merc: {}, test_trajs_merc: {}".format(len(train_trajs_merc), len(eval_trajs_merc), len(test_trajs_merc)))

        return train_trajs_merc, eval_trajs_merc, test_trajs_merc, train_trajs_ts, eval_trajs_ts, test_trajs_ts


def trajsimi_dataset_simis_creation_fn(trajs, fn_name):
    eval_trajs_merc, test_trajs_merc = trajs
    fn = _get_simi_fn(fn_name)
        
    eval_simis = _simi_matrix(fn, eval_trajs_merc)
    test_simis = _simi_matrix(fn, test_trajs_merc)
    
    max_distance = max( max(map(partial(max, default = float('-inf')), eval_simis)), \
                            max(map(partial(max, default = float('-inf')), test_simis)))
    return eval_simis, test_simis, max_distance
                            

def load_trajs(config, train_split_ratio = 0.7, eval_split_ratio = 0.8,\
                     eval_num = 1000, test_num = 2000):
    train_traj, eval_traj, test_traj, train_traj_ts, eval_traj_ts, test_traj_ts = load_trajs_from_file(config, train_split_ratio, eval_split_ratio)
    # build sims for eval and test
    simis_type = config.trajsimi_measure
    eval_traj = eval_traj[:eval_num]
    test_traj = test_traj[:test_num]
    if eval_traj_ts is not None:
        eval_traj_ts = eval_traj_ts[:eval_num]
        test_traj_ts = test_traj_ts[:test_num]

    if simis_type in ['stedr', 'cdds'] and train_traj_ts is not None:
        # 3D traj
        eval_traj_new = [[ [eval_traj[i][j][0], eval_traj[i][j][1], eval_traj_ts[i][j]] \
                        for j in range(len(eval_traj[i])) ] for i in range(len(eval_traj)) ]
        test_traj_new = [[ [test_traj[i][j][0], test_traj[i][j][1], test_traj_ts[i][j]] \
                        for j in range(len(test_traj[i])) ] for i in range(len(test_traj)) ]
    else:
        eval_traj_new = eval_traj
        test_traj_new = test_traj
    # build sims for eval and test
    eval_sims, test_sims, max_distance = trajsimi_dataset_simis_creation_fn((eval_traj_new, test_traj_new), simis_type)

    return {"train_trajs_merc": train_traj, "eval_trajs_merc": eval_traj, "test_trajs_merc": test_traj, \
            "train_trajs_ts": train_traj_ts, "eval_trajs_ts": eval_traj_ts, "test_trajs_ts": test_traj_ts, \
            "eval_simis": eval_sims, "test_simis": test_sims, "max_distance": max_distance}



def curriculum_post_ranking(batch_traj_ids, batch_matrix_simis):
    # each element of batch_matrix_simis is a matrix, the size of the matrix is the length of batch_traj_ids, and each element is the similarity between two trajectories
    # sort by the variance and mean of the similarity of each element, the smaller the variance and the larger the mean, the better
    # return the sorted batch_traj_ids
    
    # compute the mean and variance of the similarity matrix of each batch
    batch_scores = []
    for i, matrix in enumerate(batch_matrix_simis):
        # get the upper triangular matrix values(excluding the diagonal)
        matrix = np.array(matrix)
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        mean_sim = np.mean(upper_tri)
        var_sim = np.var(upper_tri)
        
        # compute the score: the larger the mean, the better, the smaller the variance, the better
        # use negative variance because we want to put the smaller variance in front
        score = mean_sim - var_sim
        batch_scores.append((i, score))
    
    # sort by the score, the higher the score, the better
    sorted_indices = [idx for idx, _ in sorted(batch_scores, key=lambda x: x[1], reverse=True)]

    # return the sorted batch_traj_ids
    return [batch_traj_ids[i] for i in sorted_indices], [batch_matrix_simis[i] for i in sorted_indices]

def calculate_traj_complexity(traj):
    """
    calculate the complexity of a trajectory, considering the following features:
    1. trajectory length
    2. average distance between trajectory points
    3. variance of distances between trajectory points
    4. turning angle changes of the trajectory
    5. trajectory curvature
    
    all features are normalized to the [0,1] interval
    """
    if len(traj) < 2:
        return 0.0
    
    # 1. trajectory length (normalized to [0,1])
    # assume the maximum trajectory length is 200 points
    MAX_TRAJ_LENGTH = Config.trajsimi_max_traj_len
    traj_length = len(traj)
    norm_length = min(traj_length / MAX_TRAJ_LENGTH, 1.0)
    
    # 2. calculate the distance between adjacent points
    distances = []
    angles = []
    for i in range(len(traj)-1):
        p1 = np.array(traj[i])
        p2 = np.array(traj[i+1])
        dist = np.linalg.norm(p2 - p1)
        distances.append(dist)
        
        # 3. calculate the turning angle
        if i < len(traj)-2:
            p3 = np.array(traj[i+2])
            v1 = p2 - p1
            v2 = p3 - p2
            # calculate the vector angle, handle zero vector cases
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 1e-10 and norm_v2 > 1e-10:  # avoid division by zero
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # handle numerical errors
                angle = np.arccos(cos_angle)
                angles.append(angle)
    
    # calculate the statistical features of the distance
    mean_dist = np.mean(distances) if distances else 0
    var_dist = np.var(distances) if distances else 0
    
    # normalize the distance features
    # assume the maximum average distance is 1000 meters
    MAX_MEAN_DIST = 1000
    norm_mean_dist = min(mean_dist / MAX_MEAN_DIST, 1.0)
    
    # assume the maximum variance of the distance is 10000
    MAX_VAR_DIST = 10000
    norm_var_dist = min(var_dist / MAX_VAR_DIST, 1.0)
    
    # calculate the statistical features of the angle
    mean_angle = np.mean(angles) if angles else 0
    var_angle = np.var(angles) if angles else 0
    
    # normalize the angle features
    # the angle is already in the range of [0,π], normalize it by dividing by π
    norm_mean_angle = mean_angle / np.pi
    # the maximum variance of the angle is π²/4
    norm_var_angle = var_angle / (np.pi * np.pi / 4)
    
    # 4. calculate the curvature(using the curvature of the circle formed by three adjacent points)
    curvatures = []
    for i in range(len(traj)-2):
        p1, p2, p3 = np.array(traj[i]), np.array(traj[i+1]), np.array(traj[i+2])
        # calculate the curvature of the circle formed by three adjacent points
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # check if the three points are collinear or too close
        if a < 1e-10 or b < 1e-10 or c < 1e-10:
            continue
            
        # use Heron's formula to calculate the area
        s = (a + b + c) / 2
        # check if the triangle inequality is satisfied
        if s <= a or s <= b or s <= c:
            continue
            
        # calculate the area, handle numerical errors
        area_squared = s * (s - a) * (s - b) * (s - c)
        if area_squared > 0:
            area = np.sqrt(area_squared)
            # calculate the curvature
            curvature = 4 * area / (a * b * c)
            curvatures.append(curvature)
    
    mean_curvature = np.mean(curvatures) if curvatures else 0
    
    # normalize the curvature
    # assume the maximum curvature is 1.0
    MAX_CURVATURE = 1.0
    norm_curvature = min(mean_curvature / MAX_CURVATURE, 1.0)
    
    # comprehensive score: all features are normalized to the [0,1] interval
    complexity_score = (
        0.2 * norm_length +     # trajectory length
        0.2 * norm_mean_dist +  # average distance
        0.2 * norm_var_dist +   # distance variance
        0.2 * norm_var_angle +  # angle variance
        0.2 * norm_curvature    # average curvature
    )
    
    return complexity_score

def curriculum_pre_ranking(trajs_merc, sampled_indices):
    # then we need to sort the trajectories by their own features(the distance variance of the trajectory points)
    sample_traj_features = []
    for idx in sampled_indices:
        traj = trajs_merc[idx]
        complexity = calculate_traj_complexity(traj)
        sample_traj_features.append((idx, complexity))
    
    # sort by the complexity, the lower the complexity, the better
    sorted_indices = [idx for idx, _ in sorted(sample_traj_features, key=lambda x: x[1])]
    return sorted_indices

def extract_traj_features(traj):
    """
    extract the feature vector of a trajectory, including:
    1. trajectory length
    2. trajectory complexity
    3. spatial distribution features of the trajectory
    4. shape features of the trajectory
    """
    if len(traj) < 2:
        return np.zeros(8)  # return 8-dimensional feature vector
    
    # 1. basic features
    traj_length = len(traj)
    complexity = calculate_traj_complexity(traj)
    
    # 2. spatial distribution features
    traj_array = np.array(traj)
    # calculate the bounding box of the trajectory
    min_coords = np.min(traj_array, axis=0)
    max_coords = np.max(traj_array, axis=0)
    # calculate the spatial span of the trajectory
    spatial_span = np.linalg.norm(max_coords - min_coords)
    # calculate the center of the trajectory
    center = np.mean(traj_array, axis=0)
    # calculate the average distance of the trajectory points to the center
    center_distances = np.linalg.norm(traj_array - center, axis=1)
    mean_center_dist = np.mean(center_distances)
    std_center_dist = np.std(center_distances)
    
    # 3. shape features
    # calculate the direction change between adjacent points
    directions = np.diff(traj_array, axis=0)
    # calculate the length of the direction vector
    direction_lengths = np.linalg.norm(directions, axis=1)
    
    # only consider the direction vectors with length greater than the threshold
    min_length = 1e-10  # set a small threshold
    valid_mask = (direction_lengths[:-1] > min_length) & (direction_lengths[1:] > min_length)
    
    if np.any(valid_mask):
        # only consider the angle between valid direction vectors
        valid_directions = directions[:-1][valid_mask]
        valid_next_directions = directions[1:][valid_mask]
        
        # calculate the direction change
        direction_changes = np.arccos(np.clip(
            np.sum(valid_directions * valid_next_directions, axis=1) / 
            (direction_lengths[:-1][valid_mask] * direction_lengths[1:][valid_mask]),
            -1.0, 1.0
        ))
        mean_direction_change = np.mean(direction_changes)
        std_direction_change = np.std(direction_changes)
    else:
        # if there is no valid direction change, use default values
        mean_direction_change = 0
        std_direction_change = 0
    
    raw_features = []
    if Config.ablation_type != 1:
        raw_features += [traj_length / Config.trajsimi_max_traj_len, complexity]
    if Config.ablation_type != 2:
        raw_features += [spatial_span / 1000, mean_center_dist / 500, std_center_dist / 500]
    if Config.ablation_type != 3:
        raw_features += [mean_direction_change / np.pi, std_direction_change / np.pi, np.mean(direction_lengths) / 100]

    # 4. normalize the features
    features = np.array(raw_features)
    
    return features

def extract_traj_features_parallel(trajs_merc, n_jobs=-1):
    """
    parallelly extract the feature vector of a trajectory
    
    Args:
        trajs_merc: trajectory data list
        n_jobs: number of parallel processes, -1 means using all available CPUs
    """
    from joblib import Parallel, delayed
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1  # 保留一个CPU核心
    
    logging.info(f"Extracting features using {n_jobs} processes...")
    features = Parallel(n_jobs=n_jobs)(
        delayed(extract_traj_features)(traj) for traj in trajs_merc
    )
    return np.array(features)



def build_training_samples(learner, trajs_merc, trajs_ts, simis_type, counts, batch_size, strtegy):
    # sample trajectory indices according to the strategy
    strtegy_start_time = time.time()
    sampled_indices = _sample_trajectories_by_strategy(
        learner, trajs_merc, counts, int(batch_size), strtegy
    )
    strtegy_end_time = time.time()
    logging.info(f"strategy time={strtegy_end_time - strtegy_start_time}")
    
    # prepare trajectory data
    sampled_trajs_merc_new = _prepare_trajectory_data(
        trajs_merc, trajs_ts, sampled_indices, simis_type
    )

    
    return _build_standard_training_samples(
            sampled_indices, sampled_trajs_merc_new, simis_type, counts, int(batch_size), strtegy
        )


def _sample_trajectories_by_strategy(learner, trajs_merc, counts, batch_size, strtegy):
    """sample trajectory indices according to different strategies"""
    # if last_sampled_indices is not None:
    #     return last_sampled_indices

    if strtegy == "mcts":
        sampled_indices = _initialize_and_sample_mcts(learner, trajs_merc, counts, batch_size)
    else:
        sampled_indices = list(range(counts * batch_size))
        random.shuffle(sampled_indices)
    
    return sampled_indices


def _initialize_and_sample_mcts(learner, trajs_merc, counts, batch_size):
    """initialize and sample the MCTS strategy"""
    # cache the MCTS
    # cache_file = Config.dataset_trajsimi_traj + f"_mcts_cache_{Config.trajsimi_measure}_{Config.max_leaf_size}_{Config.cluster_num}_{learner.model_name}.pkl"
    if learner.mcts is None:
            logging.info("Initializing MCTS strategy...")
        # if os.path.exists(cache_file):
        #     with open(cache_file, "rb") as fh:
        #         learner.mcts = pickle.load(fh)
        # else:
            initial_start_time = time.time()
            learner.mcts = MCTSStrategy(
                similarity_func=Config.trajsimi_measure,
                max_distance=learner.dic_datasets['max_distance'],
                lambda_penalty=Config.lambda_penalty,
                bounding_box=learner.bounding_box,
                grid_size=Config.grid_size,
                alpha=Config.alpha,
                beta=Config.beta,
                reward_fn=learner.compute_mcts_reward,
                n_clusters=Config.cluster_num,
                target_size=Config.max_leaf_size
            )
            trajectories_dict = {i: np.asarray(traj, dtype=np.float64) for i, traj in enumerate(trajs_merc)}
            learner.mcts.initialize(trajectories_dict)
            initial_end_time = time.time()
            logging.info(f"initialize mcts time: {initial_end_time - initial_start_time} seconds")
            # with open(cache_file, "wb") as fh:
            #     pickle.dump(learner.mcts, fh)
    
    assert learner.mcts is not None
    return learner.mcts.sample_trajectories_with_calibration(
        num_samples=int(counts * batch_size),  # make sure it is an integer
        num_iterations=Config.num_iterations, greedy=(Config.ablation_type == 5)
    )

def _prepare_trajectory_data(trajs_merc, trajs_ts, sampled_indices, simis_type):
    """prepare trajectory data, process the timestamp information"""
    sampled_trajs_merc = [trajs_merc[idx] for idx in sampled_indices]
    
    if trajs_ts is not None:
        sampled_trajs_ts = [trajs_ts[idx] for idx in sampled_indices]
        if simis_type in ['stedr', 'cdds']:
            return [
                [[sampled_trajs_merc[i][j][0], sampled_trajs_merc[i][j][1], sampled_trajs_ts[i][j]]
                 for j in range(len(sampled_trajs_merc[i]))]
                for i in range(len(sampled_trajs_merc))
            ]
    
    return sampled_trajs_merc


def _build_mcts_training_samples(learner, sampled_indices, sampled_trajs_merc_new, simis_type, batch_size):
    """build the training samples of the MCTS strategy"""
    # get the topk trajectory IDs
    time_start = time.time()
    total_traj_ids, topk_traj_ids = _get_mcts_topk_trajectories(learner, sampled_indices, batch_size)
    time_end = time.time()
    logging.info(f"get mcts topk trajectories time: {time_end - time_start} seconds")
    
    # get the trajectory from the original trajectory data, not from the sampled data
    # create the mapping from ID to index
    sampled_id_to_idx = {traj_id: idx for idx, traj_id in enumerate(sampled_indices)}
    
    # prepare trajectory data
    total_trajs_merc = []
    for traj_id in total_traj_ids:
        if traj_id in sampled_id_to_idx:
            # if the trajectory is sampled, get it from the sampled data
            total_trajs_merc.append(sampled_trajs_merc_new[sampled_id_to_idx[traj_id]])
        else:
            # if the trajectory is topk, get it from the original data
            # here we need to get the trajectory from the original trajectory data of the learner
            if hasattr(learner, 'dic_datasets') and 'train_traj' in learner.dic_datasets:
                original_traj = learner.dic_datasets['train_traj'][traj_id]
                # process the timestamp information
                if simis_type in ['stedr', 'cdds'] and hasattr(learner, 'dic_datasets') and 'train_traj_ts' in learner.dic_datasets:
                    traj_ts = learner.dic_datasets['train_traj_ts'][traj_id]
                    processed_traj = [
                        [original_traj[j][0], original_traj[j][1], traj_ts[j]]
                        for j in range(len(original_traj))
                    ]
                    total_trajs_merc.append(processed_traj)
                else:
                    total_trajs_merc.append(original_traj)
            else:
                # if the original data cannot be found, skip this trajectory
                logging.warning(f"Cannot find trajectory {traj_id} in original data, skipping...")
                continue
    
    # build the batch tuples
    batch_tuples = _build_mcts_batch_tuples(sampled_indices, topk_traj_ids, total_traj_ids)
    
    # calculate the similarity matrix
    batch_matrix_simis = load_trajs_sims_from_api(
        total_trajs_merc, total_traj_ids, simis_type, topk_traj_ids=batch_tuples
    )
    
    # 构建batch数据
    batch_traj_ids = [sampled_indices[i:i+batch_size] for i in range(0, len(sampled_indices), batch_size)]
    batch_topk_traj_ids = [topk_traj_ids[i:i+batch_size] for i in range(0, len(topk_traj_ids), batch_size)]
    batch_matrix_simis = [batch_matrix_simis[i:i+batch_size] for i in range(0, len(batch_matrix_simis), batch_size)]
    
    return total_traj_ids, batch_traj_ids, batch_topk_traj_ids, np.array(batch_matrix_simis)


def _get_mcts_topk_trajectories(learner, sampled_indices, batch_size):
    """get the topk trajectories of the MCTS strategy"""
    global _global_learner
    
    # set the global variable for multi-process use
    _global_learner = learner
    
    
    total_traj_ids = []
    total_traj_ids.extend(sampled_indices)
    topk_traj_ids = [None] * len(sampled_indices)  # pre-allocate the list to keep the order
    
    # prepare the task parameters
    tasks = [(i, traj_id, batch_size) for i, traj_id in enumerate(sampled_indices)]
    
    # use multi-process to execute in parallel
    n_workers = min(len(sampled_indices), mp.cpu_count())
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_topk_search_worker, tasks)
    
    # collect the results and keep the original order
    for i, traj_id, topk_ids, error in results:
        if error is not None:
            logging.error(f"Error getting topk for trajectory {traj_id}: {error}")
            topk_traj_ids[i] = []
        else:
            topk_traj_ids[i] = topk_ids
            total_traj_ids.extend(topk_ids)
    
    total_traj_ids = list(set(total_traj_ids))
    logging.info(f"total_trajs_merc: {len(total_traj_ids)}")
    
    
    # clean the global variable
    _global_learner = None
    
    return total_traj_ids, topk_traj_ids


def _build_mcts_batch_tuples(sampled_indices, topk_traj_ids, total_traj_ids):
    """build the batch tuples of the MCTS strategy"""
    batch_tuples = []
    for tid, topk_ids in zip(sampled_indices, topk_traj_ids):
        try:
            tid_idx = total_traj_ids.index(tid)
            topk_ids_idx = []
            for tk_id in topk_ids:
                try:
                    tk_idx = total_traj_ids.index(tk_id)
                    topk_ids_idx.append(tk_idx)
                except ValueError:
                    logging.warning(f"Topk trajectory ID {tk_id} not found in total_traj_ids, skipping...")
                    continue
            batch_tuples.append((tid_idx, topk_ids_idx))
        except ValueError:
            logging.warning(f"Sampled trajectory ID {tid} not found in total_traj_ids, skipping...")
            continue
    return batch_tuples


def _build_standard_training_samples(sampled_indices, sampled_trajs_merc_new, simis_type, counts, batch_size, strtegy):
    """build the training samples of the standard strategy"""
    # build the batch data
    batch_trajs_merc = [sampled_trajs_merc_new[i:i+int(batch_size)] for i in range(0, len(sampled_trajs_merc_new), int(batch_size))]
    batch_traj_ids = [sampled_indices[i:i+int(batch_size)] for i in range(0, len(sampled_indices), int(batch_size))]
    
    assert len(batch_trajs_merc) == counts
    
    # calculate the similarity matrix
    use_cache = False
    if strtegy == 'mcts':
        use_cache = True
    batch_matrix_simis = load_trajs_sims_from_api(batch_trajs_merc, batch_traj_ids, simis_type, use_cache)
    logging.info("generate matrix simis: {}".format(len(batch_matrix_simis)))
    
    # process the curriculum strategy
    # if strtegy in ["curriculum", "mcts"]:
    if strtegy in ["curriculum"]:
        logging.info("curriculum ranking for making training samples")
        batch_traj_ids, batch_matrix_simis = curriculum_post_ranking(batch_traj_ids, batch_matrix_simis)
    
    return sampled_indices, batch_traj_ids, np.array(batch_matrix_simis)

