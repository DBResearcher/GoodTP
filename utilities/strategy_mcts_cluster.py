# -*- encoding: utf-8 -*-
import sys
sys.path.append('..')
sys.path.append('../..')
import random
import math
import time
import logging
import numpy as np
from typing import List, Dict, Optional
from functools import partial
from sklearn.cluster import KMeans
import traj_dist.distance as tdist
import core_cpu
from config import Config as Config
import multiprocessing as mp
import os
import itertools
import requests
import pickle
import hashlib
from functools import wraps

_GLOBAL_FEATURES = None
_GLOBAL_TRAJECTORIES = None

def retry_on_failure(max_retries=3, retry_delay=1.0, backoff_factor=1.5, exceptions=(Exception,), 
                    retry_condition=None, on_retry=None):
    """
    retry decorator, support exponential backoff strategy and custom retry conditions
    
    Args:
        max_retries: maximum number of retries
        retry_delay: initial retry interval (seconds)
        backoff_factor: backoff factor
        exceptions: exception types that need to be retried
        retry_condition: custom retry condition function, receives exception and attempt number, returns whether to retry
        on_retry: callback function before retry, receives exception and attempt number
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:  # only record success information after retry
                        logging.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # check the custom retry condition
                    should_retry = True
                    if retry_condition is not None:
                        should_retry = retry_condition(e, attempt)
                    
                    if not should_retry:
                        logging.error(f"Function {func.__name__} failed and retry condition not met: {e}")
                        raise e
                    
                    logging.warning(f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries}: {e}")
                    
                    # execute the callback before retry
                    if on_retry is not None:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            logging.warning(f"Retry callback failed: {callback_error}")
                    
                    # if not the last attempt, wait and retry
                    if attempt < max_retries - 1:
                        logging.info(f"Retrying {func.__name__} in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # all retries failed
            error_msg = f"Function {func.__name__} failed after {max_retries} attempts"
            if last_exception:
                error_msg += f". Last error: {last_exception}"
            
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
        return wrapper
    return decorator

def _should_retry_api_call(exception, attempt):
    """custom retry condition: do not retry for some errors"""
    # do not retry for authentication errors, permission errors, etc.
    if isinstance(exception, requests.exceptions.HTTPError):
        if exception.response.status_code in [401, 403, 404]:
            return False
    return True

def _on_retry_callback(exception, attempt):
    """callback function before retry"""
    if isinstance(exception, requests.exceptions.Timeout):
        logging.info(f"Timeout occurred, increasing timeout for next attempt")
    elif isinstance(exception, requests.exceptions.ConnectionError):
        logging.info(f"Connection error, checking network connectivity")

@retry_on_failure(
    max_retries=3, 
    retry_delay=1.0, 
    backoff_factor=1.5,
    exceptions=(requests.exceptions.RequestException, requests.exceptions.Timeout, 
               ValueError, RuntimeError),
    retry_condition=_should_retry_api_call,
    on_retry=_on_retry_callback
)
def load_trajs_sims_from_api(batch_traj_merc_list, batch_traj_ids, simis_type='dtw', use_cache=True):
    """
    load trajectory similarity data from API, support smart retry mechanism
    
    Args:
        batch_traj_merc_list: trajectory data list
        batch_traj_ids: trajectory ID list
        simis_type: similarity calculation type
    
    Returns:
        simi_matrix: similarity matrix
    
    Raises:
        RuntimeError: when all retries fail
    """
    payload = {
        "method": simis_type,
        "batch_trajectories": batch_traj_merc_list,
        "batch_traj_ids": batch_traj_ids,
        "dataset": Config.dataset,
        "use_cache": use_cache
    }
    service_urls = [
        "http://localhost:8800/compute_similarity"
    ]
    last_exception = None
    for service_url in service_urls:
        try:
            resp = requests.post(service_url, json=payload, timeout=5000)
            resp.raise_for_status()
            result_data = resp.json()
            simi_matrix = result_data.get("batch_matrix", [])
            if not simi_matrix:
                raise RuntimeError(f"API {service_url} returned empty batch_matrix")
            break
        except Exception as e:
            logging.warning(f"Service URL {service_url} failed: {e}")
            last_exception = e
            simi_matrix = None
    else:
        raise RuntimeError(f"All service URLs failed: {service_urls}") from last_exception
    
    return simi_matrix

def _get_simi_fn(fn_name):
    fn = {'lcss': tdist.lcss, 'edr': tdist.edr, 
          'erp': tdist.erp, 'dtw': tdist.dtw,
          'dfrechet': tdist.discret_frechet, 'hausdorff': tdist.hausdorff,
          'stedr': core_cpu.stedr, 'cdds': core_cpu.cdds,
          'stlcss': core_cpu.stlcss
          }.get(fn_name, None)
    if fn_name == 'erp': 
        fn = partial(fn, g = np.asarray([0, 0], dtype = np.float64))
        # fn = partial(fn, g = np.asarray([12125125, 4056355], dtype = np.float64)) # xian_7_20inf
    elif fn_name in ['stedr', 'stlcss']:
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps, delta = Config.trajsimi_edr_lcss_delta)
    elif fn_name == 'cdds':
        fn = partial(fn, eps = Config.trajsimi_edr_lcss_eps)
    return fn

def _simi_comp_operator(fn, sub_idx):
    global _GLOBAL_TRAJECTORIES
    simi = []
    l = len(_GLOBAL_TRAJECTORIES)
    for _i in sub_idx:
        t_i = np.asarray(_GLOBAL_TRAJECTORIES[_i], dtype=np.float64)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.asarray(_GLOBAL_TRAJECTORIES[_j], dtype=np.float64)
            simi_row.append(fn(t_i, t_j))
        simi.append(simi_row)
    return simi

def _simi_matrix_single(fn, lst_trajs, show_progress=True):
    _time = time.time()
    l = len(lst_trajs)
    lst_simi = []
    for i in range(l):
        t_i = np.asarray(lst_trajs[i], dtype=np.float64)
        row = [0] * (i+1)
        for j in range(i+1, l):
            t_j = np.asarray(lst_trajs[j], dtype=np.float64)
            row.append(fn(t_i, t_j))
        lst_simi.append(row)
    arr_simi = np.asarray(lst_simi, dtype=np.float32)
    arr_simi = arr_simi + arr_simi.T
    assert arr_simi.shape[0] == arr_simi.shape[1] and arr_simi.shape[0] == l
    if show_progress:
        logging.info('simi_matrix_single computation done., @={}, #={}'.format(time.time() - _time, len(arr_simi)))
    return arr_simi

def _simi_matrix(fn, lst_trajs, show_progress=True):
    
    l = len(lst_trajs)
    if l <= 500:
        return _simi_matrix_single(fn, lst_trajs, show_progress)
    _time = time.time()
    global _GLOBAL_TRAJECTORIES
    _GLOBAL_TRAJECTORIES = lst_trajs
    
    batch_size = max(50, l // (mp.cpu_count() * 2))
    num_cores = max(1, mp.cpu_count() - 2)
    tasks = []
    for i in range(0, l, batch_size):
        tasks.append((fn, list(range(i, min(i + batch_size, l)))))
    with mp.Pool(num_cores) as pool:
        lst_simi = pool.starmap(_simi_comp_operator, tasks)
    lst_simi = list(itertools.chain.from_iterable(lst_simi))
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0]*(i+1) + row_simi
    arr_simi = np.asarray(lst_simi, dtype=np.float32)
    arr_simi = arr_simi + arr_simi.T
    assert arr_simi.shape[0] == arr_simi.shape[1] and arr_simi.shape[0] == l
    if show_progress:
        logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(arr_simi)))
    return arr_simi

# ================== ClusterTree structure ==================
class ClusterNode:
    def __init__(self, indices, level=0, parent=None, max_leaf_size=256, n_clusters=10, reward_topk=50):
        self.indices = indices
        self.level = level
        self.parent = parent
        self.children = []
        self.n_clusters = n_clusters
        self.max_leaf_size = max_leaf_size
        self.reward_topk = reward_topk
        self.center = None
        self.cached_sampled_ids = None
        self.cached_similarities = None
        if len(indices) <= max_leaf_size:
            self.is_leaf = True
            return
        self.is_leaf = False
        self._split()

    @property
    def trajectories(self):
        global _GLOBAL_TRAJECTORIES
        if _GLOBAL_TRAJECTORIES is None:
            logging.error("_GLOBAL_TRAJECTORIES is None, cannot access trajectories")
            return []
        try:
            return [_GLOBAL_TRAJECTORIES[idx] for idx in self.indices]
        except KeyError as e:
            logging.error(f"Missing trajectory index: {e}")
            return []
        except Exception as e:
            logging.error(f"Error accessing trajectories: {e}")
            return []

    @property
    def features(self):
        global _GLOBAL_FEATURES
        if _GLOBAL_FEATURES is None:
            logging.error("_GLOBAL_FEATURES is None, cannot access features")
            return []
        try:
            return [_GLOBAL_FEATURES[idx] for idx in self.indices]
        except KeyError as e:
            logging.error(f"Missing feature index: {e}")
            return []
        except Exception as e:
            logging.error(f"Error accessing features: {e}")
            return []

    @property
    def n_traj(self):
        return len(self.indices)

    def _split(self):
        features = self.features
        if features is None or len(features) == 0:
            # logging.warning(f"Node {self.indices[:5]} has no features, marking as leaf")
            self.is_leaf = True
            return
            
        n_clusters = min(self.n_clusters, len(self.indices))
        if n_clusters <= 1:
            self.is_leaf = True
            return
            
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            clusters = []
            for i in range(n_clusters):
                idxs = np.where(labels == i)[0]
                clusters.append([self.indices[j] for j in idxs])
            centers = kmeans.cluster_centers_
            min_cluster_size = max(self.reward_topk + 1, self.max_leaf_size // 2, 2)
            merged = [False] * n_clusters
            while True:
                merged_this_round = False
                small_clusters = [i for i, c in enumerate(clusters) if 0 < len(c) < min_cluster_size and not merged[i]]
                if not small_clusters:
                    break
                for i in small_clusters:
                    min_dist = float('inf')
                    nearest = -1
                    for j, c2 in enumerate(clusters):
                        if i == j or merged[j] or len(c2) == 0:
                            continue
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < min_dist:
                            min_dist = dist
                            nearest = j
                    if nearest != -1:
                        clusters[nearest].extend(clusters[i])
                        clusters[i] = []
                        merged[i] = True
                        merged_this_round = True
                if not merged_this_round:
                    break
                for i, c in enumerate(clusters):
                    if not merged[i] and len(c) > 0:
                        idxs = [self.indices.index(idx) for idx in c]
                        centers[i] = np.mean(np.array(features)[idxs], axis=0)
            # check if there is a case that cannot be split
            for cluster_indices in clusters:
                if not cluster_indices:
                    continue
                if len(cluster_indices) == len(self.indices):
                    # cannot be split, directly mark self as leaf node and terminate
                    self.is_leaf = True
                    self.children = []
                    return
            for cluster_indices in clusters:
                if not cluster_indices:
                    continue
                if len(cluster_indices) <= self.max_leaf_size:
                    child = ClusterNode(cluster_indices, level=self.level + 1, parent=self,
                                       max_leaf_size=self.max_leaf_size, n_clusters=self.n_clusters, reward_topk=self.reward_topk)
                    child.is_leaf = True
                else:
                    child = ClusterNode(cluster_indices, level=self.level + 1, parent=self,
                                       max_leaf_size=self.max_leaf_size, n_clusters=self.n_clusters, reward_topk=self.reward_topk)
                self.children.append(child)
        except Exception as e:
            logging.error(f"Error in _split for node {self.indices[:5]}: {e}")
            self.is_leaf = True
            self.children = []

    def get_all_indices(self):
        if self.is_leaf:
            return self.indices
        else:
            ids = []
            for child in self.children:
                ids.extend(child.get_all_indices())
            return ids

class ClusterTree:
    def __init__(self, trajectories_dict, max_leaf_size=256, n_clusters=10, reward_topk=None):
        from utilities.strategy_all import extract_traj_features_parallel
        import os, pickle, logging
        from config import Config
        global _GLOBAL_FEATURES, _GLOBAL_TRAJECTORIES
        if reward_topk is None:
            reward_topk = getattr(Config, 'REWARD_TOPK', 50)
        self.reward_topk = reward_topk
        all_indices = list(trajectories_dict.keys())
        all_trajs = [trajectories_dict[idx] for idx in all_indices]
        _GLOBAL_TRAJECTORIES = {idx: traj for idx, traj in zip(all_indices, all_trajs)}
        
        if not all_trajs:
            raise ValueError("No trajectories provided to ClusterTree")
            
        # ablation experiment, if ablation_type is 1-3, it means that we need to rebuild the features
        if Config.ablation_type in [1, 2, 3]:
            feature_cache_file = Config.dataset_trajsimi_traj + f"_cluster_features_cache_{len(all_trajs)}_leaf{max_leaf_size}_cluster{n_clusters}_{Config.ablation_type}.pkl"
            logging.info(f"[ClusterTree] Using ablation type {Config.ablation_type} for feature extraction")
        else:
            feature_cache_file = Config.dataset_trajsimi_traj + f"_cluster_features_cache_{len(all_trajs)}_leaf{max_leaf_size}_cluster{n_clusters}.pkl"
        
        
        all_features = None
        if os.path.exists(feature_cache_file):
            try:
                with open(feature_cache_file, "rb") as fh:
                    all_features = pickle.load(fh)
                logging.info(f"[ClusterTree] Loaded cached features from {feature_cache_file}")
            except Exception as e:
                logging.warning(f"[ClusterTree] Failed to load cached features: {e}")
        
        if all_features is None:
            logging.info(f"[ClusterTree] Extracting features for {len(all_trajs)} trajectories...")
            try:
                all_features = extract_traj_features_parallel(all_trajs, n_jobs=-1)
                if all_features is None or len(all_features) == 0:
                    raise ValueError("Feature extraction returned empty result")
                try:
                    with open(feature_cache_file, "wb") as fh:
                        pickle.dump(all_features, fh)
                    logging.info(f"[ClusterTree] Saved features to {feature_cache_file}")
                except Exception as e:
                    logging.warning(f"[ClusterTree] Failed to save feature cache: {e}")
            except Exception as e:
                logging.error(f"[ClusterTree] Feature extraction failed: {e}")
                raise
        
        # set the global features and trajectories
        # _GLOBAL_TRAJECTORIES = {idx: traj for idx, traj in zip(all_indices, all_trajs)}
        _GLOBAL_FEATURES = {idx: feat for idx, feat in zip(all_indices, all_features)}

        tree_cache_file = Config.dataset_trajsimi_traj + f"_clustertree_cache_{len(all_trajs)}_leaf{max_leaf_size}_cluster{n_clusters}_{Config.ablation_type}.pkl"
        
        if os.path.exists(tree_cache_file):
            try:
                with open(tree_cache_file, "rb") as fh:
                    cached_tree = pickle.load(fh)
                if (hasattr(cached_tree, 'root') and 
                    len(cached_tree.root.indices) == len(all_indices) and
                    set(cached_tree.root.indices) == set(all_indices)):
                    logging.info(f"[ClusterTree] Loaded cached tree from {tree_cache_file}")
                    self.root = cached_tree.root
                    self.reward_topk = getattr(cached_tree, 'reward_topk', reward_topk)
                    return
                else:
                    logging.warning(f"[ClusterTree] Cached tree validation failed, trying feature cache...")
            except Exception as e:
                logging.warning(f"[ClusterTree] Failed to load cached tree: {e}, trying feature cache...")
        
        logging.info(f"[ClusterTree] Building new tree for {len(all_trajs)} trajectories...")
        try:
            self.root = ClusterNode(all_indices, level=0, parent=None, max_leaf_size=max_leaf_size, n_clusters=n_clusters, reward_topk=self.reward_topk)
            try:
                with open(tree_cache_file, "wb") as fh:
                    pickle.dump(self, fh)
                logging.info(f"[ClusterTree] Saved tree to cache: {tree_cache_file}")
            except Exception as e:
                logging.warning(f"[ClusterTree] Failed to save tree cache: {e}")
        except Exception as e:
            logging.error(f"[ClusterTree] Failed to build tree: {e}")
            raise

    # compatible cache recovery features
    def get_global_features_dict(self):
        global _GLOBAL_FEATURES
        return _GLOBAL_FEATURES
    def get_trajectories(self, indices):
        global _GLOBAL_TRAJECTORIES
        return [_GLOBAL_TRAJECTORIES[idx] for idx in indices]
    def get_features(self, indices):
        global _GLOBAL_FEATURES
        return [_GLOBAL_FEATURES[idx] for idx in indices]

    def get_all_nodes_with_trajectories(self):
        nodes = []
        def dfs(node):
            if node.is_leaf:
                nodes.append(node)
            else:
                for child in node.children:
                    dfs(child)
        dfs(self.root)
        return nodes

    def summarize_tree(self):
        total_nodes = 0
        leaf_nodes = 0
        max_depth = 0
        def dfs(node, depth=0):
            nonlocal total_nodes, leaf_nodes, max_depth
            total_nodes += 1
            if node.is_leaf:
                leaf_nodes += 1
            if depth > max_depth:
                max_depth = depth
            for child in node.children:
                dfs(child, depth+1)
        dfs(self.root)
        logging.info(f"[ClusterTree] total nodes: {total_nodes}, leaf nodes: {leaf_nodes}, max depth: {max_depth}")

    def init_sampling_for_all_nodes(self, similarity_func_name):
        # sample and cache the similarity for all nodes
        REWARD_TOPK = 50
        
        # get all nodes (not only leaf nodes)
        all_nodes = self._get_all_nodes()
        logging.info(f"Initializing sampling for {len(all_nodes)} nodes (all levels)")
        
        batch_traj_merc_list = []
        batch_traj_ids = []
        node_to_batch_idx = {}  # record the batch index of each node
        
        # initialize sampling for each node
        for node_idx, node in enumerate(all_nodes):
            n_traj = len(node.indices)
            
            if n_traj < 2:
                # for nodes with only one trajectory, use the trajectory itself
                node.cached_sampled_ids = node.indices
                node.cached_similarities = np.zeros((n_traj, n_traj), dtype=np.float32)
                continue
                
            # calculate the sample size
            sample_size = min(n_traj // 10, node.max_leaf_size)
            sample_size = max(sample_size, max(node.max_leaf_size // 10, REWARD_TOPK + 1))
            sample_size = min(sample_size, n_traj)
            
            # make sure there are at least REWARD_TOPK + 1 trajectories for sampling
            if n_traj < REWARD_TOPK + 1:
                sample_size = n_traj
                
            # make sure the sample size is at least 2
            if sample_size < 2:
                sample_size = min(2, n_traj)
            
            # execute sampling
            if n_traj > sample_size:
                sampled_idx = random.sample(range(n_traj), sample_size)
                # extract sample_size trajectories, their features are most representative, for example, the nearest to the mean vector
                # 1. calculate the mean vector
                # mean_vec = np.mean(node.features, axis=0)
                # # 2. calculate the distance between each trajectory and the mean vector
                # distances = np.linalg.norm(node.features - mean_vec, axis=1)
                # # 3. sort by distance, extract sample_size trajectories
                # sampled_idx = np.argsort(distances)[:sample_size]
            else:
                sampled_idx = list(range(n_traj))
                
            node.cached_sampled_ids = [node.indices[i] for i in sampled_idx]
            
            # verify the sampling result
            if not node.cached_sampled_ids:
                # force using all available trajectories
                node.cached_sampled_ids = node.indices[:min(len(node.indices), 10)]
            
            # get the sampled trajectory data
            try:
                sampled_trajs = [node.trajectories[i] for i in sampled_idx]
                batch_traj_merc_list.append([traj.tolist() for traj in sampled_trajs])
                batch_traj_ids.append(node.cached_sampled_ids)
                node_to_batch_idx[node_idx] = len(batch_traj_merc_list) - 1
            except Exception as e:
                # use the original indices as fallback
                node.cached_sampled_ids = node.indices[:min(len(node.indices), 10)]
                sampled_trajs = [node.trajectories[i] for i in range(min(len(node.indices), 10))]
                batch_traj_merc_list.append([traj.tolist() for traj in sampled_trajs])
                batch_traj_ids.append(node.cached_sampled_ids)
                node_to_batch_idx[node_idx] = len(batch_traj_merc_list) - 1
        
        # batch API calls
        try:
            import math

            def split_list(lst, n):
                k, m = divmod(len(lst), n)
                return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

            num_splits = 10
            actual_splits = min(num_splits, len(batch_traj_merc_list))
            sub_merc_lists = split_list(batch_traj_merc_list, actual_splits)
            sub_id_lists = split_list(batch_traj_ids, actual_splits)

            batch_similarities = []
            for sub_merc, sub_ids in zip(sub_merc_lists, sub_id_lists):
                result = load_trajs_sims_from_api(sub_merc, sub_ids, simis_type=similarity_func_name)
                batch_similarities.extend(result)
            # sys.exit()
            assert len(batch_similarities) == len(batch_traj_merc_list)
            # assign the similarity matrix to each node
            for node_idx, node in enumerate(all_nodes):
                batch_idx = node_to_batch_idx[node_idx]
                similarities = batch_similarities[batch_idx]
                assert len(similarities) == len(node.cached_sampled_ids)
                node.cached_similarities = np.array(similarities, dtype=np.float32)
                    
        except Exception as e:
            logging.error(f"Failed to calculate the similarity cache for all nodes: {e}.")
            raise e
          
        # final statistics
        total_sampled = sum(len(node.cached_sampled_ids) for node in all_nodes)
        logging.info(f"Total sampled trajectories across all nodes: {total_sampled}")

    def _get_all_nodes(self):
        """get all nodes (not only leaf nodes)"""
        all_nodes = []
        
        def dfs(node):
            all_nodes.append(node)
            for child in node.children:
                dfs(child)
        
        dfs(self.root)
        return all_nodes

    def print_all_nodes_info(self):
        """print the detailed information of all nodes in the cluster tree"""
        logging.info("=" * 60)
        logging.info("CLUSTER TREE NODES SUMMARY")
        logging.info("=" * 60)
        
        # statistics information
        total_nodes = 0
        leaf_nodes = 0
        internal_nodes = 0
        nodes_with_cache = 0
        nodes_without_cache = 0
        
        def count_nodes(node):
            nonlocal total_nodes, leaf_nodes, internal_nodes, nodes_with_cache, nodes_without_cache
            total_nodes += 1
            
            if node.is_leaf:
                leaf_nodes += 1
            else:
                internal_nodes += 1
                
            if hasattr(node, 'cached_sampled_ids') and node.cached_sampled_ids is not None and len(node.cached_sampled_ids) > 0:
                nodes_with_cache += 1
            else:
                nodes_without_cache += 1
                
            for child in node.children:
                count_nodes(child)
        
        count_nodes(self.root)
        
        logging.info(f"Total nodes: {total_nodes}")
        logging.info(f"Leaf nodes: {leaf_nodes}")
        logging.info(f"Internal nodes: {internal_nodes}")
        logging.info(f"Nodes with valid cache: {nodes_with_cache}")
        logging.info(f"Nodes without valid cache: {nodes_without_cache}")
        
        # show the number of nodes in each level
        level_counts = {}
        def count_by_level(node, level=0):
            level_counts[level] = level_counts.get(level, 0) + 1
            for child in node.children:
                count_by_level(child, level + 1)
        
        count_by_level(self.root)
        logging.info("Nodes by level:")
        for level in sorted(level_counts.keys()):
            logging.info(f"  Level {level}: {level_counts[level]} nodes")
        
        logging.info("=" * 60)

# ================== MCTSStrategy based on ClusterTree ==================
class MCTSNode:
    def __init__(self, clusternode, parent=None, terminal_node=False):
        self.clusternode = clusternode
        self.parent = parent
        self.children = {}  # {'select': MCTSNode, 'expand': {idx: MCTSNode, ...}}
        self.expanded_actions = set()
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = terminal_node
        self.reward = None

class MCTSStrategy:
    def __init__(self, similarity_func, max_distance, target_size=256, lambda_penalty=0.01, C=1.414, \
                 bounding_box=None, grid_size=20, alpha=0.5, beta=(0.3, 0.25, 0.25, 0.2), reward_fn=None, n_clusters=10):
        if reward_fn is None:
            raise ValueError("reward_fn cannot be None for MCTSStrategy")
        
        self.target_size = target_size
        self.lambda_penalty = lambda_penalty
        self.C = C
        self.similarity_func_name = similarity_func
        self.max_distance = max_distance
        self.trajectories_data = {}
        self.clustertree = None
        self.bounding_box = bounding_box
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.reward_fn = reward_fn
        self.mcts_root = None
        self.n_clusters = n_clusters
        
        logging.info(f"MCTSStrategy initialized with target_size={target_size}, similarity_func={similarity_func}")

    def initialize(self, trajectories: Dict[int, np.ndarray]):
        if not trajectories:
            raise ValueError("trajectories dictionary is empty")
        
        self.trajectories_data = trajectories
        logging.info(f"Initializing MCTSStrategy with {len(trajectories)} trajectories")
        
        try:
            self.clustertree = ClusterTree(trajectories, max_leaf_size=self.target_size, n_clusters=self.n_clusters)
            self.clustertree.init_sampling_for_all_nodes(self.similarity_func_name)
            self.clustertree.print_all_nodes_info()
            self.clustertree.summarize_tree()
            self.mcts_root = self._make_node(self.clustertree.root)
            logging.info("MCTSStrategy initialization completed successfully")
        except Exception as e:
            logging.error(f"Failed to initialize MCTSStrategy: {e}")
            raise

    def reset(self):
        self.clustertree = None
        self.mcts_root = None

    def _make_node(self, clusternode, parent=None, terminal_node=False):
        return MCTSNode(clusternode, parent, terminal_node)

    def _is_leaf(self, node):
        return node.is_terminal

    def _available_actions(self, node):
        if self._is_leaf(node):
            return None
        elif node.clusternode.is_leaf or not node.clusternode.children:
            return ['select']
        else:
            return ['select', 'expand']

    def _collect_trajectories(self, selected_nodes):
        ids = set()
        for clusternode in selected_nodes:
            ids.update(clusternode.get_all_indices())
        return list(ids)


    def _select(self, root_node):
        path = []
        node = root_node
        while True:
            path.append(node)
            actions = self._available_actions(node)
            if actions is None:
                return path, node, None
            if not hasattr(node, 'expanded_actions'):
                node.expanded_actions = set()
            unexpanded = [a for a in actions if a not in node.expanded_actions]
            if unexpanded:
                return path, node, unexpanded[0]
            best_score = -float('inf')
            best_action = None
            for act in actions:
                child = node.children.get(act)
                if child is None or (isinstance(child, dict) and all(sub.visits == 0 for sub in child.values())):
                    score = float('inf')
                elif isinstance(child, dict):
                    scores = []
                    for sub in child.values():
                        if sub.visits == 0:
                            scores.append(float('inf'))
                        else:
                            mean = sub.total_reward / sub.visits
                            ucb = mean + self.C * math.sqrt(math.log(node.visits+1) / sub.visits)
                            scores.append(ucb)
                    score = max(scores) if scores else float('-inf')
                else:
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        mean = child.total_reward / child.visits
                        score = mean + self.C * math.sqrt(math.log(node.visits+1) / child.visits)
                if score > best_score:
                    best_score = score
                    best_action = act
            child = node.children.get(best_action)
            if best_action == 'expand' and isinstance(child, dict):
                best_sub_score = -float('inf')
                best_sub = None
                for sub in child.values():
                    if sub.visits == 0:
                        sub_score = float('inf')
                    else:
                        mean = sub.total_reward / sub.visits
                        sub_score = mean + self.C * math.sqrt(math.log(node.visits+1) / sub.visits)
                    if sub_score > best_sub_score:
                        best_sub_score = sub_score
                        best_sub = sub
                node = best_sub
            else:
                node = child

    def _expand(self, node, action):
        if 'children' not in node.__dict__:
            node.children = {}
            node.expanded_actions = set()
        if action == 'select':
            if node.children.get('select') is None:
                child = self._make_node(node.clusternode, parent=node, terminal_node=True)
                node.children['select'] = child
            else:
                child = node.children['select']
            node.expanded_actions.add(action)
            return child
        elif action == 'expand':
            child_dict = {}
            for idx, child_clusternode in enumerate(node.clusternode.children):
                child_dict[idx] = self._make_node(child_clusternode, parent=node)
            node.children['expand'] = child_dict
            node.expanded_actions.add(action)
            return child_dict
        else:
            raise ValueError(f"未知动作: {action}")

    def _simulate(self, node, n_rollouts=1):
        actions = self._available_actions(node)
        if not actions:
            return 0
        def rollout():
            act = random.choice(actions)
            if act == 'select':
                sampled_ids = node.clusternode.cached_sampled_ids
                similarities = node.clusternode.cached_similarities
                if sampled_ids is None or similarities is None:
                    logging.warning(f"Node {node.clusternode.indices[:5]} has None cached data, returning 0 reward")
                    return 0
                reward = self.reward_fn(sampled_ids, similarities)
                return reward
            elif act == 'expand':
                candidate_nodes = [self._make_node(child_clusternode, None) for child_clusternode in node.clusternode.children]
                if not candidate_nodes:
                    return 0
                rewards = []
                for child in candidate_nodes:
                    r = self._simulate(child, n_rollouts=1)
                    rewards.append(r)
                if rewards:
                    return np.mean(rewards)
                else:
                    return 0
            else:
                return 0
        rewards = [rollout() for _ in range(n_rollouts)]
        return np.mean(rewards)

    def _select_expand_simulate_path(self, root_node, n_rollouts=3):
        def compute_reward(node):
            sampled_ids = node.clusternode.cached_sampled_ids
            similarities = node.clusternode.cached_similarities
            if sampled_ids is None or similarities is None:
                logging.warning(f"Node {node.clusternode.indices[:5]} has None cached data, returning 0 reward")
                return 0
            reward = self.reward_fn(sampled_ids, similarities)
            return reward
        path, node, action = self._select(root_node)
        if action is not None:
            child = self._expand(node, action)
            if action == 'select' and child is not None:
                path.append(child)
                reward = compute_reward(child)
                return path, reward
            elif action == 'expand' and isinstance(child, dict):
                child_keys = list(child.keys())
                if not child_keys:
                    reward = self._simulate(node, n_rollouts=n_rollouts)
                    return path, reward
                if '_expand_counter' not in node.__dict__:
                    node._expand_counter = 0
                idx = child_keys[node._expand_counter % len(child_keys)]
                node._expand_counter += 1
                sim_node = child[idx]
                path.append(sim_node)
                reward = self._simulate(sim_node, n_rollouts=n_rollouts)
                return path, reward
        else:
                reward = compute_reward(node)
                return path, reward

    def _backpropagate(self, path, reward):
        for node in reversed(path):
            node.visits += 1
            node.total_reward += reward
            if node.is_terminal:
                node.reward = reward

    
    def sample_trajectories_with_calibration(self, num_samples: int, num_iterations: int = 1000, reset_mcts=False, show_progress=True, greedy=False) -> list:
        def compute_reward(node):
            sampled_ids = node.clusternode.cached_sampled_ids
            similarities = node.clusternode.cached_similarities
            if sampled_ids is None or similarities is None:
                logging.warning(f"Node {node.clusternode.indices[:5]} has None cached data, returning 0 reward")
                return 0
        if reset_mcts or self.mcts_root is None:
            self.mcts_root = self._make_node(self.clustertree.root)
        root_node = self.mcts_root
        # MCTS main loop
        if greedy:
            logging.info("Using greedy selection in MCTS")
        else:
            for it in range(num_iterations):
                path, reward = self._select_expand_simulate_path(root_node)
                self._backpropagate(path, reward)
                if show_progress and (it+1) % max(1, num_iterations//10) == 0:
                    logging.info(f"MCTS iteration {it+1}/{num_iterations}")
            # reward calibration
            def calibration_reward(node):
                if 'children' not in node.__dict__ or node.children.get('expand') is None:
                    return
                select_node = node.children.get('select')
                select_reward = getattr(select_node, 'reward', None)
                expand_nodes = node.children.get('expand').values()
                expand_rewards_list = []
                for expand_node in expand_nodes:
                    expand_select_node = expand_node.children.get('select')
                    expand_select_reward = getattr(expand_select_node, 'reward', None)
                    expand_rewards_list.append(expand_select_reward)
                if expand_rewards_list and select_reward is not None and all(r is not None for r in expand_rewards_list):
                    total_expand_reward = sum(expand_rewards_list)
                    for idx, expand_node in enumerate(expand_nodes):
                        expand_select_node = expand_node.children.get('select')
                        # calibration formula
                        expand_select_node.reward = select_reward/total_expand_reward * expand_rewards_list[idx] * len(select_node.clusternode.get_all_indices()) / len(expand_node.clusternode.get_all_indices())
                        # recursive update the parent node
                        update_reward_delta = (expand_select_node.reward - expand_rewards_list[idx]) * expand_select_node.visits
                        current_node = expand_node
                        while current_node.parent is not None:
                            current_node.parent.total_reward += update_reward_delta
                            current_node = current_node.parent
                        calibration_reward(expand_node)
            calibration_reward(root_node)
        # hierarchical sampling
        def sampling_from_mcts(node, depth=0):
            if 'children' not in node.__dict__ or node.children.get('expand') is None:
                if Config.trajsimi_measure in ["dtw", "hausdorff", "erp"]:
                # if Config.node_random_sampling:
                # if hasattr(Config, 'node_random_sampling') and Config.node_random_sampling:
                    available_trajectories = node.clusternode.get_all_indices()
                else:
                    available_trajectories = node.clusternode.cached_sampled_ids
                # else:
                    # available_trajectories = node.clusternode.get_all_indices()
                if available_trajectories is None:
                    logging.warning(f"Node {node.clusternode.indices[:5]} has None cached_sampled_ids, using all indices")
                    available_trajectories = node.clusternode.get_all_indices()
                if len(available_trajectories) >= self.target_size:
                    sampled_ids = random.sample(available_trajectories, self.target_size)
                else:
                    sampled_ids = available_trajectories
                return sampled_ids, depth
            select_node = node.children.get('select')
            expand_node = node.children.get('expand')
            if greedy:
                select_reward = compute_reward(select_node)
                expand_reward_list = [compute_reward(child)*len(child.clusternode.get_all_indices())/len(node.clusternode.get_all_indices()) for child in expand_node.values()]
            else:
                select_reward = getattr(select_node, 'reward', 0)
                expand_reward_list = [(child.total_reward/child.visits if child.visits > 0 else 0)*len(child.clusternode.get_all_indices())/len(node.clusternode.get_all_indices()) for child in expand_node.values()]
            total_expand_reward = sum(expand_reward_list)
            if total_expand_reward > 0:
                select_score = select_reward / (select_reward + total_expand_reward)
            else:
                select_score = 0.5
            if random.random() < select_score:
                return sampling_from_mcts(select_node, depth+1)
            else:
                expand_node_list = list(expand_node.values())
                if total_expand_reward > 0:
                    expand_node_list_score = [expand_reward_list[i]/total_expand_reward for i in range(len(expand_reward_list))]
                else:
                    expand_node_list_score = [1.0/len(expand_reward_list)] * len(expand_reward_list)
                expand_node_list_score_cumsum = np.cumsum(expand_node_list_score)
                random_value = random.random()
                for i in range(len(expand_node_list_score_cumsum)):
                    if random_value < expand_node_list_score_cumsum[i]:
                        return sampling_from_mcts(expand_node_list[i], depth+1)
                return sampling_from_mcts(expand_node_list[-1], depth+1)
            
        all_sampled_ids = []
        all_depths = []
        max_attempts = num_samples * 10  # prevent infinite loop
        attempts = 0
        while len(all_sampled_ids) < num_samples and attempts < max_attempts:
            try:
                sampled_ids, depth = sampling_from_mcts(root_node)
                if not sampled_ids:
                    logging.warning("sampling_from_mcts returned empty sampled_ids")
                    attempts += 1
                    continue
                sampled_id_idx = [i for i, tid in enumerate(sampled_ids) if tid not in all_sampled_ids]
                if len(sampled_id_idx) > 0:
                    all_sampled_ids.extend([sampled_ids[i] for i in sampled_id_idx])
                    all_depths.extend([depth] * len(sampled_id_idx))
                attempts += 1
            except Exception as e:
                logging.error(f"Error in sampling_from_mcts: {e}")
                attempts += 1
                continue
                
        if show_progress:
            logging.info(f"sampled {len(all_sampled_ids)} trajectories from MCTS")
            if all_depths:
                depth_distribution = np.bincount(all_depths)
                depth_distribution = depth_distribution / np.sum(depth_distribution)
                for i in range(len(depth_distribution)):
                    logging.info(f"sampling depth {i}: {depth_distribution[i]:.2f}")
            else:
                logging.info("no trajectories sampled")
                
        num_samples_int = int(num_samples)
        if len(all_sampled_ids) > num_samples_int:
            final_sampled_ids = random.sample(all_sampled_ids, num_samples_int)
        elif len(all_sampled_ids) < num_samples_int:
            all_ids = list(self.trajectories_data.keys())
            remain = [tid for tid in all_ids if tid not in all_sampled_ids]
            if remain:
                final_sampled_ids = all_sampled_ids + random.sample(remain, min(num_samples_int-len(all_sampled_ids), len(remain)))
            else:
                logging.warning("No remaining trajectories to sample from, returning what we have")
                final_sampled_ids = all_sampled_ids
        else:
            final_sampled_ids = all_sampled_ids
            
        if not final_sampled_ids:
            logging.error("No trajectories sampled, returning random subset")
            all_ids = list(self.trajectories_data.keys())
            final_sampled_ids = random.sample(all_ids, min(num_samples_int, len(all_ids)))
            
        return final_sampled_ids[:num_samples_int]


