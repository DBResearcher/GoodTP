import math
import numpy as np
import multiprocessing as mp
import traj_dist.distance as tdist
import sys
import os

# 添加core_cpu模块的路径
sys.path.append("/home/haitao/data/GoodST/code/baseline/TrajSimiMeasures")
import core_cpu
from functools import partial

from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
import sys

import redis, struct


# 配置 logging，输出到标准输出，并设置日志级别为 WARNING
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 获取 logger 对象
logger = logging.getLogger("my_fastapi_app")

# 定义 method 到单字节编码的映射
METHOD_MAP = {'lcss': 0, 'edr': 1, 
            'erp': 2, 'dtw': 3,
            'dfrechet': 4, 'hausdorff': 5,
            'stedr': 6, 'cdds': 7,
            'stlcss': 8
            }

# 定义 dataset 到单字节编码的映射
DATASET_MAP = {'xian': 0, 'porto': 1, 'geolife': 2, 'chengdu': 3, 'germany': 4}

# 全局 Redis 连接变量（在子进程中初始化）
redis_conn = None
# 全局轨迹数据变量（在主进程中设置，子进程共享）
global_lst_trajs = None
global_lst_traj_ids = None

def init_worker(fn_name):
    """
    ProcessPoolExecutor 初始化函数，在每个子进程中调用，
    初始化全局 Redis 连接，避免重复创建连接。
    """
    global redis_conn
    # 只使用 method 决定使用的 DB，不考虑 dataset
    db_key = METHOD_MAP.get(fn_name, 0)
    redis_conn = redis.Redis(host='localhost', port=6379, db=db_key)
    logger.info(f"Worker process initialized Redis connection for method={fn_name}, db={db_key}.")

def get_pair_key(traj_id1, traj_id2, dataset_id):
    """
    构造缓存 key：1 字节的 dataset 编码 + 两个 4 字节的轨迹ID（按字典序排序）
    使用4字节存储轨迹ID，因为数据量不超过一百万
    """
    # 按字典序排序，确保 key 的顺序固定
    if traj_id1 > traj_id2:
        traj_id1, traj_id2 = traj_id2, traj_id1
    # 使用 struct.pack 将数据打包成二进制：!B 表示1字节无符号整数，II 表示两个4字节无符号整数
    key = struct.pack('!BII', dataset_id, traj_id1, traj_id2)
    return key


def simi_comp_operator(fn_name, fn, lst_trajs, lst_traj_ids, dataset_id, sub_idx, expire_sec=3600000, use_cache=True):
    """
    对于 sub_idx 中的每个原始行索引 _i，计算其与后续所有行的相似度，采用批量 pipeline 方式：
      1. 收集所有需要计算的 pair key；
      2. 用 pipeline 批量 get 缓存结果；
      3. 对未命中缓存的对批量计算相似度，并用 pipeline 批量 setex 缓存；
      4. 根据 pair (i,j) 还原每行结果。
      
    返回格式: list of tuples: (row_index, row_result)，其中 row_result 是一个列表，
    包含该行与后续各行的相似度。
    """
    results = {}  # 用于存储每个 (i,j) 对应的相似度
    pair_list = []  # 列表中每个元素为 (i, j, key)
    l = len(lst_trajs)

    # 1. 收集所有待查询的 pair：遍历每个 _i in sub_idx, 对 _j 从 _i+1 到 l-1
    for _i in sub_idx:
        for _j in range(_i + 1, l):
            key = get_pair_key(lst_traj_ids[_i], lst_traj_ids[_j], dataset_id)
            pair_list.append((_i, _j, key))

    global redis_conn
    if redis_conn is None:
        # 防御性检查，只使用 method 决定 DB
        db_key = METHOD_MAP.get(fn_name, 0)
        redis_conn = redis.Redis(host='localhost', port=6379, db=db_key)

    # 2. 使用 pipeline 批量 get 所有 pair 的缓存结果
    if use_cache:
        pipe = redis_conn.pipeline()
        for (_, _, key) in pair_list:
            pipe.get(key)
        cached_results = pipe.execute()
    else:
        cached_results = [None] * len(pair_list)

    # 3. 遍历 pair_list 和对应的 cached_results
    missing = []  # 用于记录未命中的 (i, j, key)
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

    # 4. 对所有未命中的 pair 批量计算相似度
    if missing:
        # 逐个计算，若计算量大可以考虑并行，但此处为了简化批量写入，我们在同一进程中计算
        for (i, j, key) in missing:
            try:
                sim = fn(lst_trajs[i], lst_trajs[j])
                sim = round(sim, 2)
                results[(i, j)] = sim
            except Exception as e:
                logger.error(f"Error computing similarity between {i} and {j}: {e}")
                results[(i, j)] = 0.0

        # 5. 用 pipeline 批量 setex 未命中项到 Redis
        if use_cache:
            pipe = redis_conn.pipeline()
            for (i, j, key) in missing:
                if i==j or results[(i, j)] > 0:
                    pipe.setex(key, expire_sec, str(results[(i, j)]))
            pipe.execute()

    # 6. 组装每行结果：对于每个 _i in sub_idx，row_result 列表包含 _j 从 _i+1 到 l-1 的相似度
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
    计算相似度矩阵并返回 (二维 list 或 np.array)
    trajectories: list[list[float]] or list[list[list]] 视实际情况而定
    traj_ids: list[int] 轨迹ID列表
    dataset: str 数据集名称
    """
    fn = _get_simi_fn(fn_name)
    if fn is None:
        logger.error(f"Unsupported method: {fn_name}")
        raise ValueError(f"Unsupported method: {fn_name}")
    
    if dataset not in DATASET_MAP:
        logger.error(f"Unsupported dataset: {dataset}")
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    dataset_id = DATASET_MAP[dataset]
    
    # 提前将所有轨迹转换为 numpy 数组
    np_trajs = [np.asarray(traj, dtype=np.float64) for traj in trajectories]

    l = len(np_trajs)
    if l == 0:
        return []

    if not processes:
        # 默认用 CPU 核心数 - 3
        processes = 1 #max(mp.cpu_count() - 3, 1)

    # 按照 batch_size 把所有下标分块
    tasks = []
    total_batch = math.ceil(l / batch_size)
    for i in range(total_batch):
        start_i = batch_size * i
        end_i = min(batch_size * (i + 1), l)
        tasks.append((fn_name, fn, np_trajs, traj_ids, dataset_id, range(start_i, end_i), 3600000, use_cache))

    # 使用 ProcessPoolExecutor 运行任务
    results_with_idx = []
    with ProcessPoolExecutor(max_workers=processes, initializer=init_worker, initargs=(fn_name,), mp_context=mp.get_context("spawn")) as executor:
        future_to_task = {executor.submit(simi_comp_operator, *task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                results_with_idx.extend(future.result())
            except Exception as exc:
                logger.error(f"Task generated an exception: {exc}")

    # 根据行索引排序结果
    results_with_idx.sort(key=lambda x: x[0])
    # 提取有序的 row_result 列表
    ordered_results = [row_result for _, row_result in results_with_idx]

    # 构造对称矩阵
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
    批量计算多个轨迹组的相似度矩阵
    batch_trajectories: list[list[list[list[float]]]] 批量轨迹数据
    batch_traj_ids: list[list[int]] 批量轨迹ID数据
    dataset: str 数据集名称
    """
    if len(batch_trajectories) != len(batch_traj_ids):
        raise ValueError("batch_trajectories and batch_traj_ids must have the same length")
    
    # 如果计算st相似度，那么查看轨迹是否有第三维度，没有的话，补上第三维度
    if fn_name in ['stedr', 'stlcss', 'cdds']:
        if len(batch_trajectories[0][0][0]) < 3:
            new_batch_trajectories = []
            for trajectories, traj_ids in zip(batch_trajectories, batch_traj_ids):
                new_trajectories = []
                for traj in trajectories:
                    new_trajectories.append([[point[0], point[1], float(idx)] for idx, point in enumerate(traj)])
                new_batch_trajectories.append(new_trajectories)
            batch_trajectories = new_batch_trajectories
        
    # 并行处理每个batch
    # assert processes == 4
    with ProcessPoolExecutor(max_workers=processes) as executor:
        # 提交所有任务并保存future和对应的索引
        future_to_index = {}
        for i, (trajectories, traj_ids) in enumerate(zip(batch_trajectories, batch_traj_ids)):
            future = executor.submit(compute_simi_matrix, fn_name, trajectories, traj_ids, dataset, use_cache, batch_size, processes)
            future_to_index[future] = i
        
        # 初始化结果列表
        batch_results = [None] * len(batch_trajectories)
        
        # 按完成顺序收集结果，但保持原始顺序
        for future in as_completed(future_to_index):
            batch_index = future_to_index[future]
            try:
                result = future.result()
                # 验证返回的矩阵维度与输入batch长度一致
                expected_dim = len(batch_trajectories[batch_index])
                if result is not None:
                    if len(result) != expected_dim or (len(result) > 0 and len(result[0]) != expected_dim):
                        raise ValueError(f"Batch {batch_index}: Matrix dimension mismatch. Expected {expected_dim}x{expected_dim}, got {len(result)}x{len(result[0]) if result else 0}")
                batch_results[batch_index] = result
            except Exception as exc:
                logger.error(f"Batch {batch_index} computation generated an exception: {exc}")
                batch_results[batch_index] = None
    
    return batch_results